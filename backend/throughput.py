"""Memoria — persistent, append-only throughput logging.

In plain English: the in-process metrics (``metrics.py``) are a live, bounded
sliding window — great for "what's p95 right now", useless for "how many GB/hour
did last night's run sustain". This module writes one JSON line per processing
job to ``~/.memoria/metrics/processing_runs.jsonl`` so throughput survives across
restarts and can be charted later.

One record per ``process_media_batch`` run, and one per file in single-file mode.
Stage timings that aren't measurable for a given path are omitted rather than
faked. Writes never raise — a logging failure must not kill the pipeline.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import metrics  # reuse SNAPSHOT_DIR so MEMORIA_METRICS_DIR controls both

logger = logging.getLogger("memoria.throughput")

RUNS_PATH = metrics.SNAPSHOT_DIR / "processing_runs.jsonl"

# Stable per-process id so all records from one worker run share a run_id.
RUN_ID = uuid.uuid4().hex[:12]

_write_lock = threading.Lock()
_warned = False


def new_batch_id() -> str:
    return uuid.uuid4().hex[:12]


def build_record(
    *,
    mode: str,
    files: int,
    total_bytes: int,
    total_sec: float,
    enabled_features: dict,
    stage_sec: dict | None = None,
    config_load_error: bool = False,
    batch_id: str | None = None,
) -> dict:
    """Assemble a throughput record. Derived rates are 0.0 when total_sec<=0."""
    gb = total_bytes / 1e9
    per_hour = (files / total_sec * 3600.0) if total_sec > 0 else 0.0
    gb_per_hour = (gb / total_sec * 3600.0) if total_sec > 0 else 0.0
    # Drop unmeasured (None) stage timings instead of writing nulls/zeros.
    clean_stage = {k: round(v, 4) for k, v in (stage_sec or {}).items() if v is not None}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "batch_id": batch_id or new_batch_id(),
        "mode": mode,
        "config_load_error": bool(config_load_error),
        "files": int(files),
        "bytes": int(total_bytes),
        "gb": round(gb, 6),
        "total_sec": round(total_sec, 4),
        "files_per_hour": round(per_hour, 2),
        "gb_per_hour": round(gb_per_hour, 4),
        "enabled_features": dict(enabled_features),
        "stage_sec": clean_stage,
    }


def log_run(record: dict) -> None:
    """Append one record as a JSON line. Never raises."""
    global _warned
    try:
        RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(",", ":"))
        with _write_lock:
            with open(RUNS_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:
        if not _warned:
            _warned = True
            logger.warning("throughput: failed to append run record: %s", exc)


def read_recent(limit: int = 20) -> dict:
    """Return the most recent records (newest first) + the single latest one.

    ``{"latest": {...}|None, "runs": [...]}``. Tails the file; tolerant of
    partial/corrupt trailing lines.
    """
    limit = max(1, min(int(limit), 500))
    runs: deque[dict] = deque(maxlen=limit)
    try:
        if RUNS_PATH.exists():
            with open(RUNS_PATH, "r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        runs.append(json.loads(raw))
                    except json.JSONDecodeError:
                        continue
    except OSError as exc:
        logger.warning("throughput: failed to read runs: %s", exc)
    ordered = list(reversed(runs))  # newest first
    return {"latest": ordered[0] if ordered else None, "runs": ordered}


class StageClock:
    """Tiny additive timer accumulator for building ``stage_sec``.

    Thread-safe so the double-buffered batch pipeline (halves A/B running
    concurrently) can both contribute to the same dict. In pipelined mode the
    summed phase-seconds may exceed wall-clock ``total_sec`` because phases
    overlap — that's intentional, ``total_sec`` is the authoritative wall time.
    """

    def __init__(self) -> None:
        self._d: dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._d[stage] = self._d.get(stage, 0.0) + float(seconds)

    def time(self, stage: str):
        clock = self

        class _Ctx:
            def __enter__(self):
                self._t0 = time.perf_counter()
                return self

            def __exit__(self, *exc):
                clock.add(stage, time.perf_counter() - self._t0)
                return False

        return _Ctx()

    def as_dict(self) -> dict[str, float]:
        with self._lock:
            return dict(self._d)
