"""Memoria — in-process metrics (counters + histograms).

In plain English: a tiny pure-Python alternative to StatsD/Prometheus.
The pipeline records "how long did this step take?" and "how many photos
did we drop because of a decoder error?" as it runs; the FastAPI server
serves those numbers at ``/api/v1/metrics``; ``backend/bench.py`` parses
them into the markdown that PERFORMANCE.md is meant to contain.

We don't use a real metrics library because the constraints are unusual:
single-machine, no network, no extra process, three different runtime
contexts (FastAPI's asyncio loop, the Huey worker, the watchdog observer)
all writing into the same store.

Zero-dependency, thread-safe. Designed for the FastAPI async server, the
single-thread Huey worker, and the watchdog observer threads to share one
in-memory store. Histograms keep a bounded sliding window (last 2048
samples) and compute percentiles on demand via :mod:`statistics`.

Public API
----------
``timer(name, **tags)`` — context manager; records elapsed ms.
``count(name, n=1, **tags)`` — bumps a counter.
``record(name, value, **tags)`` — pushes a raw sample into a histogram.
``snapshot()`` — full state for the ``/api/v1/metrics`` endpoint.
``log_summary(logger)`` — one-line per series flushed at INFO.

Series keys are flat strings ``"name|tag=value|tag=value"`` with tags
sorted alphabetically — keeps a future ``?format=prometheus`` exporter
trivial without nested dicts.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from runtime_paths import METRICS_DIR

_logger = logging.getLogger("memoria.metrics")

_MAX_SAMPLES = 2048
# Default raised from 256 → 512 to accommodate the comprehensive BENCH-PROBE
# instrumentation (system resource sampling, per-stage GPU probes, per-extension
# file counters, watcher scan stats). Override via MEMORIA_METRICS_MAX_SERIES.
_MAX_SERIES = int(os.getenv("MEMORIA_METRICS_MAX_SERIES", "512"))
_MAX_TAG_LEN = 64


def _format_key(name: str, tags: dict[str, object]) -> str:
    """Compose a flat metric key like ``"api.request.duration_ms|route=/photos|method=GET"``.

    Tags are sorted alphabetically so identical (name, tags) pairs always
    produce the exact same string regardless of caller insertion order.
    Bar characters and unprintables are stripped from values so they
    can't break the parser that splits the key later.
    """
    if not tags:
        return name
    parts: list[str] = []
    for key in sorted(tags):
        raw = tags[key]
        value = "" if raw is None else str(raw)
        # Strip control chars + bar separator collisions.
        value = "".join(ch for ch in value if ch.isprintable() and ch != "|")
        if len(value) > _MAX_TAG_LEN:
            value = value[:_MAX_TAG_LEN]
        parts.append(f"{key}={value}")
    return name + "|" + "|".join(parts)


class _Histogram:
    """One histogram series. Backed by a fixed-size deque so memory usage
    is bounded regardless of how many samples arrive. When the deque is
    full, oldest entries are evicted automatically (sliding window).
    Thread-safe via an internal lock so the worker and the API can both
    push samples concurrently."""

    __slots__ = ("samples", "lock", "last_ts")

    def __init__(self) -> None:
        self.samples: deque[float] = deque(maxlen=_MAX_SAMPLES)
        self.lock = threading.Lock()
        self.last_ts: float = 0.0

    def push(self, value: float) -> None:
        with self.lock:
            self.samples.append(float(value))
            self.last_ts = time.time()

    def snapshot(self) -> dict:
        with self.lock:
            data = list(self.samples)
            last = self.last_ts
        n = len(data)
        if n == 0:
            return {"n": 0}
        out: dict = {
            "n": n,
            "mean": sum(data) / n,
            "min": min(data),
            "max": max(data),
            "last_ts": last,
        }
        if n >= 2:
            try:
                # `statistics.quantiles(n=100)` returns the 99 cut points
                # that split the sorted data into 100 equal slices. Index
                # 49 is the median (p50), 94 is p95, 98 is p99 — these are
                # the percentile labels we expose in the API response.
                qs = statistics.quantiles(data, n=100, method="inclusive")
                # quantiles returns 99 cut points; index 49 ~ p50, 94 ~ p95, 98 ~ p99.
                out["p50"] = qs[49]
                out["p95"] = qs[94]
                out["p99"] = qs[98]
            except statistics.StatisticsError:
                pass
        else:
            out["p50"] = out["p95"] = out["p99"] = data[0]
        return out


class _Counter:
    __slots__ = ("value", "lock", "last_ts")

    def __init__(self) -> None:
        self.value: int = 0
        self.lock = threading.Lock()
        self.last_ts: float = 0.0

    def add(self, n: int) -> None:
        with self.lock:
            self.value += int(n)
            self.last_ts = time.time()

    def snapshot(self) -> dict:
        with self.lock:
            return {"value": self.value, "last_ts": self.last_ts}


_histograms: dict[str, _Histogram] = {}
_counters: dict[str, _Counter] = {}
_registry_lock = threading.Lock()
_overflow_warned: set[str] = set()


def _series_overflow(kind: str, key: str) -> bool:
    """Return True when a new series cannot be created. Logs once per kind."""
    total = len(_histograms) + len(_counters)
    if total < _MAX_SERIES:
        return False
    if kind not in _overflow_warned:
        _overflow_warned.add(kind)
        _logger.warning(
            "metrics: series cap (%d) reached, dropping new %s key %r",
            _MAX_SERIES, kind, key,
        )
    return True


def _get_histogram(key: str) -> _Histogram | None:
    h = _histograms.get(key)
    if h is not None:
        return h
    with _registry_lock:
        h = _histograms.get(key)
        if h is None:
            if _series_overflow("histogram", key):
                return None
            h = _Histogram()
            _histograms[key] = h
    return h


def _get_counter(key: str) -> _Counter | None:
    c = _counters.get(key)
    if c is not None:
        return c
    with _registry_lock:
        c = _counters.get(key)
        if c is None:
            if _series_overflow("counter", key):
                return None
            c = _Counter()
            _counters[key] = c
    return c


@contextmanager
def timer(name: str, **tags: object) -> Iterator[dict]:
    """Record elapsed milliseconds into the histogram named ``name``.

    Yields a small dict so callers can stash extra tags after entering — the
    final flush still uses the tags supplied at ``__enter__`` time.
    """
    key = _format_key(name, tags)
    state = {"start": time.perf_counter()}
    try:
        yield state
    finally:
        elapsed_ms = (time.perf_counter() - state["start"]) * 1000.0
        h = _get_histogram(key)
        if h is not None:
            h.push(elapsed_ms)


def count(name: str, n: int = 1, **tags: object) -> None:
    key = _format_key(name, tags)
    c = _get_counter(key)
    if c is not None:
        c.add(n)


def record(name: str, value: float, **tags: object) -> None:
    key = _format_key(name, tags)
    h = _get_histogram(key)
    if h is not None:
        h.push(float(value))


def snapshot() -> dict:
    """Full state suitable for the ``/api/v1/metrics`` endpoint."""
    hist = {}
    for key, h in list(_histograms.items()):
        hist[key] = h.snapshot()
    ctr = {}
    for key, c in list(_counters.items()):
        ctr[key] = c.snapshot()
    return {
        "histograms": hist,
        "counters": ctr,
        "series_count": len(hist) + len(ctr),
        "series_cap": _MAX_SERIES,
    }


def reset() -> None:
    """Wipe all series. Useful for benchmarks."""
    with _registry_lock:
        _histograms.clear()
        _counters.clear()
        _overflow_warned.clear()


# ── Cross-process snapshot bridge ───────────────────────────────────────────
# The watcher, the Huey worker, and the FastAPI server all live in separate
# OS processes, so each has its own in-memory store. Worker-side timings
# (face detection, scene encoding, sync_metadata) would otherwise be invisible
# to the /api/v1/metrics endpoint. To bridge the gap, processes that aren't
# the API write their snapshot atomically to a shared directory; the API
# endpoint merges those JSON files with its own in-memory snapshot.

SNAPSHOT_DIR = METRICS_DIR


def write_snapshot_file(name: str, directory: Path | None = None) -> Path | None:
    """Atomically write the current snapshot to ``<directory>/<name>.json``.

    Used by long-running side-process workers (Huey consumer, watcher) so
    the FastAPI /api/v1/metrics endpoint can surface their numbers. Returns
    the written path, or None on failure (failures are logged once at WARN
    so we don't drown logs in repeated I/O errors).
    """
    target_dir = Path(directory) if directory else SNAPSHOT_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _logger.warning("metrics: cannot create snapshot dir %s: %s", target_dir, exc)
        return None

    payload = snapshot()
    payload["_pid"] = os.getpid()
    payload["_name"] = name
    payload["_written_ts"] = time.time()

    final = target_dir / f"{name}.json"
    tmp = target_dir / f"{name}.json.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        os.replace(tmp, final)
        return final
    except OSError as exc:
        _logger.warning("metrics: snapshot write failed (%s): %s", final, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def read_snapshots_dir(directory: Path | None = None) -> dict[str, dict]:
    """Read every ``*.json`` snapshot file under *directory*.

    Returns ``{name: snapshot_dict}``. Files with a ``_written_ts`` older
    than 5 minutes are skipped (the producing process is probably down).
    Used by the API's ``/api/v1/metrics`` endpoint to merge worker- and
    watcher-side metrics into the response.
    """
    target_dir = Path(directory) if directory else SNAPSHOT_DIR
    if not target_dir.exists():
        return {}
    out: dict[str, dict] = {}
    cutoff = time.time() - 300.0
    for path in target_dir.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("_written_ts", 0) < cutoff:
            continue
        name = data.get("_name") or path.stem
        out[name] = data
    return out


class PeriodicSnapshotter:
    """Thread-based periodic snapshot writer.

    Used by long-running side processes (Huey worker, watcher) to keep their
    metrics file fresh without coupling the main code to a scheduler. ``stop()``
    flushes once more before exiting.
    """

    def __init__(self, name: str, interval_s: float = 30.0, directory: Path | None = None):
        self._name = name
        self._interval = interval_s
        self._dir = directory
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name=f"metrics-flush-{self._name}", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # One final flush so the last samples aren't lost.
        write_snapshot_file(self._name, self._dir)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            write_snapshot_file(self._name, self._dir)


def log_summary(target_logger: logging.Logger | None = None, top_n: int = 20) -> None:
    """Emit a compact one-line-per-series summary at INFO level.

    Logged through ``memoria.metrics`` by default — each entry-point's own
    ``logging`` config routes the line.
    """
    log = target_logger or _logger
    snap = snapshot()
    hist = snap["histograms"]
    ctr = snap["counters"]
    if not hist and not ctr:
        return

    log.info("[METRIC] series=%d (cap=%d)", snap["series_count"], snap["series_cap"])

    # Histograms — show the busiest first.
    items = sorted(hist.items(), key=lambda kv: kv[1].get("n", 0), reverse=True)
    for key, s in items[:top_n]:
        if s.get("n", 0) == 0:
            continue
        log.info(
            "[METRIC] %s n=%d mean=%.2fms p50=%.2f p95=%.2f p99=%.2f max=%.2f",
            key,
            s["n"],
            s.get("mean", 0.0),
            s.get("p50", 0.0),
            s.get("p95", 0.0),
            s.get("p99", 0.0),
            s.get("max", 0.0),
        )

    items = sorted(ctr.items(), key=lambda kv: kv[1]["value"], reverse=True)
    for key, s in items[:top_n]:
        log.info("[METRIC] %s count=%d", key, s["value"])
