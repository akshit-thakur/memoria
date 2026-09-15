"""Worker heartbeat — atomic JSON file at ~/.memoria/worker_heartbeat.json.

Written by the Huey consumer process so the FastAPI /progress endpoint can
detect stalls without polling Postgres.  All writes are atomic (tmp + os.replace)
matching the existing current.txt / metrics-snapshot patterns.

Fields:
    pid                  — worker OS pid
    written_ts           — float unix timestamp of last write (refreshed every 30 s
                           by the background sample loop even while a task is stuck)
    current_file         — path being processed right now, or ""
    current_started_ts   — float unix ts when current_file was set, or null
    last_completed_ts    — float unix ts of the last successful completion, or null
    completed_since_start — int count of files completed since worker started
    paused_memory        — bool: true while sleeping due to RAM/VRAM pressure
    pressure             — "ok" | "warn" | "critical"
"""

import json
import os
import time
import logging

from runtime_paths import WORKER_HEARTBEAT_PATH

logger = logging.getLogger("memoria.worker.heartbeat")

_HEARTBEAT_PATH = WORKER_HEARTBEAT_PATH
_HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Module-level mutable state; only the single-threaded Huey consumer writes here.
_state: dict = {
    "pid": os.getpid(),
    "written_ts": time.time(),
    "current_file": "",
    "current_started_ts": None,
    "last_completed_ts": None,
    "completed_since_start": 0,
    "paused_memory": False,
    "pressure": "ok",
}


def touch(**updates) -> None:
    """Merge *updates* into the heartbeat state and atomically write the file."""
    _state.update(updates)
    _state["written_ts"] = time.time()
    _state["pid"] = os.getpid()
    _flush()


def _flush() -> None:
    tmp = _HEARTBEAT_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(_state))
        os.replace(tmp, _HEARTBEAT_PATH)
    except Exception as exc:
        logger.warning("Failed to write worker heartbeat: %s", exc)
