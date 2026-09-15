"""Environment-driven paths for Memoria's writable runtime state.

Native installs keep all writable state below ``~/.memoria`` when
``MEMORIA_STATE_DIR`` is unset. Container installs can set one persistent
state root (and, where useful, the narrower overrides) without changing
application code.

This module only resolves paths.  Callers that write a file are responsible
for creating its parent directory immediately before use.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    """Return an expanded path from *name*, falling back to *default*.

    Empty environment values are treated as unset.  Paths are deliberately
    not resolved here: container mount targets may not exist at import time,
    and resolving a symlink would make logs/configuration less predictable.
    """
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser() if raw else Path(default).expanduser()


STATE_DIR = _env_path("MEMORIA_STATE_DIR", Path.home() / ".memoria")
CACHE_DIR = _env_path("MEMORIA_CACHE_DIR", STATE_DIR / "cache")
EDITS_DIR = _env_path("MEMORIA_EDITS_DIR", STATE_DIR / "edits")
TRASH_DIR = _env_path("MEMORIA_TRASH_DIR", STATE_DIR / "trash")
METRICS_DIR = _env_path("MEMORIA_METRICS_DIR", STATE_DIR / "metrics")
MODELS_DIR = _env_path("MEMORIA_MODELS_DIR", STATE_DIR / "models")

CURRENT_FILE_PATH = STATE_DIR / "current.txt"
SCAN_SESSION_PATH = STATE_DIR / "scan_session.json"
WORKER_HEARTBEAT_PATH = STATE_DIR / "worker_heartbeat.json"
TRT_ENGINE_CACHE_DIR = CACHE_DIR / "trt_engines"

# Queue and scan databases are durable writable state too. Keeping them below
# the same default state root avoids writes into a source checkout and gives
# native and container runtimes the same override hierarchy.
DATA_DIR = _env_path("MEMORIA_DATA_DIR", STATE_DIR / "data")
QUEUE_DB_PATH = _env_path("HUEY_DB", DATA_DIR / "queue.db")
SCAN_INDEX_DB_PATH = _env_path(
    "MEMORIA_SCAN_INDEX_DB", DATA_DIR / "scan_index.db"
)

# Model downloads are writable runtime state. An explicit
# ``MEMORIA_YUNET_MODEL_PATH`` can still select a pre-provisioned checkpoint.
_yunet_default = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = _env_path("MEMORIA_YUNET_MODEL_PATH", _yunet_default)


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a conventional boolean environment variable."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def media_is_read_only() -> bool:
    """Whether API operations may mutate original library files."""
    return env_flag("MEMORIA_MEDIA_READ_ONLY", default=False)
