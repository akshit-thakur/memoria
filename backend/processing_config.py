"""Memoria — per-stage processing feature toggles + presets.

In plain English: which pipeline stages run is now driven by a row in the
``app_settings`` table (key ``processing_features``) instead of being hardwired.
The FastAPI server reads/writes that row over asyncpg; the separate Huey worker
process reads it here over psycopg with a short TTL cache so it doesn't hit the
DB on every single file.

Two non-obvious rules live here so the API and the worker stay in sync:
  * SSCD dedup reuses the cached thumbnail (see tasks.py compute_dedup_embedding),
    so enabling ``sscd_dedup`` forces ``thumbnail`` on — :func:`normalize`.
  * ``stat`` and ``sync`` are always-on (not in FEATURE_KEYS): they're the
    minimum needed for the row to exist and be reviewable in the UI.

If the worker can't read the row it falls back to all-on DEFAULTS *and* signals
the failure (``config_load_error``) so the throughput log can mark that run as
``fallback_full`` rather than silently claiming it ran the saved preset.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import psycopg
from database_config import DATABASE_URL

logger = logging.getLogger("memoria.processing_config")

SETTINGS_KEY = "processing_features"

# The toggleable stages. stat_file + sync_metadata are intentionally absent —
# they always run (minimum metadata required for the UI/review workflow).
FEATURE_KEYS: tuple[str, ...] = (
    "thumbnail",
    "exif",
    "phash",
    "blur",
    "faces",
    "scene_clip",
    "sscd_dedup",
)

# Default = everything on, i.e. the pipeline's behavior before toggles existed.
DEFAULTS: dict[str, bool] = {k: True for k in FEATURE_KEYS}

# Named presets. Each is a complete feature dict.
PRESETS: dict[str, dict[str, bool]] = {
    "full": dict(DEFAULTS),
    "dedup_first": {
        "thumbnail": True,
        "exif": False,
        "phash": False,
        "blur": False,
        "faces": False,
        "scene_clip": False,
        "sscd_dedup": True,
    },
    "semantic_search": {
        "thumbnail": True,
        "exif": False,
        "phash": False,
        "blur": False,
        "faces": False,
        "scene_clip": True,
        "sscd_dedup": False,
    },
    "face_indexing": {
        "thumbnail": True,
        "exif": False,
        "phash": False,
        "blur": False,
        "faces": True,
        "scene_clip": False,
        "sscd_dedup": False,
    },
}

_TTL_SECONDS = 5.0
_cache_lock = threading.Lock()
_cache: dict | None = None
_cache_ts: float = 0.0


def coerce_features(features: dict) -> dict[str, bool]:
    """Return a complete, well-typed feature dict with rule #1 applied.

    Missing keys fall back to DEFAULTS; unknown keys are dropped. If sscd_dedup
    is enabled, thumbnail is forced on (SSCD reuses the cached thumbnail).
    """
    out = {k: bool(features.get(k, DEFAULTS[k])) for k in FEATURE_KEYS}
    if out["sscd_dedup"]:
        out["thumbnail"] = True
    return out


def _match_preset(features: dict[str, bool]) -> str:
    for name, preset in PRESETS.items():
        if preset == features:
            return name
    return "custom"


def normalize(body: dict, current: dict | None = None) -> dict:
    """Resolve an API request body to a stored config dict.

    Accepts ``{"preset": name}`` and/or ``{"features": {key: bool}}``. A named
    preset seeds the features; explicit feature keys override on top. The result
    is coerced (rule #1) and re-labelled — if overrides diverge from the named
    preset the label becomes ``"custom"``.

    Raises ValueError on an unknown preset or unknown feature key.
    """
    preset = body.get("preset")
    features_override = body.get("features") or {}

    if preset is not None and preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset!r}")
    for k in features_override:
        if k not in FEATURE_KEYS:
            raise ValueError(f"unknown feature key: {k!r}")

    if preset is not None:
        base = dict(PRESETS[preset])
    elif current is not None:
        # Feature updates from the UI are intentionally sparse.  Seed them
        # from the stored configuration so toggling one stage cannot reset all
        # unrelated stages to DEFAULTS.
        current_features = current.get("features", current)
        base = coerce_features(current_features if isinstance(current_features, dict) else {})
    else:
        base = dict(DEFAULTS)
    base.update({k: bool(v) for k, v in features_override.items()})
    features = coerce_features(base)
    return {"preset": _match_preset(features), "features": features}


def invalidate_cache() -> None:
    """Drop the worker-side TTL cache (e.g. right after a write in-process)."""
    global _cache, _cache_ts
    with _cache_lock:
        _cache = None
        _cache_ts = 0.0


def load_features() -> dict:
    """Worker-side read of the processing config (psycopg, TTL-cached).

    Returns ``{"preset", "features", "config_load_error"}``. On any DB/parse
    failure logs a WARN once and returns all-on DEFAULTS with
    ``config_load_error=True`` and ``preset="fallback_full"`` so callers can mark
    the run accordingly.
    """
    global _cache, _cache_ts
    now = time.monotonic()
    with _cache_lock:
        if _cache is not None and (now - _cache_ts) < _TTL_SECONDS:
            return _cache

    result: dict
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value FROM app_settings WHERE key = %s",
                    (SETTINGS_KEY,),
                )
                row = cur.fetchone()
        if not row or not isinstance(row[0], dict):
            # Row missing (pre-seed) → defaults, but NOT an error.
            features = coerce_features({})
            result = {"preset": _match_preset(features), "features": features,
                      "config_load_error": False}
        else:
            raw = row[0]
            features = coerce_features(raw.get("features", {}))
            preset = raw.get("preset") or _match_preset(features)
            result = {"preset": preset, "features": features,
                      "config_load_error": False}
    except Exception as exc:
        logger.warning("processing_config: load failed, using all-on defaults: %s", exc)
        features = dict(DEFAULTS)
        result = {"preset": "fallback_full", "features": features,
                  "config_load_error": True}

    with _cache_lock:
        _cache = result
        _cache_ts = time.monotonic()
    return result
