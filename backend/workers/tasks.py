"""Memoria – GPU Worker Pipeline Tasks.

In plain English: this is the heart of the system. Every photo the
watcher discovers ends up here as a Huey task. The pipeline runs eight
steps in order:

   1. **Thumbnail**  — pre-generate the 240×240 cached JPEG so the UI
                       has something to show immediately.
   2. **File stat**  — record file size + mime type.
   3. **EXIF**       — pull camera metadata; convert GPS DMS → decimal.
   4. **pHash**      — a 64-bit perceptual hash via numpy DCT for
                       exact-duplicate grouping.
   5. **Blur**       — 6-model ensemble score, 0 (sharp) → 1 (blurry).
   6. **Faces**      — configurable consensus detection + a registered
                       embedding driver (one vector per face).
   7. **Scene**      — registered image/text-model embedding for search.
   8. **Dedup**      — SSCD embedding (512-d) for near-duplicate
                       grouping.
   9. **Sync**       — two-phase write: sidecar JSON marked "pending",
                       Postgres upsert, then sidecar rewritten to
                       "synced". A crash between phases is recoverable.

Two entry points:
  • ``process_media(path)``         — the safety-net path; one file per
                                       task when batching is explicitly 1.
  • ``process_media_batch(paths)``  — the high-throughput path; CPU and
                                       GPU stages overlap on different
                                       halves of the batch. Enabled by
                                       ``MEMORIA_BATCH_SIZE > 1``.

Both entry points are ``@huey.task`` registrations; the Huey consumer runs one
task at a time to avoid duplicate model loads and bound memory pressure across
deployment sizes. ``vram_breathe()`` clears an available
accelerator cache and gates batch size and sleep duration against live RSS;
see ``backend/memory_mgmt.py``.
"""

import gc
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from workers import huey

# Shared helpers live in backend/ — add it to the path so these imports
# work whether tasks.py is loaded as workers.tasks (via huey_consumer) or
# imported directly from scripts.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from image_utils import (
    THUMB_CACHE_DIR,
    decode_cv2_bgr,
    decode_pil,
    make_thumbnail,
)
from database_config import DATABASE_URL
from memory_mgmt import (
    gpu_cache_clear as _gpu_cache_clear,
    monitor as _monitor,
    vram_breathe as _vram_breathe,
)
from runtime_paths import CURRENT_FILE_PATH, QUEUE_DB_PATH
from sidecar import (
    get_lock as _sidecar_lock,
    load_sidecar as _load_sidecar,
    save_sidecar as _save_sidecar,
)

import metrics
import processing_config
import throughput
from workers import heartbeat as _heartbeat


# Per-stage accelerator telemetry feeds diagnostics and benchmark reports.
def _record_gpu_telemetry(stage: str) -> None:
    """Record current CUDA allocation in a stage-tagged histogram."""
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            used_mb = _torch.cuda.memory_allocated() / (1024 * 1024)
            metrics.record("worker.gpu.allocated_mb", used_mb, stage=stage)
    except Exception:
        pass


@huey.task()
def reset_metrics():
    """Wipe the worker process's in-memory metrics + delete the snapshot file.

    Triggered by ``POST /api/v1/metrics/reset`` for benchmark runs. The
    snapshot deletion stops a stale file from polluting the next ``/metrics``
    response until the next periodic flush rewrites it.
    """
    metrics.reset()
    snap = metrics.SNAPSHOT_DIR / "worker.json"
    try:
        snap.unlink(missing_ok=True)
    except OSError:
        pass

def _scene_reindex_batch_size() -> int:
    """Return the configured scene migration batch within its safety cap."""
    try:
        value = int(os.getenv("MEMORIA_SCENE_REINDEX_BATCH_SIZE", "16"))
    except (TypeError, ValueError):
        return 16
    return value if 1 <= value <= 64 else 16

# Module-level Postgres connection singleton.
# psycopg3 connections are not thread-safe; the Huey consumer is
# single-threaded so one connection is sufficient and avoids ~5 new
# handshakes per file at 300k-item scale.
_pg_conn: "psycopg.Connection | None" = None
_pg_lock = threading.Lock()
_FACE_MODEL_SQL_LITERALS = {
    "face/insightface-buffalo-l-arcface": (
        "'face/insightface-buffalo-l-arcface'"
    ),
    "face/facenet-vggface2": "'face/facenet-vggface2'",
    "face/legacy-unknown-512": "'face/legacy-unknown-512'",
}


def _get_pg() -> "psycopg.Connection":
    """Return the singleton psycopg3 connection, reconnecting if needed."""
    global _pg_conn
    with _pg_lock:
        if _pg_conn is None or _pg_conn.closed:
            _pg_conn = psycopg.connect(DATABASE_URL, autocommit=True)
            return _pg_conn
        try:
            _pg_conn.execute("SELECT 1")
        except Exception:
            try:
                _pg_conn.close()
            except Exception:
                pass
            _pg_conn = psycopg.connect(DATABASE_URL, autocommit=True)
        return _pg_conn


def _persist_scene_embedding(conn, media_id: int, payload: dict) -> None:
    """Validate and persist a model-versioned scene vector, if one was produced."""

    embedding = payload.get("scene_embedding")
    descriptor = payload.get("scene_model")
    if embedding is None or not descriptor:
        return

    import model_store

    model_key = descriptor["model_key"]
    # Registration is idempotent and deliberately checked on every write.
    # A surrounding media transaction may roll back, so a process-local
    # "seen" cache could otherwise outlive the catalog row it represented.
    model_store.register_model(conn, descriptor)
    runtime = payload.get("scene_runtime") or {}
    model_store.upsert_scene_embedding(
        conn,
        media_id=media_id,
        model_key=model_key,
        embedding=embedding,
        runtime_provider=runtime.get("actual_provider", "unknown"),
        runtime_precision=runtime.get("actual_precision", "unknown"),
    )


logger = logging.getLogger("memoria.worker")


def _get_ingestion_scene_engine():
    """Resolve ingestion against the durable desired scene namespace.

    Search remains pinned to the active namespace until a complete re-index
    promotes the desired model.  Reading desired state here lets media that
    arrives during a migration contribute to the target namespace instead of
    silently falling back to the process environment.  A genuinely absent
    state is only expected during first-boot startup, where the configured
    descriptor is the safe bootstrap identity.
    """
    import model_store
    from scene_engine import configured_model_spec, get_engine

    state = model_store.get_model_state(_get_pg(), "scene")
    descriptor = state.get("desired_model") if state else None
    if descriptor is None:
        logger.warning(
            "Scene model state is not initialized; using the configured "
            "bootstrap descriptor"
        )
        descriptor = configured_model_spec()
    return get_engine(descriptor)

# Pagination chunk for admin backfill tasks (reprocess_faces, rescore_blur).
# Keeps Python RSS bounded at 70k+ rows.
_ADMIN_CHUNK_SIZE = 1000

# Max Pillow thumbnail/preview concurrency per batch (Spec 01 B4).
_PILLOW_BATCH_CAP = 10

# Task counter – tracks cumulative files processed (thread-safe).
_task_counter = 0
_task_lock = threading.Lock()


def _get_queue_size() -> int:
    """Best-effort count of pending tasks in the Huey SQLite queue.

    Used by ``process_media`` for the ``(count/total)`` progress log line.
    Returns 0 on any error so logging never crashes the pipeline.
    """
    try:
        with sqlite3.connect(str(QUEUE_DB_PATH), timeout=5) as con:
            row = con.execute("SELECT COUNT(*) FROM task").fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0

# Path written by the worker so the FastAPI stats endpoint can report it.
_CURRENT_FILE_PATH = CURRENT_FILE_PATH
_CURRENT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Thumbnail Pre-generation (fast, runs first) ─────────────────────────────
def generate_thumbnail(file_path: str):
    """Pre-generate the cached 240×240 JPEG thumbnail so it is ready
    before the heavier analysis pipeline completes."""
    import hashlib
    from constants import is_image

    if not is_image(file_path):
        return  # video / unknown – nothing to pre-generate
    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(file_path.encode()).hexdigest()
    cached = THUMB_CACHE_DIR / f"{cache_key}.jpg"
    if cached.exists():
        metrics.count("worker.thumbnail.cached")
        return  # already done
    logger.info("Generating thumbnail: %s", file_path)
    with metrics.timer("worker.thumbnail"):
        make_thumbnail(file_path, cached)
    _vram_breathe()


# ── File stat (populates file_size + mime_type during sync_metadata) ────────

def stat_file(file_path: str) -> dict:
    """Return {file_size, mime_type} for a file on disk. Both may be None."""
    import mimetypes
    with metrics.timer("worker.stat"):
        file_size = None
        try:
            file_size = os.stat(file_path).st_size
        except OSError:
            pass
        mime_type, _ = mimetypes.guess_type(file_path)
    return {"file_size": file_size, "mime_type": mime_type}


# ── Step A: EXIF Extraction ─────────────────────────────────────────────────

def _exifread_ratio_to_float(r) -> float:
    if hasattr(r, "num") and hasattr(r, "den") and r.den:
        return float(r.num) / float(r.den)
    return float(r)


def _extract_gps(tags) -> dict | None:
    """Pull decimal lat/lon from exifread GPS tags. Returns None if absent.

    In plain English: cameras store GPS as degrees + minutes + seconds
    plus a separate "N/S" / "E/W" reference. We convert DMS to a single
    signed decimal degree, then sanity-check the result so we don't write
    impossible coordinates to the database. The (0, 0) check catches a
    common "GPS was never set" garbage value (the literal middle of the
    Atlantic Ocean).
    """
    lat_ratio = tags.get("GPS GPSLatitude")
    lat_ref = tags.get("GPS GPSLatitudeRef")
    lon_ratio = tags.get("GPS GPSLongitude")
    lon_ref = tags.get("GPS GPSLongitudeRef")
    if not (lat_ratio and lon_ratio):
        return None
    try:
        def _dms_to_deg(val) -> float:
            # Standard DMS → decimal degrees formula:
            #   d + (m / 60) + (s / 3600)
            d, m, s = [_exifread_ratio_to_float(x) for x in val.values]
            return d + m / 60.0 + s / 3600.0

        lat = _dms_to_deg(lat_ratio)
        lon = _dms_to_deg(lon_ratio)
        # Flip the sign for southern / western hemispheres.
        if lat_ref and str(lat_ref).strip().upper() == "S":
            lat = -lat
        if lon_ref and str(lon_ref).strip().upper() == "W":
            lon = -lon
        # Sanity bounds — reject clearly invalid values (0,0 is typically garbage).
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            return None
        return {"lat": lat, "lng": lon}
    except Exception:
        # Any parse failure: silently drop GPS, the rest of EXIF survives.
        return None


def extract_exif(file_path: str) -> dict:
    """Extract EXIF metadata from the media file using ``exifread``.

    Stores all string-form tags under ``exif``. If GPS tags are present,
    also writes decimal ``_gps_lat`` / ``_gps_lng`` for direct JSONB querying.
    """
    import exifread

    logger.info("Extracting EXIF: %s", file_path)
    exif: dict = {}
    with metrics.timer("worker.exif"):
        try:
            with open(file_path, "rb") as f:
                tags = exifread.process_file(f, details=False)
            exif = {k: str(v) for k, v in tags.items()}
            gps = _extract_gps(tags)
            if gps:
                exif["_gps_lat"] = str(gps["lat"])
                exif["_gps_lng"] = str(gps["lng"])
        except Exception as exc:
            logger.warning("EXIF extraction failed for %s: %s", file_path, exc)
            metrics.count("worker.exif.error")
    return {"file_path": file_path, "exif": exif}


# ── Step B: Perceptual Hash (pHash) via DCT ──────────────────────────────────
def compute_phash(payload: dict) -> dict:
    """Compute a 64-bit perceptual hash (pHash) via DCT for deduplication.

    Algorithm:
      1. Load image → grayscale → resize to 32×32.
      2. 2D DCT-II via numpy matrix multiply (fast on small arrays).
      3. Take the top-left 8×8 low-frequency block.
      4. Compare each coefficient to the median (excluding DC) → 64-bit hash.
      5. Return as a 16-char lowercase hex string.

    Graceful fallback: returns "" on any failure so the rest of the pipeline
    (blur, faces, sync) still runs.
    """
    file_path = payload["file_path"]
    logger.info("Computing pHash: %s", file_path)

    phash = ""
    with metrics.timer("worker.phash"):
        try:
            import numpy as np
            from PIL import Image

            # pillow-heif is registered globally in image_utils.py; rawpy + HEIC
            # are handled inside decode_pil. No per-call registration needed.
            img = decode_pil(file_path)
            # Cache full-res PIL for detect_faces to reuse (avoids a second
            # rawpy decode on RAW/HEIC, which costs 2-3 s per file).
            payload["_pil_full"] = img
            # Down-sample to 32×32 greyscale. Greyscale is the standard
            # input for pHash because colour doesn't change perceptual
            # similarity in the way that brightness does.
            img = img.convert("L").resize((32, 32), Image.LANCZOS)
            arr = np.asarray(img, dtype=np.float32)

            # 2D DCT-II via matrix multiply. The standard textbook
            # implementation iterates two nested loops; we build the basis
            # matrix once and use `basis @ arr @ basis.T`, which is a
            # single BLAS call. For tiny 32×32 arrays this is ~50× faster
            # than scipy.fft.dct on the same data.
            N = arr.shape[0]
            n = np.arange(N)
            basis = np.cos(np.pi * (2 * n + 1) * n[:, None] / (2 * N)) * np.sqrt(2.0 / N)
            basis[0] *= 1.0 / np.sqrt(2.0)
            dct = basis @ arr @ basis.T

            # Take the top-left 8×8 block — these are the lowest-frequency
            # DCT coefficients, which capture the "rough shape" of the image
            # and ignore fine detail. The DC component (index 0) is the
            # overall brightness and is excluded from the median so the hash
            # is brightness-invariant.
            low = dct[:8, :8].flatten()
            median = float(np.median(low[1:]))  # exclude DC component
            # Each coefficient contributes one bit: 1 if it's above the
            # median, 0 otherwise. Sixty-four coefficients → 64-bit hash →
            # 16 hex characters.
            h = 0
            for i, v in enumerate(low):
                if v > median:
                    h |= (1 << i)
            phash = f"{h:016x}"
        except Exception as exc:
            logger.warning("pHash computation failed for %s: %s", file_path, exc)
            metrics.count("worker.phash.error")

    payload["phash"] = phash
    return payload


# ── Step C: GPU-Accelerated Blur Detection ───────────────────────────────────
def detect_blur(payload: dict) -> dict:
    """Compute blur score using 6-model weighted ensemble (blur_detect module)."""
    file_path = payload["file_path"]
    logger.info("Detecting blur: %s", file_path)

    if _monitor.pressure() == "critical":
        logger.warning("Skipping blur detection (RAM critical): %s", file_path)
        metrics.count("worker.blur.skipped_ram")
        payload["blur"] = 0.0
        return payload

    with metrics.timer("worker.blur"):
        try:
            from scripts.blur_detection.blur_detect import score_image
            blur_score = score_image(file_path)
        except Exception as exc:
            logger.error("Blur detection failed for %s: %s", file_path, exc)
            metrics.count("worker.blur.error")
            blur_score = 0.0

    payload["blur"] = blur_score
    _vram_breathe()
    return payload


# ── Step D: Face Detection & Embedding ───────────────────────────────────────
def detect_faces(payload: dict) -> dict:
    """Run multi-model ensemble face detection via face_engine shim.

    Populates payload["faces"] with consensus-filtered detections.
    Each entry: {"bbox": {x,y,w,h}, "confidence": float,
                 "detection_model": str, "embedding": [512 floats]}.
    Sidecar gets bboxes only (embeddings are too large for JSON).
    """
    file_path = payload["file_path"]
    logger.info("Detecting faces: %s", file_path)

    if _monitor.pressure() == "critical":
        logger.warning("Skipping face detection (RAM critical): %s", file_path)
        metrics.count("worker.faces.skipped_ram")
        payload["faces"] = []
        payload["faces_complete"] = False
        return payload

    with metrics.timer("worker.faces"):
        try:
            from face_engine import get_detector

            try:
                cached_pil = payload.pop("_pil_full", None)
                if cached_pil is not None:
                    import cv2 as _cv2
                    import numpy as _np
                    img = _cv2.cvtColor(_np.array(cached_pil), _cv2.COLOR_RGB2BGR)
                    del cached_pil
                else:
                    img = decode_cv2_bgr(file_path)
            except Exception as exc:
                logger.warning("Cannot read image for face detection: %s (%s)", file_path, exc)
                metrics.count("worker.faces.decode_error")
                payload["faces"] = []
                payload["faces_complete"] = False
                return payload

            detector = get_detector()
            payload["face_model_key"] = getattr(detector, "embedding_model_key", None)
            payload["face_pipeline_key"] = getattr(detector, "detection_pipeline_key", None)
            faces = detector.detect(img)

            # Store full results (with embeddings) for DB sync.
            payload["faces"] = faces
            payload["faces_complete"] = True
            # Store bbox-only version for sidecar (embeddings too large).
            payload["faces_sidecar"] = [
                {"bbox": f["bbox"], "confidence": f["confidence"],
                 "detection_model": f.get("detection_model", "ensemble")}
                for f in faces
            ]
            metrics.record("worker.faces.count", len(faces))
            logger.info("Detected %d face(s) in %s", len(faces), file_path)
        except Exception as exc:
            logger.error("Face detection failed for %s: %s", file_path, exc)
            metrics.count("worker.faces.error")
            payload["faces"] = []
            payload["faces_complete"] = False

    _record_gpu_telemetry("faces_post")
    _vram_breathe()
    return payload


# ── Step E: Pluggable Scene / Search Embedding ───────────────────────────
def compute_scene_embedding(payload: dict) -> dict:
    """Encode a unit-norm image embedding with the desired scene engine.

    Reuses the cached thumbnail when present and falls back to the original.
    Persistence is model-versioned in ``scene_embeddings`` so vector spaces
    with different dimensions can coexist safely during migrations.
    """
    file_path = payload["file_path"]

    if _monitor.pressure() == "critical":
        logger.warning("Skipping scene embedding (RAM critical): %s", file_path)
        metrics.count("worker.scene.skipped_ram")
        payload["scene_embedding"] = None
        return payload

    with metrics.timer("worker.scene"):
        try:
            import hashlib
            from PIL import Image
            cache_key = hashlib.md5(file_path.encode()).hexdigest()
            thumb = THUMB_CACHE_DIR / f"{cache_key}.jpg"
            src = str(thumb) if thumb.exists() else file_path
            # The selected driver owns its final resize/crop. This upper bound
            # avoids decoding an enormous source into transient RAM first.
            img = decode_pil(src, max_dim=1024)
            # Cache for compute_dedup_embedding: same src + max_dim, so the
            # decoded image can be reused directly to avoid a second file read.
            payload["_img_1024"] = img
            engine = _get_ingestion_scene_engine()
            payload["scene_embedding"] = engine.encode_image(img)
            payload["scene_model"] = engine.spec.as_dict()
            payload["scene_runtime"] = engine.runtime_info.get("execution", {})
        except Exception as exc:
            logger.error("Scene embedding failed for %s: %s", file_path, exc)
            metrics.count("worker.scene.error")
            payload["scene_embedding"] = None

    _record_gpu_telemetry("scene_post")
    _vram_breathe()
    return payload


# ── Step E2: Dedup Embedding (SSCD) ──────────────────────────────────────────
def compute_dedup_embedding(payload: dict) -> dict:
    """Encode a unit-norm SSCD copy-detection embedding for near-dup grouping.

    Reuses the cached 240×240 thumbnail when present and falls back to the
    original file otherwise. Stored on ``media.dedup_embedding`` (nullable
    ``vector(512)`` — see migration 007). Failures soft-set the field to None
    so sync_metadata can still upsert the rest of the row.
    """
    file_path = payload["file_path"]

    if _monitor.pressure() == "critical":
        logger.warning("Skipping dedup embedding (RAM critical): %s", file_path)
        metrics.count("worker.dedup.skipped_ram")
        payload["dedup_embedding"] = None
        return payload

    img = None
    with metrics.timer("worker.dedup"):
        try:
            import hashlib
            from dedup_engine import encode_image

            cache_key = hashlib.md5(file_path.encode()).hexdigest()
            thumb = THUMB_CACHE_DIR / f"{cache_key}.jpg"
            src = str(thumb) if thumb.exists() else file_path
            # Reuse the image decoded by compute_scene_embedding when available
            # (identical src path + max_dim=1024), otherwise decode fresh.
            img = payload.pop("_img_1024", None) or decode_pil(src, max_dim=1024)
            payload["dedup_embedding"] = encode_image(img)
        except Exception as exc:
            logger.error("Dedup embedding failed for %s: %s", file_path, exc)
            metrics.count("worker.dedup.error")
            payload["dedup_embedding"] = None
        finally:
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass

    _record_gpu_telemetry("dedup_post")
    _vram_breathe()
    return payload


# ── Processing Log helpers (Spec 02 B2) ───────────────────────────────────────

def _upsert_processing_log(file_path: str, status: str, error_msg: str | None = None):
    """Insert or update a processing_log row for the given file."""
    conn = _get_pg()
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, attempt FROM processing_log WHERE file_path = %s ORDER BY id DESC LIMIT 1",
                (file_path,),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE processing_log SET status = %s, error_msg = %s, attempt = %s, updated_at = now() WHERE id = %s",
                    (status, error_msg, row[1] + (1 if status == "running" else 0), row[0]),
                )
            else:
                cur.execute(
                    "INSERT INTO processing_log (file_path, status, error_msg) VALUES (%s, %s, %s)",
                    (file_path, status, error_msg),
                )


# ── Step E: Sync to Sidecar JSON & Postgres (Two-Phase — Spec 02) ────────────

def _sidecar_entry_data(payload: dict) -> dict:
    """Build the sidecar entry dict from a pipeline payload."""
    return {
        "phash": payload.get("phash", ""),
        "faces": payload.get("faces_sidecar", payload.get("faces", [])),
        "exif": payload.get("exif", {}),
        "blur": payload.get("blur", 0.0),
    }


def sync_metadata(payload: dict) -> dict:
    """Two-phase sidecar → Postgres sync.

    Phase 1: Write sidecar with ``sidecar_state: "pending"``.
    Phase 2: Upsert to Postgres.
    Phase 3: Rewrite sidecar with ``sidecar_state: "synced"``.

    If phase 2 fails, sidecar is marked ``"failed"`` for later reconciliation.

    In plain English: writes happen to *two* places — the sidecar JSON on
    disk and the Postgres database. If we crash between the two, the
    sidecar's ``sidecar_state`` field tells the recovery script
    (``reconcile_sidecars.py``) what state we left in:

      • absent       — never processed, run the full pipeline.
      • "pending"    — sidecar exists but DB row may not; replay the DB
                       write.
      • "synced"     — both halves are committed; nothing to do.
      • "failed"     — last attempt errored; replay the DB write next run.

    This makes the pipeline safe to interrupt at any point.
    """
    file_path = payload["file_path"]
    p = Path(file_path)
    directory = str(p.parent)
    filename = p.name
    logger.info("Syncing metadata: %s", file_path)

    entry_data = _sidecar_entry_data(payload)

    # Phase 1: Write sidecar with state='pending'.
    with metrics.timer("worker.sync.sidecar_pending"):
        with _sidecar_lock(directory):
            sidecar = _load_sidecar(directory)
            sidecar[filename] = {**entry_data, "sidecar_state": "pending"}
            _save_sidecar(directory, sidecar)

    _sync_db_t0 = time.perf_counter()
    # Phase 2: Upsert to Postgres.
    try:
        conn = _get_pg()
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO media (file_path, filename, directory, extension,
                                       phash, blur_score, exif_data,
                                       file_size, mime_type, dedup_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (file_path) DO UPDATE SET
                        phash           = EXCLUDED.phash,
                        blur_score      = EXCLUDED.blur_score,
                        exif_data       = EXCLUDED.exif_data,
                        file_size       = COALESCE(EXCLUDED.file_size, media.file_size),
                        mime_type       = COALESCE(EXCLUDED.mime_type, media.mime_type),
                        dedup_embedding = COALESCE(EXCLUDED.dedup_embedding, media.dedup_embedding),
                        updated_at      = NOW()
                    RETURNING id
                    """,
                    (
                        file_path,
                        filename,
                        directory,
                        p.suffix.lower().lstrip("."),
                        payload.get("phash", "") or None,
                        payload.get("blur", 0.0),
                        json.dumps(payload.get("exif", {})),
                        payload.get("file_size"),
                        payload.get("mime_type"),
                        payload.get("dedup_embedding"),
                    ),
                )
                media_id = cur.fetchone()[0]
                _persist_scene_embedding(conn, media_id, payload)

                # Insert face embeddings (skip entries without embeddings).
                faces = payload.get("faces", [])
                inserted_face_rows = 0
                face_model_key = payload.get("face_model_key")
                if payload.get("faces_complete") and face_model_key:
                    face_pipeline_key = payload.get("face_pipeline_key")
                    if not face_pipeline_key:
                        raise ValueError(
                            "completed face result has no pipeline provenance key"
                        )
                    # Replace only this model namespace. Old/alternate vector
                    # spaces remain queryable until their own migration completes.
                    cur.execute(
                        """
                        DELETE FROM face_embeddings
                        WHERE media_id = %s AND embedding_model_key = %s
                        """,
                        (media_id, face_model_key),
                    )
                    for face in faces:
                        embedding = face.get("embedding")
                        if not embedding:
                            continue
                        embedding_model_key = face.get("embedding_model_key") or face_model_key
                        if embedding_model_key != face_model_key:
                            raise ValueError(
                                "face result model identity does not match the active embedder"
                            )
                        detection_pipeline_key = (
                            face.get("detection_pipeline_key") or face_pipeline_key
                        )
                        if detection_pipeline_key != face_pipeline_key:
                            raise ValueError(
                                "face result pipeline identity does not match "
                                "the active detector pipeline"
                            )
                        cur.execute(
                            """
                            INSERT INTO face_embeddings
                                (media_id, bbox, embedding, detection_model, confidence,
                                 embedding_model_key, embedding_dimension,
                                 detection_pipeline_key, detector_models,
                                 runtime_provider, runtime_precision)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                media_id,
                                json.dumps(face["bbox"]),
                                embedding,
                                face.get("detection_model", "ensemble"),
                                face.get("confidence", 0.0),
                                embedding_model_key,
                                int(face.get("embedding_dimension", len(embedding))),
                                detection_pipeline_key,
                                json.dumps(face.get("detector_models", [])),
                                face.get("runtime_provider", "unknown"),
                                face.get("runtime_precision", "unknown"),
                            ),
                        )
                        inserted_face_rows += 1
                    cur.execute(
                        """
                        INSERT INTO ml_processing_state
                            (media_id, kind, model_key, status, item_count,
                             duration_ms, error, updated_at)
                        VALUES (%s, 'face-embedding', %s, 'completed', %s,
                                NULL, NULL, NOW())
                        ON CONFLICT (media_id, kind, model_key) DO UPDATE SET
                            status = 'completed',
                            item_count = EXCLUDED.item_count,
                            duration_ms = EXCLUDED.duration_ms,
                            error = NULL,
                            updated_at = NOW()
                        """,
                        (media_id, face_model_key, inserted_face_rows),
                    )
        metrics.record(
            "worker.sync.db_upsert", (time.perf_counter() - _sync_db_t0) * 1000.0
        )
        metrics.record("worker.sync.faces", inserted_face_rows)
    except Exception as exc:
        logger.error("Postgres upsert failed for %s: %s", file_path, exc)
        metrics.count("worker.sync.error")
        # Mark sidecar as failed.
        with _sidecar_lock(directory):
            sidecar = _load_sidecar(directory)
            if filename in sidecar:
                sidecar[filename]["sidecar_state"] = "failed"
                _save_sidecar(directory, sidecar)
        _upsert_processing_log(file_path, "failed", str(exc))
        _vram_breathe()
        return payload

    # Phase 3: Mark sidecar as synced.
    with metrics.timer("worker.sync.sidecar_synced"):
        with _sidecar_lock(directory):
            sidecar = _load_sidecar(directory)
            if filename in sidecar:
                sidecar[filename]["sidecar_state"] = "synced"
                _save_sidecar(directory, sidecar)

    _upsert_processing_log(file_path, "synced")
    logger.info("Metadata synced for %s", file_path)
    _vram_breathe()
    return payload


# ── Face Reprocessing (for already-indexed media) ────────────────────────────
def _face_reindex_benchmark(job: dict, detector, aggregate: dict) -> tuple[dict, dict]:
    """Build truthful wall-clock and runtime metrics before atomic cutover."""
    aggregate = dict(aggregate)
    detect_duration_ms = float(aggregate.get("detect_duration_ms", 0.0))
    started_at = job.get("started_at")
    if isinstance(started_at, str):
        started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    if isinstance(started_at, datetime):
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        wall_duration_ms = max(
            detect_duration_ms,
            (datetime.now(timezone.utc) - started_at).total_seconds() * 1000.0,
        )
    else:
        wall_duration_ms = detect_duration_ms
    sample_count = int(aggregate.get("images_processed", 0))
    items_per_second = (
        sample_count * 1000.0 / wall_duration_ms if wall_duration_ms else 0.0
    )
    runtime = detector.runtime_info.get("execution", {})
    aggregate["wall_duration_ms"] = wall_duration_ms
    aggregate["items_per_second"] = items_per_second
    benchmark = {
        "benchmark_type": "face_reindex",
        "sample_count": sample_count,
        # Face reprocessing calls detector.detect() once per image. The
        # database scan chunk is an orchestration bound, not an ML batch.
        "batch_size": 1,
        "duration_ms": wall_duration_ms,
        "items_per_second": items_per_second,
        "peak_vram_mb": aggregate.get("peak_vram_mb"),
        "runtime_provider": runtime.get("actual_provider", "unknown"),
        "runtime_precision": runtime.get("actual_precision", "unknown"),
        "metrics": aggregate,
    }
    return aggregate, benchmark


def _finalize_face_reindex(
    conn, job: dict, detector, model_key: str, extensions: list[str], aggregate: dict
) -> tuple[dict, bool, int, int]:
    """Atomically validate coverage, benchmark, promote, and complete a face job."""
    import model_store

    aggregate, benchmark = _face_reindex_benchmark(job, detector, aggregate)
    with conn.transaction():
        finalized, activated, remaining = model_store.finalize_reindex_and_activate(
            conn,
            job["id"],
            metrics=aggregate,
            eligible_extensions=extensions,
            benchmark=benchmark,
        )
        unclustered = 0
        if activated:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM face_embeddings
                    WHERE embedding_model_key = %s AND cluster_id IS NULL
                    """,
                    (model_key,),
                )
                unclustered = int(cur.fetchone()[0] or 0)
    if activated:
        logger.info(
            "[FACE REPROCESS] Atomically activated model %s with benchmark job %s",
            model_key,
            job["id"],
        )
    return finalized, activated, remaining, unclustered


@huey.task(retries=1, retry_delay=60)
def reprocess_faces():
    """Run face analysis for media missing this model's completion marker.

    This handles the case where files were indexed before face detection was
    wired up — their sidecars say "synced" with empty faces, so the normal
    pipeline skips them. Successful zero-face passes are durable completion
    records, so they are not repeatedly analyzed.
    """
    if _monitor.pressure() == "critical":
        logger.warning("[FACE REPROCESS] memory critical; retrying in 30 seconds.")
        _gpu_cache_clear()
        reprocess_faces.schedule(args=(), delay=30)
        return

    import model_store
    from constants import IMAGE_EXTENSIONS
    from face_engine import get_detector

    logger.info("[FACE REPROCESS] Scanning the incomplete face-model namespace …")

    detector = get_detector()
    face_model_key = getattr(detector, "embedding_model_key", None)
    if not face_model_key:
        logger.warning("[FACE REPROCESS] Configured face engine has no embedder; nothing to index.")
        return
    face_pipeline_key = getattr(detector, "detection_pipeline_key", None)
    if not face_pipeline_key:
        raise RuntimeError("Configured face engine has no pipeline provenance key")
    extensions = sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS)

    conn = _get_pg()
    state = model_store.get_model_state(conn, "face-embedding")
    if state is None or state.get("desired_model_key") != face_model_key:
        raise model_store.ModelStateConflict(
            f"configured face model {face_model_key!r} is not the desired model"
        )
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM media m
            LEFT JOIN ml_processing_state ps
              ON ps.media_id = m.id
             AND ps.kind = 'face-embedding'
             AND ps.model_key = %s
             AND ps.status = 'completed'
            WHERE m.extension = ANY(%s) AND ps.media_id IS NULL
            """,
            (face_model_key, extensions),
        )
        missing_items = int(cur.fetchone()[0] or 0)
    if missing_items == 0 and state.get("active_model_key") == face_model_key:
        logger.info("[FACE REPROCESS] Face namespace is already complete.")
        return

    job, _ = model_store.create_reindex_job(
        conn,
        kind="face-embedding",
        model_key=face_model_key,
        total_items=missing_items,
        batch_size=_ADMIN_CHUNK_SIZE,
    )
    job_id = job["id"]
    if job["status"] == "queued":
        job = model_store.start_reindex_job(conn, job_id)
    if job["status"] == "cancel_requested":
        model_store.finalize_reindex_and_activate(
            conn, job_id, eligible_extensions=extensions,
        )
        return
    if job["status"] in {"completed", "failed", "cancelled"}:
        return

    processed = 0
    last_id = int(job.get("cursor_media_id") or 0)
    aggregate = dict(job.get("metrics") or {})
    stopped_early = False

    while True:
        chunk_processed = 0
        chunk_failed = 0
        chunk_detect_ms = 0.0
        chunk_cursor = last_id
        if _monitor.pressure() == "critical":
            logger.warning("[FACE REPROCESS] RAM critical — stopping after %d.", processed)
            stopped_early = True
            break

        conn = _get_pg()
        job = model_store.get_reindex_job(conn, job_id)
        if job is None:
            raise model_store.ModelStoreError(f"face re-index job {job_id} disappeared")
        if job["status"] == "cancel_requested":
            model_store.finalize_reindex_and_activate(
                conn, job_id, eligible_extensions=extensions,
            )
            return
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.file_path
                FROM media m
                LEFT JOIN ml_processing_state ps
                  ON ps.media_id = m.id
                 AND ps.kind = 'face-embedding'
                 AND ps.model_key = %s
                 AND ps.status = 'completed'
                WHERE ps.media_id IS NULL
                  AND m.id > %s
                  AND m.extension = ANY(%s)
                ORDER BY m.id
                LIMIT %s
                """,
                (face_model_key, last_id, extensions, _ADMIN_CHUNK_SIZE),
            )
            batch = cur.fetchall()

        if not batch:
            break

        for media_id, file_path in batch:
            if _monitor.pressure() == "critical":
                logger.warning("[FACE REPROCESS] RAM critical — stopping mid-chunk at %d.", processed)
                stopped_early = True
                break

            chunk_cursor = media_id
            try:
                _CURRENT_FILE_PATH.write_text(file_path)
            except Exception:
                pass

            try:
                try:
                    img = decode_cv2_bgr(file_path)
                except Exception as exc:
                    logger.warning("[FACE REPROCESS] Cannot read %s: %s", file_path, exc)
                    metrics.count("worker.reprocess_faces.decode_error")
                    chunk_failed += 1
                    continue

                detect_started = time.perf_counter()
                with metrics.timer("worker.reprocess_faces.detect"):
                    faces = detector.detect(img)
                chunk_detect_ms += (time.perf_counter() - detect_started) * 1000.0
                metrics.record("worker.reprocess_faces.faces_per_image", len(faces))

                if not faces:
                    logger.debug("[FACE REPROCESS] No faces in %s", file_path)
                else:
                    logger.info("[FACE REPROCESS] (%d) %d face(s) in %s",
                                processed + 1, len(faces), file_path)

                conn = _get_pg()
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM face_embeddings "
                            "WHERE media_id = %s AND embedding_model_key = %s",
                            (media_id, face_model_key),
                        )
                        inserted = 0
                        for face in faces:
                            embedding = face.get("embedding")
                            if not embedding:
                                continue
                            embedding_model_key = (
                                face.get("embedding_model_key") or face_model_key
                            )
                            if embedding_model_key != face_model_key:
                                raise ValueError(
                                    "face result model identity does not match "
                                    "the active embedder"
                                )
                            detection_pipeline_key = (
                                face.get("detection_pipeline_key") or face_pipeline_key
                            )
                            if detection_pipeline_key != face_pipeline_key:
                                raise ValueError(
                                    "face result pipeline identity does not match "
                                    "the active detector pipeline"
                                )
                            cur.execute(
                                """
                                INSERT INTO face_embeddings
                                    (media_id, bbox, embedding, detection_model, confidence,
                                     embedding_model_key, embedding_dimension,
                                     detection_pipeline_key, detector_models,
                                     runtime_provider, runtime_precision)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    media_id,
                                    json.dumps(face["bbox"]),
                                    embedding,
                                    face.get("detection_model", "ensemble"),
                                    face.get("confidence", 0.0),
                                    embedding_model_key,
                                    int(face.get("embedding_dimension", len(embedding))),
                                    detection_pipeline_key,
                                    json.dumps(face.get("detector_models", [])),
                                    face.get("runtime_provider", "unknown"),
                                    face.get("runtime_precision", "unknown"),
                                ),
                            )
                            inserted += 1
                        cur.execute(
                            """
                            INSERT INTO ml_processing_state
                                (media_id, kind, model_key, status, item_count,
                                 duration_ms, error, updated_at)
                            VALUES (%s, 'face-embedding', %s, 'completed', %s,
                                    NULL, NULL, NOW())
                            ON CONFLICT (media_id, kind, model_key) DO UPDATE SET
                                status = 'completed', item_count = EXCLUDED.item_count,
                                error = NULL, updated_at = NOW()
                            """,
                            (media_id, face_model_key, inserted),
                        )

                chunk_processed += 1
                processed += 1
                metrics.count("worker.reprocess_faces.processed")
                sidecar_faces = [
                    {"bbox": f["bbox"], "confidence": f["confidence"],
                     "detection_model": f.get("detection_model", "ensemble")}
                    for f in faces
                ]
                try:
                    _update_sidecar_faces(file_path, sidecar_faces)
                except Exception as sidecar_exc:
                    logger.warning(
                        "[FACE REPROCESS] Sidecar update failed for %s: %s",
                        file_path,
                        sidecar_exc,
                    )

            except Exception as exc:
                logger.error("[FACE REPROCESS] Error on %s: %s", file_path, exc)
                chunk_failed += 1

            _vram_breathe()

        runtime = detector.runtime_info.get("execution", {})
        aggregate["images_processed"] = (
            int(aggregate.get("images_processed", 0)) + chunk_processed
        )
        aggregate["detect_duration_ms"] = (
            float(aggregate.get("detect_duration_ms", 0.0)) + chunk_detect_ms
        )
        aggregate["peak_vram_mb"] = max(
            float(aggregate.get("peak_vram_mb", 0.0)),
            _monitor.gpu_bytes_used() / (1024 * 1024),
        )
        aggregate["runtime_provider"] = runtime.get("actual_provider", "unknown")
        aggregate["runtime_precision"] = runtime.get("actual_precision", "unknown")
        if chunk_cursor > last_id:
            job = model_store.update_reindex_job(
                conn,
                job_id,
                cursor_media_id=chunk_cursor,
                processed_items=int(job.get("processed_items") or 0) + chunk_processed,
                failed_items=int(job.get("failed_items") or 0) + chunk_failed,
                metrics=aggregate,
            )
            last_id = chunk_cursor
        metrics.count("worker.face_reindex.processed", chunk_processed)
        metrics.count("worker.face_reindex.failed", chunk_failed)
        if stopped_early:
            break

    try:
        _CURRENT_FILE_PATH.write_text("")
    except Exception:
        pass

    logger.info("[FACE REPROCESS] Done. Processed %d files.", processed)

    if stopped_early:
        logger.info(
            "[FACE REPROCESS] Persisted progress for job %s; retrying after "
            "memory pressure subsides.",
            job_id,
        )
        reprocess_faces.schedule(args=(), delay=30)
        return

    try:
        conn = _get_pg()
        finalized, activated, remaining, unclustered = _finalize_face_reindex(
            conn, job, detector, face_model_key, extensions, aggregate,
        )
    except Exception:
        logger.exception("[FACE REPROCESS] Atomic face-model cutover failed")
        raise

    if not activated:
        if remaining and finalized["status"] in {"running", "indexing"}:
            reprocess_faces()
        return

    if unclustered:
        logger.info(
            "[FACE REPROCESS] Queueing cluster_faces for %d unclustered rows.",
            unclustered,
        )
        cluster_faces()


# ── Blur Re-scoring (for already-indexed media) ─────────────────────────────
@huey.task(retries=1, retry_delay=60)
def rescore_blur():
    """Run blur detection on all media that still have blur_score = 0.

    Handles files indexed before the blur engine was wired up.
    """
    if _monitor.pressure() == "critical":
        logger.warning("Skipping blur re-scoring — RAM critical.")
        return

    from scripts.blur_detection.blur_detect import score_image

    logger.info("[BLUR RESCORE] Scanning for un-scored media …")

    processed = 0
    last_id = 0

    while True:
        if _monitor.pressure() == "critical":
            logger.warning("[BLUR RESCORE] RAM critical — stopping after %d.", processed)
            break

        conn = _get_pg()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, file_path
                FROM media
                WHERE blur_score = 0.0
                  AND id > %s
                  AND extension IN ('jpg','jpeg','png','gif','bmp','tiff','tif','webp','heic','heif','avif')
                ORDER BY id
                LIMIT %s
                """,
                (last_id, _ADMIN_CHUNK_SIZE),
            )
            batch = cur.fetchall()

        if not batch:
            break

        for media_id, file_path in batch:
            if _monitor.pressure() == "critical":
                logger.warning("[BLUR RESCORE] RAM critical — stopping mid-chunk at %d.", processed)
                break

            try:
                with metrics.timer("worker.rescore_blur.score"):
                    blur_score = score_image(file_path)
                _get_pg().execute(
                    "UPDATE media SET blur_score = %s, updated_at = NOW() WHERE id = %s",
                    (blur_score, media_id),
                )
                processed += 1
                metrics.count("worker.rescore_blur.processed")
                logger.info("[BLUR RESCORE] (%d) %s → %.4f", processed, file_path, blur_score)
            except Exception as exc:
                logger.error("[BLUR RESCORE] Error on %s: %s", file_path, exc)
                metrics.count("worker.rescore_blur.error")

            _vram_breathe()

        last_id = batch[-1][0]

    logger.info("[BLUR RESCORE] Done. Scored %d files.", processed)


def _scene_reindex_total(conn, model_key: str, force: bool) -> int:
    from constants import IMAGE_EXTENSIONS

    extensions = sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS)
    with conn.cursor() as cur:
        if force:
            cur.execute(
                "SELECT COUNT(*) FROM media WHERE extension = ANY(%s)",
                (extensions,),
            )
        else:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM media m
                LEFT JOIN scene_embeddings se
                  ON se.media_id = m.id AND se.model_key = %s
                WHERE m.extension = ANY(%s) AND se.media_id IS NULL
                """,
                (model_key, extensions),
            )
        return int(cur.fetchone()[0] or 0)


def queue_scene_reindex(*, force: bool = False) -> tuple[dict, bool]:
    """Register the configured model and idempotently enqueue its re-index."""

    import model_store
    from scene_engine import configured_model_spec

    conn = _get_pg()
    spec = configured_model_spec()
    model = model_store.register_model(conn, spec)
    state = model_store.reconcile_model_state(conn, "scene", model["model_key"])
    total = _scene_reindex_total(conn, model["model_key"], force)
    batch_size = _scene_reindex_batch_size()
    job, created = model_store.create_reindex_job(
        conn,
        kind="scene",
        model_key=model["model_key"],
        total_items=total,
        force_reindex=force,
        batch_size=batch_size,
    )
    if job["status"] in {"queued", "running", "indexing", "cancel_requested"}:
        # Re-dispatching an existing job is safe with the single-worker queue
        # and gives the compatibility trigger a way to recover a job whose
        # prior worker process exited before writing its continuation.
        reindex_scene_embeddings(str(job["id"]))
        logger.info(
            "[SCENE REINDEX] %s job=%s target=%s active=%s items=%d",
            "queued" if created else "resumed",
            job["id"], model["model_key"], state.get("active_model_key"), total,
        )
    return job, created


@huey.task(retries=1, retry_delay=60)
def backfill_scene_embeddings(force: bool = False):
    """Compatibility trigger for the resumable, model-versioned re-indexer."""

    job, _ = queue_scene_reindex(force=bool(force))
    return str(job["id"])


@huey.task(retries=2, retry_delay=60)
def reindex_scene_embeddings(job_id: str):
    """Process one bounded re-index chunk and queue continuation at the tail."""

    import hashlib

    import model_store
    from constants import IMAGE_EXTENSIONS
    from scene_engine import get_engine

    conn = _get_pg()
    job = model_store.get_reindex_job(conn, job_id)
    if job is None or job["status"] in {"completed", "failed", "cancelled"}:
        return

    def cancel_job(reason: str) -> None:
        cancelled = model_store.cancel_reindex_job(conn, job_id)
        if cancelled["status"] == "cancelled":
            model_store.update_reindex_job(conn, job_id, error=reason)
            current_state = model_store.get_model_state(conn, "scene")
            if (
                current_state
                and current_state.get("desired_model_key") == job["model_key"]
                and current_state.get("active_model_key") != job["model_key"]
            ):
                model_store.mark_model_state_failed(
                    conn, "scene", job["model_key"], reason
                )

    def fail_job(error: str) -> None:
        failed = model_store.fail_reindex_job(conn, job_id, error)
        if failed["status"] == "cancel_requested":
            cancel_job("re-index cancelled")
        elif failed["status"] == "failed":
            model_store.mark_model_state_failed(
                conn, "scene", job["model_key"], error
            )

    if job["status"] == "cancel_requested":
        cancel_job("re-index cancelled")
        return

    state = model_store.get_model_state(conn, "scene")
    if state is None or state.get("desired_model_key") != job["model_key"]:
        cancel_job(
            "configured scene model changed before this job completed"
        )
        return

    if _monitor.pressure() == "critical":
        logger.warning("[SCENE REINDEX] memory critical; delaying job %s", job_id)
        _gpu_cache_clear()
        reindex_scene_embeddings.schedule(args=(job_id,), delay=30)
        return

    if job["status"] == "queued":
        job = model_store.start_reindex_job(conn, job_id)
        if job["status"] == "cancel_requested":
            cancel_job("re-index cancelled")
            return
        if job["status"] in {"completed", "failed", "cancelled"}:
            return

    model = model_store.get_model(conn, job["model_key"])
    if model is None:
        fail_job("target model is not registered")
        return

    extensions = sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS)
    chunk_size = max(1, min(int(job.get("batch_size") or 16), 64))
    # Establish the next durable queue hand-off before reading or processing a
    # chunk. With one Huey worker, it cannot overtake this task; if the process
    # exits anywhere below, the successor resumes from the last committed
    # cursor and idempotently repeats at most one chunk.
    reindex_scene_embeddings(job_id)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.id, m.file_path
            FROM media m
            LEFT JOIN scene_embeddings se
              ON se.media_id = m.id AND se.model_key = %s
            WHERE m.id > %s
              AND m.extension = ANY(%s)
              AND (%s OR se.media_id IS NULL)
            ORDER BY m.id
            LIMIT %s
            """,
            (
                job["model_key"], int(job["cursor_media_id"]), extensions,
                bool(job["force_reindex"]), chunk_size,
            ),
        )
        rows = cur.fetchall()

    if not rows:
        aggregate = dict(job.get("metrics") or {})
        try:
            encode_duration_ms = float(aggregate.get("encode_duration_ms", 0.0))
            encoded = int(aggregate.get("images_encoded", 0))
            started_at = job.get("started_at")
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if isinstance(started_at, datetime):
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=timezone.utc)
                wall_duration_ms = max(
                    encode_duration_ms,
                    (datetime.now(timezone.utc) - started_at).total_seconds() * 1000.0,
                )
            else:
                wall_duration_ms = encode_duration_ms
            aggregate["wall_duration_ms"] = wall_duration_ms
            aggregate["encode_items_per_second"] = (
                (encoded * 1000.0 / encode_duration_ms)
                if encode_duration_ms
                else 0.0
            )
            benchmark_payload = {
                "benchmark_type": "scene_reindex",
                "sample_count": encoded,
                "batch_size": chunk_size,
                "duration_ms": wall_duration_ms,
                "items_per_second": (
                    (encoded * 1000.0 / wall_duration_ms)
                    if wall_duration_ms
                    else 0.0
                ),
                "peak_vram_mb": aggregate.get("peak_vram_mb"),
                "runtime_provider": aggregate.get("runtime_provider", "unknown"),
                "runtime_precision": aggregate.get("runtime_precision", "unknown"),
                "metrics": aggregate,
            }
            finalized, activated, remaining = (
                model_store.finalize_reindex_and_activate(
                    conn,
                    job_id,
                    metrics=aggregate,
                    eligible_extensions=extensions,
                    benchmark=benchmark_payload,
                )
            )
        except (model_store.ModelStateConflict, ValueError) as exc:
            fail_job(str(exc))
            return
        except Exception:
            # Database/connection failures are retryable. Do not turn a
            # transient cutover failure into a terminal model-state failure.
            logger.exception("[SCENE REINDEX] transient cutover failure job=%s", job_id)
            raise
        if not activated:
            if remaining and finalized["status"] in {"running", "indexing"}:
                logger.info(
                    "[SCENE REINDEX] coverage changed during scan; restarting "
                    "job=%s for %d missing vector(s)",
                    job_id,
                    remaining,
                )
            else:
                logger.info(
                    "[SCENE REINDEX] cutover skipped job=%s status=%s",
                    job_id,
                    finalized["status"],
                )
            return
        logger.info("[SCENE REINDEX] completed and activated job=%s", job_id)
        return

    try:
        engine = get_engine(model)
    except Exception as exc:
        fail_job(f"could not initialize target scene engine: {exc}")
        return
    runtime = engine.runtime_info.get("execution", {})
    adaptive = max(1, _monitor.adaptive_batch_size())
    micro_size = max(1, min(chunk_size, adaptive))
    successes = 0
    failures = 0
    encode_ms = 0.0

    for start in range(0, len(rows), micro_size):
        micro = rows[start:start + micro_size]
        images: list = []
        media_ids: list[int] = []
        for media_id, file_path in micro:
            try:
                cache_key = hashlib.md5(file_path.encode()).hexdigest()
                thumb = THUMB_CACHE_DIR / f"{cache_key}.jpg"
                source = str(thumb) if thumb.exists() else file_path
                images.append(decode_pil(source, max_dim=1024))
                media_ids.append(media_id)
            except Exception as exc:
                failures += 1
                logger.warning("[SCENE REINDEX] decode failed for %s: %s", file_path, exc)

        if not images:
            continue
        started = time.perf_counter()
        try:
            vectors = engine.encode_images(images)
            if len(vectors) != len(media_ids):
                raise RuntimeError(
                    "scene engine returned "
                    f"{len(vectors)} vectors for {len(media_ids)} images"
                )
        except Exception as exc:
            # Isolate non-OOM failures. The engine itself already responds to
            # CUDA OOM by clearing caches and recursively halving the batch.
            logger.warning("[SCENE REINDEX] batch encode failed; isolating: %s", exc)
            vectors = []
            surviving_ids: list[int] = []
            for media_id, image in zip(media_ids, images):
                try:
                    vectors.append(engine.encode_image(image))
                    surviving_ids.append(media_id)
                except Exception as item_exc:
                    failures += 1
                    logger.error(
                        "[SCENE REINDEX] encode failed media=%s: %s",
                        media_id, item_exc,
                    )
            media_ids = surviving_ids
        finally:
            encode_ms += (time.perf_counter() - started) * 1000.0
            # Model loading can fall back from CUDA to CPU after an OOM; read
            # actual execution state after inference for truthful provenance.
            runtime = engine.runtime_info.get("execution", {})
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass

        persisted = 0
        try:
            with conn.transaction():
                for media_id, vector in zip(media_ids, vectors):
                    model_store.upsert_scene_embedding(
                        conn,
                        media_id=media_id,
                        model_key=job["model_key"],
                        embedding=vector,
                        runtime_provider=runtime.get("actual_provider", "unknown"),
                        runtime_precision=runtime.get("actual_precision", "unknown"),
                    )
                    persisted += 1
            successes += persisted
        except Exception as exc:
            failures += len(vectors)
            logger.error("[SCENE REINDEX] vector persistence failed: %s", exc)
        _vram_breathe()

    aggregate = dict(job.get("metrics") or {})
    aggregate["images_encoded"] = int(aggregate.get("images_encoded", 0)) + successes
    aggregate["encode_duration_ms"] = (
        float(aggregate.get("encode_duration_ms", 0.0)) + encode_ms
    )
    aggregate["peak_vram_mb"] = max(
        float(aggregate.get("peak_vram_mb", 0.0)),
        _monitor.gpu_bytes_used() / (1024 * 1024),
    )
    aggregate["runtime_provider"] = runtime.get("actual_provider", "unknown")
    aggregate["runtime_precision"] = runtime.get("actual_precision", "unknown")
    model_store.update_reindex_job(
        conn,
        job_id,
        cursor_media_id=int(rows[-1][0]),
        processed_items=int(job.get("processed_items") or 0) + successes,
        failed_items=int(job.get("failed_items") or 0) + failures,
        metrics=aggregate,
    )
    metrics.count("worker.scene_reindex.processed", successes)
    metrics.count("worker.scene_reindex.failed", failures)


# ── Dedup-Embedding Backfill (SSCD) ──────────────────────────────────────────
@huey.task(retries=1, retry_delay=60)
def backfill_dedup_embeddings():
    """Encode SSCD dedup embeddings for media rows where ``dedup_embedding`` is NULL.

    Walks the table in keyset batches of ``_ADMIN_CHUNK_SIZE`` and encodes
    images in mini-batches of 32 on whatever device ``dedup_engine`` selects.
    Honors the memory kill-switch the same way ``backfill_scene_embeddings``
    does. Soft-aborts (with a single warning) if the SSCD checkpoint is
    unavailable so the task is safe to schedule unconditionally.
    """
    if _monitor.pressure() == "critical":
        logger.warning("Skipping dedup-embedding backfill — RAM critical.")
        return

    import hashlib
    from dedup_engine import encode_images_batch, ensure_model_present

    if ensure_model_present() is None:
        logger.warning("[DEDUP BACKFILL] SSCD checkpoint unavailable — aborting.")
        return

    BATCH = 32
    last_id = 0
    processed = 0

    logger.info("[DEDUP BACKFILL] Scanning for media without dedup_embedding …")

    while True:
        if _monitor.pressure() == "critical":
            logger.warning("[DEDUP BACKFILL] RAM critical — stopping after %d.", processed)
            break

        conn = _get_pg()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, file_path FROM media
                WHERE dedup_embedding IS NULL
                  AND id > %s
                  AND extension IN ('jpg','jpeg','png','gif','bmp','tiff','tif','webp','heic','heif','avif')
                ORDER BY id
                LIMIT %s
                """,
                (last_id, _ADMIN_CHUNK_SIZE),
            )
            batch = cur.fetchall()

        if not batch:
            break

        for chunk_start in range(0, len(batch), BATCH):
            if _monitor.pressure() == "critical":
                logger.warning("[DEDUP BACKFILL] RAM critical mid-chunk — stopping at %d.", processed)
                break

            chunk = batch[chunk_start:chunk_start + BATCH]
            images: list = []
            ids_for_chunk: list[int] = []

            for media_id, file_path in chunk:
                try:
                    cache_key = hashlib.md5(file_path.encode()).hexdigest()
                    thumb = THUMB_CACHE_DIR / f"{cache_key}.jpg"
                    src = str(thumb) if thumb.exists() else file_path
                    img = decode_pil(src, max_dim=1024)
                    images.append(img)
                    ids_for_chunk.append(media_id)
                except Exception as exc:
                    logger.warning("[DEDUP BACKFILL] skip %s: %s", file_path, exc)

            if not images:
                continue

            try:
                with metrics.timer("worker.dedup_backfill.encode_batch", n=len(images)):
                    embs = encode_images_batch(images)
            except Exception as exc:
                logger.error("[DEDUP BACKFILL] encode batch failed: %s", exc)
                metrics.count("worker.dedup_backfill.encode_error")
                continue
            finally:
                for img in images:
                    try:
                        img.close()
                    except Exception:
                        pass

            if embs is None:
                logger.warning("[DEDUP BACKFILL] encoder returned None — aborting.")
                return

            conn = _get_pg()
            with conn.transaction():
                with conn.cursor() as cur:
                    for media_id, emb in zip(ids_for_chunk, embs):
                        cur.execute(
                            "UPDATE media SET dedup_embedding = %s, updated_at = NOW() WHERE id = %s",
                            (emb, media_id),
                        )

            processed += len(embs)
            metrics.count("worker.dedup_backfill.processed", len(embs))
            _vram_breathe()

        last_id = batch[-1][0]
        logger.info("[DEDUP BACKFILL] processed=%d, last_id=%d", processed, last_id)

    logger.info("[DEDUP BACKFILL] Done. Encoded %d media.", processed)
    _gpu_cache_clear()
    gc.collect()


def _update_sidecar_faces(file_path: str, faces: list):
    """Update just the faces field in an existing sidecar entry."""
    p = Path(file_path)
    directory = str(p.parent)
    filename = p.name
    try:
        with _sidecar_lock(directory):
            sidecar = _load_sidecar(directory)
            if filename in sidecar:
                sidecar[filename]["faces"] = faces
                _save_sidecar(directory, sidecar)
    except Exception as exc:
        logger.warning("Failed to update sidecar faces for %s: %s", file_path, exc)


# ── Step F: Face Clustering ──────────────────────────────────────────────────
@huey.task(retries=2, retry_delay=60)
def cluster_faces():
    """Cluster all unclustered face embeddings using DBSCAN + HDBSCAN verify.

    Loads embeddings from Postgres, runs clustering via face_engine shim,
    then writes back cluster assignments and updates face_clusters table.
    Guarded by the memory kill-switch.
    """
    if _monitor.pressure() == "critical":
        logger.warning("Skipping face clustering — RAM critical.")
        return

    import numpy as np
    from pgvector.psycopg import register_vector
    from scipy.sparse import csr_matrix

    from face_detection.face_cluster import (
        cluster_dbscan_precomputed,
        cluster_hdbscan,
        cross_verify,
    )

    logger.info("Starting face clustering …")

    # DBSCAN params (also used for the SQL pre-filter so the kNN graph only
    # contains edges within reach of a core point).
    EPS = 0.5
    MIN_SAMPLES = 2
    KNN_K = 20

    # Load embeddings once. We need them for HDBSCAN cross-verification; the
    # DBSCAN pass uses a sparse k-NN graph from the pgvector HNSW index.
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT active_model_key
                FROM ml_model_state
                WHERE kind = 'face-embedding'
                """
            )
            model_row = cur.fetchone()
            if model_row is None or not model_row[0]:
                logger.info("No active face embedding model is configured.")
                return
            face_model_key = model_row[0]
            face_model_literal = _FACE_MODEL_SQL_LITERALS.get(face_model_key)
            if face_model_literal is None:
                # A future custom face space has no registered partial ANN
                # index. Keep clustering correct with an exact scan.
                cur.execute("SET LOCAL enable_indexscan = off")
            cur.execute(
                "SELECT id, embedding FROM face_embeddings "
                "WHERE cluster_id IS NULL AND embedding_model_key = %s ORDER BY id",
                (face_model_key,),
            )
            rows = cur.fetchall()

            if len(rows) < 2:
                logger.info(
                    "Not enough unclustered faces (%d) to cluster.", len(rows)
                )
                return

            ids = [r[0] for r in rows]
            embeddings = np.array(
                [
                    v.to_list() if hasattr(v := r[1], "to_list") else v
                    for r in rows
                ],
                dtype=np.float32,
            )

            # L2-normalise (HDBSCAN-on-euclidean ≈ cosine for unit vectors).
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

            # Hidden-cluster absorption: redirect new faces of hidden persons
            # into their hidden cluster so they don't respawn as a new visible
            # cluster. Must run before the kNN graph build.
            cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            if face_model_literal is None:
                hidden_model_predicate = "fe.embedding_model_key = %s"
                hidden_params = (face_model_key, ids, EPS)
            else:
                hidden_model_predicate = (
                    f"fe.embedding_model_key = {face_model_literal}"
                )
                hidden_params = (ids, EPS)
            cur.execute(
                f"""
                SELECT b.id, n.cluster_id, b.embedding <=> n.embedding AS dist
                FROM face_embeddings b
                CROSS JOIN LATERAL (
                    SELECT fe.id, fe.cluster_id, fe.embedding
                    FROM face_embeddings fe
                    JOIN face_clusters fc ON fc.id = fe.cluster_id
                    WHERE fc.is_hidden = TRUE
                      AND {hidden_model_predicate}
                    ORDER BY fe.embedding <=> b.embedding
                    LIMIT 1
                ) n
                WHERE b.id = ANY(%s)
                  AND b.embedding <=> n.embedding <= %s
                """,
                hidden_params,
            )
            absorbed = cur.fetchall()  # [(face_id, hidden_cluster_id, dist), ...]

            if absorbed:
                for face_id, hidden_cid, _ in absorbed:
                    cur.execute(
                        "UPDATE face_embeddings SET cluster_id = %s WHERE id = %s",
                        (hidden_cid, face_id),
                    )
                hidden_cids = list({cid for _, cid, _ in absorbed})
                cur.execute(
                    """
                    UPDATE face_clusters fc
                    SET face_count = sub.cnt
                    FROM (
                        SELECT cluster_id, COUNT(*) AS cnt
                        FROM face_embeddings
                        WHERE cluster_id = ANY(%s)
                        GROUP BY cluster_id
                    ) sub
                    WHERE fc.id = sub.cluster_id
                    """,
                    (hidden_cids,),
                )
                absorbed_ids = {fid for fid, _, _ in absorbed}
                rows = [r for r in rows if r[0] not in absorbed_ids]
                logger.info(
                    "Absorbed %d face(s) into hidden cluster(s).", len(absorbed_ids)
                )
                if len(rows) < 2:
                    logger.info("All unclustered faces absorbed into hidden clusters.")
                    conn.commit()
                    return
                ids = [r[0] for r in rows]
                embeddings = np.array(
                    [
                        v.to_list() if hasattr(v := r[1], "to_list") else v
                        for r in rows
                    ],
                    dtype=np.float32,
                )
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms

            # Pull top-k cosine neighbours per face. A built-in model gets a
            # literal predicate matching its partial HNSW index; a future
            # custom namespace takes the exact fallback selected above.
            with metrics.timer("worker.cluster_faces.hnsw_knn", k=KNN_K, n=len(ids)):
                if face_model_literal is None:
                    knn_model_predicate = "embedding_model_key = %s"
                    knn_params = (face_model_key, KNN_K, ids, EPS)
                else:
                    knn_model_predicate = (
                        f"embedding_model_key = {face_model_literal}"
                    )
                    knn_params = (KNN_K, ids, EPS)
                cur.execute(
                    f"""
                    SELECT b.id, n.id, b.embedding <=> n.embedding AS dist
                    FROM face_embeddings b
                    CROSS JOIN LATERAL (
                        SELECT id, embedding
                        FROM face_embeddings
                        WHERE cluster_id IS NULL
                          AND {knn_model_predicate}
                          AND id <> b.id
                        ORDER BY embedding <=> b.embedding
                        LIMIT %s
                    ) n
                    WHERE b.id = ANY(%s)
                      AND b.embedding <=> n.embedding <= %s
                    """,
                    knn_params,
                )
                knn = cur.fetchall()
            metrics.record("worker.cluster_faces.knn_edges", len(knn))

    # Build a symmetric sparse cosine-distance matrix. DBSCAN with
    # metric="precomputed" treats absent entries as out-of-reach.
    id_to_idx = {fid: i for i, fid in enumerate(ids)}
    n = len(ids)
    if knn:
        rows_idx = np.fromiter((id_to_idx[b] for b, _, _ in knn), dtype=np.int32, count=len(knn))
        cols_idx = np.fromiter((id_to_idx[nb] for _, nb, _ in knn), dtype=np.int32, count=len(knn))
        # Clamp to [eps_floor, eps] so explicit zeros (rare collisions on
        # identical embeddings) survive csr_matrix's zero-elision.
        data = np.fromiter((max(d, 1e-9) for _, _, d in knn), dtype=np.float32, count=len(knn))
        dist_csr = csr_matrix((data, (rows_idx, cols_idx)), shape=(n, n))
        dist_csr = dist_csr.maximum(dist_csr.T)  # symmetrise
    else:
        dist_csr = csr_matrix((n, n), dtype=np.float32)

    with metrics.timer("worker.cluster_faces.dbscan", n=len(ids)):
        labels = cluster_dbscan_precomputed(dist_csr, eps=EPS, min_samples=MIN_SAMPLES)
    with metrics.timer("worker.cluster_faces.hdbscan_verify", n=len(ids)):
        labels_v = cluster_hdbscan(embeddings, min_cluster_size=3)
    confidence = cross_verify(labels, labels_v)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    metrics.record("worker.cluster_faces.n_clusters", n_clusters)
    metrics.record("worker.cluster_faces.n_noise", n_noise)
    if len(ids) > 0:
        metrics.record(
            "worker.cluster_faces.noise_pct", 100.0 * n_noise / len(ids)
        )
    logger.info("Clustering done: %d clusters, %d noise", n_clusters, n_noise)

    # Map label → cluster DB id.
    label_to_cluster: dict[int, int] = {}

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for idx, (face_id, label) in enumerate(zip(ids, labels)):
                if label == -1:
                    continue  # noise — leave unclustered

                if label not in label_to_cluster:
                    cur.execute(
                        "INSERT INTO face_clusters (face_count, confidence_score) "
                        "VALUES (0, 0.0) RETURNING id",
                    )
                    cluster_id = cur.fetchone()[0]
                    label_to_cluster[label] = cluster_id

                cluster_id = label_to_cluster[label]
                cur.execute(
                    "UPDATE face_embeddings SET cluster_id = %s WHERE id = %s",
                    (cluster_id, face_id),
                )

            # Update face_count and confidence_score per cluster.
            for label, cluster_id in label_to_cluster.items():
                mask = labels == label
                count = int(mask.sum())
                avg_conf = float(np.mean([
                    1.0 if c == "high" else 0.5
                    for c, m in zip(confidence, mask) if m
                ]))
                # Pick representative: first face in this cluster.
                rep_face_id = ids[int(np.where(mask)[0][0])]
                cur.execute(
                    "SELECT media_id FROM face_embeddings WHERE id = %s",
                    (rep_face_id,),
                )
                rep_media_id = cur.fetchone()[0]
                cur.execute(
                    "UPDATE face_clusters SET face_count = %s, confidence_score = %s, "
                    "representative_media_id = %s WHERE id = %s",
                    (count, avg_conf, rep_media_id, cluster_id),
                )
        conn.commit()

    logger.info("Face clusters written to database.")
    _gpu_cache_clear()
    gc.collect()


# ── On-demand library scan ───────────────────────────────────────────────────
@huey.task(retries=1, retry_delay=30)
def scan_library_folder(path: str) -> int:
    """Incrementally scan one folder tree and enqueue new or changed media."""
    from watcher import scan_single_root

    logger.info("[LIBRARY SCAN] starting on-demand scan for %s", path)
    enqueued = scan_single_root(path)
    logger.info("[LIBRARY SCAN] finished %s (%d files enqueued)", path, enqueued)
    return enqueued


# ── Delete Media ──────────────────────────────────────────────────────────────
@huey.task(retries=3)
def delete_media(file_path: str):
    """Remove a deleted media file from the sidecar JSON and database."""
    p = Path(file_path)
    directory = str(p.parent)
    filename = p.name
    logger.info("Removing deleted media: %s", file_path)

    # Remove entry from the sidecar JSON.
    try:
        with _sidecar_lock(directory):
            sidecar = _load_sidecar(directory)
            if filename in sidecar:
                del sidecar[filename]
                _save_sidecar(directory, sidecar)
    except Exception as exc:
        logger.warning("Failed to update sidecar for %s: %s", file_path, exc)

    # Delete from Postgres (cascades to face_embeddings, duplicate_members, processing_log).
    _get_pg().execute("DELETE FROM media WHERE file_path = %s", (file_path,))

    logger.info("Removed metadata for %s", file_path)


# ── Full Pipeline ────────────────────────────────────────────────────────────
@huey.task(retries=3, retry_delay=30)
def process_media(file_path: str):
    """Run the full pipeline for a single media file synchronously.

    Huey processes tasks sequentially (single consumer), so we run all
    steps inline instead of chaining separate task dispatches.  This keeps
    the pipeline simple and reduces SQLite queue contention.

    **Idempotency:** If a sidecar entry already exists for this file, skip
    all ML steps and only sync the existing metadata to Postgres.
    """
    from constants import is_video

    global _task_counter
    with _task_lock:
        _task_counter += 1
        count = _task_counter
    remaining = _get_queue_size()
    total = count + remaining

    # Signal current file to the FastAPI stats endpoint.
    try:
        _CURRENT_FILE_PATH.write_text(file_path)
    except Exception:
        pass
    _heartbeat.touch(
        current_file=file_path,
        current_started_ts=time.time(),
        paused_memory=False,
    )

    p = Path(file_path)
    directory = str(p.parent)
    filename = p.name

    # Durable extension and file-size telemetry explains decoder outliers.
    try:
        extension = p.suffix.lower().lstrip(".") or "unknown"
        metrics.count("worker.file_ext", ext=extension)
        file_size_mb = os.stat(file_path).st_size / (1024 * 1024)
        metrics.record("worker.file_size_mb", file_size_mb)
    except Exception:
        pass

    # ── Video short-circuit (Phase E1) ─────────────────────────────────
    # Videos are still discovered by the watcher (so they show up in
    # MEDIA_EXTENSIONS) but the image pipeline can't process them. Mark
    # them synced so re-runs skip cleanly; keyframe extraction is a
    # future phase.
    if is_video(file_path):
        logger.info("[SKIP video] (%d/%d) -> %s", count, total, file_path)
        try:
            with _sidecar_lock(directory):
                sidecar = _load_sidecar(directory)
                sidecar[filename] = {"sidecar_state": "synced", "type": "video"}
                _save_sidecar(directory, sidecar)
        except Exception as exc:
            logger.warning("Failed to write video sidecar entry for %s: %s", file_path, exc)
        _upsert_processing_log(file_path, "synced")
        metrics.count("worker.process_media.skipped_video")
        try:
            _CURRENT_FILE_PATH.write_text("")
        except Exception:
            pass
        _heartbeat._state["completed_since_start"] += 1
        _heartbeat.touch(
            current_file="",
            current_started_ts=None,
            last_completed_ts=time.time(),
            paused_memory=False,
        )
        return

    logger.info("[PROCESSING] (%d/%d) -> %s", count, total, file_path)

    _upsert_processing_log(file_path, "running")

    # ── Idempotency: check sidecar_state (Spec 02 A2) ────────────────
    sidecar = _load_sidecar(directory)
    entry = sidecar.get(filename)
    state = entry.get("sidecar_state") if entry else None

    if state == "synced":
        # Fully processed — skip entirely.
        logger.info("[SIDECAR SYNCED] Already complete, skipping: %s", file_path)
        with metrics.timer("worker.process_media", stage="synced_skip"):
            generate_thumbnail(file_path)
    elif state in ("pending", "failed"):
        # ML done but DB not confirmed — skip ML, retry DB upsert.
        logger.info("[SIDECAR %s] Skipping ML, retrying DB sync: %s", state.upper(), file_path)
        with metrics.timer("worker.process_media", stage="pending_retry"):
            payload = {"file_path": file_path, **_sidecar_entry_data(entry), **stat_file(file_path)}
            generate_thumbnail(file_path)
            sync_metadata(payload)
    else:
        # New file (no sidecar entry) — run the pipeline, gated per-stage by the
        # processing config. stat + sync always run (minimum metadata for the UI).
        cfg = processing_config.load_features()
        feats = cfg["features"]
        clock = throughput.StageClock()
        with metrics.timer("worker.process_media", stage="full"):
            if feats["thumbnail"]:
                with clock.time("thumbnail"):
                    generate_thumbnail(file_path)
            if feats["exif"]:
                with clock.time("exif"):
                    payload = extract_exif(file_path)
            else:
                payload = {"file_path": file_path, "exif": {}}
            with clock.time("stat"):
                payload.update(stat_file(file_path))
            if feats["phash"]:
                with clock.time("phash"):
                    payload = compute_phash(payload)
            if feats["blur"]:
                with clock.time("blur"):
                    payload = detect_blur(payload)
            if feats["faces"]:
                with clock.time("faces"):
                    payload = detect_faces(payload)
            if feats["scene_clip"]:
                with clock.time("scene_clip"):
                    payload = compute_scene_embedding(payload)
            try:
                if feats["sscd_dedup"]:
                    with clock.time("sscd_dedup"):
                        payload = compute_dedup_embedding(payload)
                with clock.time("db_write"):
                    sync_metadata(payload)
            finally:
                # The scene stage deliberately shares its decoded image with
                # SSCD. If SSCD is disabled or a later stage fails, release it
                # here instead of waiting for garbage collection under load.
                cached_image = payload.pop("_img_1024", None)
                if cached_image is not None:
                    try:
                        cached_image.close()
                    except Exception:
                        pass
        try:
            throughput.log_run(throughput.build_record(
                mode=cfg["preset"],
                files=1,
                total_bytes=int(payload.get("file_size") or 0),
                total_sec=sum(clock.as_dict().values()),
                enabled_features=feats,
                stage_sec=clock.as_dict(),
                config_load_error=cfg.get("config_load_error", False),
            ))
        except Exception as exc:
            logger.warning("throughput log failed for %s: %s", file_path, exc)

    logger.info("[COMPLETED] -> Generated Sidecar for: %s", file_path)

    # Clear current-file marker and record completion.
    try:
        _CURRENT_FILE_PATH.write_text("")
    except Exception:
        pass
    _heartbeat._state["completed_since_start"] += 1
    _heartbeat.touch(
        current_file="",
        current_started_ts=None,
        last_completed_ts=time.time(),
        paused_memory=False,
    )

    # ── Hardware Safeguard (Kill Switch) ───────────────────────────
    # If memory pressure is critical, pause and clear GPU caches.
    if _monitor.pressure() == "critical":
        try:
            rss = _monitor.rss_percent() * 100
        except RuntimeError:
            rss = -1.0
        logger.warning(
            "[RAM SAFEGUARD] System RAM at %.1f%% – sleeping 10s to allow OS to reclaim memory.",
            rss,
        )
        metrics.count("worker.ram_safeguard.triggered")
        _heartbeat.touch(paused_memory=True, pressure="critical")
        _gpu_cache_clear()
        gc.collect()
        time.sleep(10)
        _heartbeat.touch(paused_memory=False)


# ── Phase E6.2: Batched pipeline ─────────────────────────────────────────────
# `process_media_batch` is a Huey task that processes N images in one go,
# fanning the CPU steps across a thread pool and using detect_batch /
# encode_images_batch for the GPU steps. The single-image `process_media`
# above stays as the safety-net path (sidecar reconciliation, requeue_missing,
# MEMORIA_BATCH_SIZE=1 fallback, partial-batch retry on RAM-critical bail).
#
# Batch size is set by the watcher via `MEMORIA_BATCH_SIZE`; this task itself
# is agnostic to the chosen size — it processes whatever list it's handed.


@dataclass
class _BatchEntry:
    """Per-image state through the batched pipeline."""
    file_path: str
    payload: dict = field(default_factory=dict)
    error: str | None = None


def _safe_decode_cv2(path: str):
    """decode_cv2_bgr wrapped to return None on failure (per-image isolation)."""
    try:
        return decode_cv2_bgr(path)
    except Exception as exc:
        logger.warning("[BATCH] cv2 decode failed for %s: %s", path, exc)
        return None


def _safe_decode_pil_for_clip(path: str, file_path: str):
    """decode_pil(max_dim=1024) — same as compute_scene_embedding's single-path."""
    try:
        return decode_pil(path, max_dim=1024)
    except Exception as exc:
        logger.warning("[BATCH] scene decode failed for %s: %s", file_path, exc)
        return None


def _bstep_decode_meta_phash(entry: _BatchEntry, feats: dict) -> None:
    """CPU step: thumbnail + EXIF + stat + pHash, gated by *feats*. stat always
    runs (minimum metadata). Mutates entry.payload in place."""
    try:
        if feats["thumbnail"]:
            generate_thumbnail(entry.file_path)
        if feats["exif"]:
            payload = extract_exif(entry.file_path)
        else:
            payload = {"file_path": entry.file_path, "exif": {}}
        payload.update(stat_file(entry.file_path))
        if feats["phash"]:
            payload = compute_phash(payload)
        entry.payload = payload
    except Exception as exc:
        entry.error = f"decode_meta_phash: {exc}"
        logger.warning("[BATCH] %s for %s", entry.error, entry.file_path)


def _bstep_blur(entry: _BatchEntry) -> None:
    """CPU+GPU step: 6-model blur ensemble. Adds entry.payload['blur']."""
    if entry.error or not entry.payload:
        return
    try:
        entry.payload = detect_blur(entry.payload)
    except Exception as exc:
        entry.error = f"blur: {exc}"
        logger.warning("[BATCH] %s for %s", entry.error, entry.file_path)


def _bstep_faces_batch(entries: list[_BatchEntry]) -> None:
    """GPU step: batched face detection via FaceDetector.detect_batch."""
    valid = [e for e in entries if not e.error and e.payload]
    if not valid:
        return
    if _monitor.pressure() == "critical":
        logger.warning("[BATCH] face: RAM critical — skipping face detection.")
        for e in valid:
            e.payload["faces"] = []
            e.payload["faces_sidecar"] = []
            e.payload["faces_complete"] = False
        return

    # Decode cv2 BGR images concurrently; the GPU detect runs serialized
    # inside detect_batch with its own thread pool.
    from concurrent.futures import ThreadPoolExecutor
    images: list = []
    decoded: list[_BatchEntry] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_safe_decode_cv2, [e.file_path for e in valid]))
    for e, img in zip(valid, results):
        if img is None:
            e.payload["faces"] = []
            e.payload["faces_sidecar"] = []
            e.payload["faces_complete"] = False
        else:
            images.append(img)
            decoded.append(e)

    if not decoded:
        return

    try:
        from face_engine import get_detector
        detector = get_detector()
        for e in decoded:
            e.payload["face_model_key"] = getattr(detector, "embedding_model_key", None)
            e.payload["face_pipeline_key"] = getattr(detector, "detection_pipeline_key", None)
        all_faces = detector.detect_batch(images, max_workers=2)
    except Exception as exc:
        logger.error("[BATCH] detect_batch failed: %s", exc)
        for e in decoded:
            e.payload["faces"] = []
            e.payload["faces_sidecar"] = []
            e.payload["faces_complete"] = False
        return

    for e, faces in zip(decoded, all_faces):
        e.payload["faces"] = faces
        e.payload["faces_complete"] = True
        e.payload["faces_sidecar"] = [
            {
                "bbox": f["bbox"],
                "confidence": f["confidence"],
                "detection_model": f.get("detection_model", "ensemble"),
            }
            for f in faces
        ]
    _vram_breathe()


def _bstep_scene_embed_batch(entries: list[_BatchEntry]) -> None:
    """Accelerated step: batch encoding through the desired scene engine."""
    valid = [e for e in entries if not e.error and e.payload]
    if not valid:
        return
    if _monitor.pressure() == "critical":
        logger.warning("[BATCH] scene model: RAM critical — skipping embedding.")
        for e in valid:
            e.payload["scene_embedding"] = None
        return

    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    def _src_for(e: _BatchEntry) -> str:
        cache_key = hashlib.md5(e.file_path.encode()).hexdigest()
        thumb = THUMB_CACHE_DIR / f"{cache_key}.jpg"
        return str(thumb) if thumb.exists() else e.file_path

    pil_images: list = []
    decoded: list[_BatchEntry] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(
            lambda e: _safe_decode_pil_for_clip(_src_for(e), e.file_path),
            valid,
        ))
    for e, img in zip(valid, results):
        if img is None:
            e.payload["scene_embedding"] = None
        else:
            pil_images.append(img)
            decoded.append(e)

    if not decoded:
        return

    try:
        engine = _get_ingestion_scene_engine()
        embs = engine.encode_images(pil_images)
    except Exception as exc:
        logger.error("[BATCH] encode_images_batch failed: %s", exc)
        for e in decoded:
            e.payload["scene_embedding"] = None
        return
    finally:
        for image in pil_images:
            try:
                image.close()
            except Exception:
                pass

    for e, emb in zip(decoded, embs):
        e.payload["scene_embedding"] = emb
        e.payload["scene_model"] = engine.spec.as_dict()
        e.payload["scene_runtime"] = engine.runtime_info.get("execution", {})
    _vram_breathe()


def _bstep_dedup_embed_batch(entries: list[_BatchEntry]) -> None:
    """GPU step: tensor-batched SSCD encoding via encode_images_batch."""
    valid = [e for e in entries if not e.error and e.payload]
    if not valid:
        return
    if _monitor.pressure() == "critical":
        logger.warning("[BATCH] DEDUP: RAM critical — skipping dedup embedding.")
        for e in valid:
            e.payload["dedup_embedding"] = None
        return

    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    def _src_for(e: _BatchEntry) -> str:
        cache_key = hashlib.md5(e.file_path.encode()).hexdigest()
        thumb = THUMB_CACHE_DIR / f"{cache_key}.jpg"
        return str(thumb) if thumb.exists() else e.file_path

    pil_images: list = []
    decoded: list[_BatchEntry] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(
            lambda e: _safe_decode_pil_for_clip(_src_for(e), e.file_path),
            valid,
        ))
    for e, img in zip(valid, results):
        if img is None:
            e.payload["dedup_embedding"] = None
        else:
            pil_images.append(img)
            decoded.append(e)

    if not decoded:
        return

    try:
        from dedup_engine import encode_images_batch
        embs = encode_images_batch(pil_images)
    except Exception as exc:
        logger.error("[BATCH] dedup encode_images_batch failed: %s", exc)
        for e in decoded:
            e.payload["dedup_embedding"] = None
        return
    finally:
        for image in pil_images:
            try:
                image.close()
            except Exception:
                pass

    if embs is None:
        # SSCD checkpoint not present — soft-skip without spamming.
        for e in decoded:
            e.payload["dedup_embedding"] = None
        return

    for e, emb in zip(decoded, embs):
        e.payload["dedup_embedding"] = emb
    _vram_breathe()


def _sync_metadata_batch(entries: list[_BatchEntry]) -> None:
    """Two-phase sync, batched: sidecars grouped by directory; one Postgres
    transaction for all upserts; processing_log updated per file at the end.
    """
    valid = [e for e in entries if e.payload]
    if not valid:
        return
    with metrics.timer("worker.batch.phase_sync", n=len(valid)):
        _sync_metadata_batch_inner(entries, valid)


def _sync_metadata_batch_inner(entries: list[_BatchEntry], valid: list[_BatchEntry]) -> None:
    # Group entries by their parent directory for sidecar batching.
    by_dir: dict[str, list[_BatchEntry]] = {}
    for e in valid:
        d = str(Path(e.file_path).parent)
        by_dir.setdefault(d, []).append(e)

    # Phase 1: write all sidecars marking each entry pending. One JSON
    # rewrite per directory regardless of how many entries it contains.
    for directory, dir_entries in by_dir.items():
        try:
            with _sidecar_lock(directory):
                sidecar = _load_sidecar(directory)
                for e in dir_entries:
                    filename = Path(e.file_path).name
                    if e.error:
                        # Don't write a pending entry for already-failed items;
                        # their state will be set in phase 3.
                        continue
                    entry_data = _sidecar_entry_data(e.payload)
                    sidecar[filename] = {**entry_data, "sidecar_state": "pending"}
                _save_sidecar(directory, sidecar)
        except Exception as exc:
            logger.error("[BATCH] sidecar phase 1 failed for %s: %s", directory, exc)
            for e in dir_entries:
                if not e.error:
                    e.error = f"sidecar phase 1: {exc}"

    # Phase 2: bulk Postgres upsert in one transaction.
    successful: list[str] = []
    failed: list[tuple[str, str]] = []
    upsertable = [e for e in valid if not e.error]

    if upsertable:
        try:
            conn = _get_pg()
            with conn.transaction():
                with conn.cursor() as cur:
                    for e in upsertable:
                        # One bad media row must not poison the surrounding
                        # PostgreSQL transaction and roll back every other
                        # item in the batch. A reusable savepoint provides
                        # true per-item isolation without giving up the batch
                        # commit boundary.
                        cur.execute("SAVEPOINT memoria_batch_item")
                        try:
                            p = Path(e.file_path)
                            cur.execute(
                                """
                                INSERT INTO media (file_path, filename, directory, extension,
                                                   phash, blur_score, exif_data,
                                                   file_size, mime_type, dedup_embedding)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (file_path) DO UPDATE SET
                                    phash           = EXCLUDED.phash,
                                    blur_score      = EXCLUDED.blur_score,
                                    exif_data       = EXCLUDED.exif_data,
                                    file_size       = COALESCE(EXCLUDED.file_size, media.file_size),
                                    mime_type       = COALESCE(EXCLUDED.mime_type, media.mime_type),
                                    dedup_embedding = COALESCE(EXCLUDED.dedup_embedding, media.dedup_embedding),
                                    updated_at      = NOW()
                                RETURNING id
                                """,
                                (
                                    e.file_path,
                                    p.name,
                                    str(p.parent),
                                    p.suffix.lower().lstrip("."),
                                    e.payload.get("phash", "") or None,
                                    e.payload.get("blur", 0.0),
                                    json.dumps(e.payload.get("exif", {})),
                                    e.payload.get("file_size"),
                                    e.payload.get("mime_type"),
                                    e.payload.get("dedup_embedding"),
                                ),
                            )
                            media_id = cur.fetchone()[0]
                            _persist_scene_embedding(conn, media_id, e.payload)

                            faces = e.payload.get("faces", [])
                            face_model_key = e.payload.get("face_model_key")
                            if e.payload.get("faces_complete") and face_model_key:
                                face_pipeline_key = e.payload.get("face_pipeline_key")
                                if not face_pipeline_key:
                                    raise ValueError(
                                        "completed face result has no pipeline provenance key"
                                    )
                                cur.execute(
                                    """
                                    DELETE FROM face_embeddings
                                    WHERE media_id = %s AND embedding_model_key = %s
                                    """,
                                    (media_id, face_model_key),
                                )
                                for face in faces:
                                    embedding = face.get("embedding")
                                    if not embedding:
                                        continue
                                    embedding_model_key = (
                                        face.get("embedding_model_key") or face_model_key
                                    )
                                    if embedding_model_key != face_model_key:
                                        raise ValueError(
                                            "face result model identity does not match "
                                            "the active embedder"
                                        )
                                    detection_pipeline_key = (
                                        face.get("detection_pipeline_key")
                                        or face_pipeline_key
                                    )
                                    if detection_pipeline_key != face_pipeline_key:
                                        raise ValueError(
                                            "face result pipeline identity does not match "
                                            "the active detector pipeline"
                                        )
                                    cur.execute(
                                        """
                                        INSERT INTO face_embeddings
                                            (media_id, bbox, embedding, detection_model, confidence,
                                             embedding_model_key, embedding_dimension,
                                             detection_pipeline_key, detector_models,
                                             runtime_provider, runtime_precision)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                        """,
                                        (
                                            media_id,
                                            json.dumps(face["bbox"]),
                                            embedding,
                                            face.get("detection_model", "ensemble"),
                                            face.get("confidence", 0.0),
                                            embedding_model_key,
                                            int(face.get("embedding_dimension", len(embedding))),
                                            detection_pipeline_key,
                                            json.dumps(face.get("detector_models", [])),
                                            face.get("runtime_provider", "unknown"),
                                            face.get("runtime_precision", "unknown"),
                                        ),
                                    )
                                cur.execute(
                                    """
                                    INSERT INTO ml_processing_state
                                        (media_id, kind, model_key, status, item_count,
                                         duration_ms, error, updated_at)
                                    VALUES (%s, 'face-embedding', %s, 'completed', %s,
                                            NULL, NULL, NOW())
                                    ON CONFLICT (media_id, kind, model_key) DO UPDATE SET
                                        status = 'completed',
                                        item_count = EXCLUDED.item_count,
                                        error = NULL,
                                        updated_at = NOW()
                                    """,
                                    (
                                        media_id,
                                        face_model_key,
                                        sum(1 for face in faces if face.get("embedding")),
                                    ),
                                )
                            cur.execute("RELEASE SAVEPOINT memoria_batch_item")
                            successful.append(e.file_path)
                        except Exception as exc:
                            cur.execute("ROLLBACK TO SAVEPOINT memoria_batch_item")
                            cur.execute("RELEASE SAVEPOINT memoria_batch_item")
                            e.error = f"db upsert: {exc}"
                            failed.append((e.file_path, str(exc)))
                            logger.error(
                                "[BATCH] db upsert failed for %s: %s",
                                e.file_path, exc,
                            )
        except Exception as exc:
            logger.error("[BATCH] db transaction failed: %s", exc)
            for e in upsertable:
                if not e.error:
                    e.error = f"db transaction: {exc}"
                    failed.append((e.file_path, str(exc)))

    # Phase 3: rewrite sidecars to synced (success) or failed (error). Again,
    # one JSON rewrite per directory.
    for directory, dir_entries in by_dir.items():
        try:
            with _sidecar_lock(directory):
                sidecar = _load_sidecar(directory)
                changed = False
                for e in dir_entries:
                    filename = Path(e.file_path).name
                    if e.error:
                        if filename in sidecar:
                            sidecar[filename]["sidecar_state"] = "failed"
                        else:
                            # Phase 1 didn't write because of an early error;
                            # record the failure here so reconciliation sees it.
                            sidecar[filename] = {"sidecar_state": "failed"}
                        changed = True
                    elif filename in sidecar:
                        sidecar[filename]["sidecar_state"] = "synced"
                        changed = True
                if changed:
                    _save_sidecar(directory, sidecar)
        except Exception as exc:
            logger.error("[BATCH] sidecar phase 3 failed for %s: %s", directory, exc)

    # Update processing_log per file.
    for path in successful:
        _upsert_processing_log(path, "synced")
    for path, err in failed:
        _upsert_processing_log(path, "failed", err)


# ── Pipeline orchestration helpers (Phase E6.2 v2 — CPU/GPU overlap) ────────
# `_run_batch_serial` is the original pipeline shape — every step blocks the
# next. Used for very small batches where double-buffering overhead would
# eat the win.
#
# `_run_batch_pipelined` splits the batch into halves A and B and runs them
# as a 4-stage pipeline so CPU and GPU work overlap on different physical
# resources:
#
#     stage 1 [serial]:  CPU(A)
#     stage 2 [parallel]: GPU(A)  ||  CPU(B)
#     stage 3 [parallel]: SYNC(A) ||  GPU(B)
#     stage 4 [serial]:  SYNC(B)
#
# Both halves use the same singleton FaceDetector and scene_engine. The two
# GPU sub-stages don't run concurrently — they're sequential by construction
# (stage 2 GPU(A) then stage 3 GPU(B)) — so there's no GPU-stream contention.
# The overlap happens between GPU and CPU on different resources.

def _phase_cpu(es: list[_BatchEntry], feats: dict, clock: "throughput.StageClock") -> None:
    """Decode + EXIF + stat + pHash + blur for *es*, gated by *feats*."""
    from concurrent.futures import ThreadPoolExecutor
    if not es:
        return
    with metrics.timer("worker.batch.phase_cpu", n=len(es)), clock.time("cpu_meta"):
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda e: _bstep_decode_meta_phash(e, feats), es))
    if feats["blur"] and _monitor.pressure() != "critical":
        with clock.time("blur"):
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(_bstep_blur, es))


def _phase_gpu(es: list[_BatchEntry], feats: dict, clock: "throughput.StageClock") -> None:
    """Face, scene/search, and SSCD encoding for *es*, gated by *feats*."""
    if not es:
        return
    with metrics.timer("worker.batch.phase_gpu", n=len(es)):
        if feats["faces"]:
            with clock.time("faces"):
                _bstep_faces_batch(es)
        if feats["scene_clip"]:
            with clock.time("scene_clip"):
                _bstep_scene_embed_batch(es)
        if feats["sscd_dedup"]:
            with clock.time("sscd_dedup"):
                _bstep_dedup_embed_batch(es)


def _run_batch_serial(entries: list[_BatchEntry], feats: dict, clock: "throughput.StageClock") -> None:
    """Original pipeline — every phase blocks the next. Best for tiny batches."""
    _phase_cpu(entries, feats, clock)
    _phase_gpu(entries, feats, clock)
    with clock.time("db_write"):
        _sync_metadata_batch(entries)


def _run_batch_pipelined(entries: list[_BatchEntry], feats: dict, clock: "throughput.StageClock") -> None:
    """Double-buffered pipeline so CPU and GPU phases overlap.

    Splits the batch into halves A and B and orchestrates the 4 stages
    described above. Worth it for batches >= 8; below that the orchestration
    overhead eats the parallelism win.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = len(entries)
    half = n // 2
    a, b = entries[:half], entries[half:]

    # Stage 1: CPU on A (serial; nothing else to do yet).
    _phase_cpu(a, feats, clock)

    # Stage 2: GPU on A   ||   CPU on B
    # Both halves' face/scene engines use shared singletons; inference locks
    # serialize accelerator work while CPU on B uses Python threads (GIL releases
    # during decode I/O and through cv2/PIL native code).
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="batch-pipe")
    try:
        f_gpu_a = pool.submit(_phase_gpu, a, feats, clock)
        f_cpu_b = pool.submit(_phase_cpu, b, feats, clock)
        # Wait for both. .result() re-raises any exception from the thread.
        f_gpu_a.result()
        f_cpu_b.result()

        # Stage 3: Sync on A   ||   GPU on B
        # Sync is DB + filesystem; doesn't touch the GPU, so the two
        # genuinely run on different resources.
        def _sync_a():
            with clock.time("db_write"):
                _sync_metadata_batch(a)
        f_sync_a = pool.submit(_sync_a)
        f_gpu_b = pool.submit(_phase_gpu, b, feats, clock)
        f_sync_a.result()
        f_gpu_b.result()
    finally:
        pool.shutdown(wait=True)

    # Stage 4: Sync on B (serial).
    with clock.time("db_write"):
        _sync_metadata_batch(b)


@huey.task(retries=3, retry_delay=30)
def process_media_batch(file_paths: list[str]):
    """Batched companion to process_media (Phase E6.2).

    The Huey consumer is still single-threaded. Within this task:
      * CPU steps (decode, EXIF, stat, pHash, blur) fan across a thread pool.
      * Face detection runs through FaceDetector.detect_batch.
      * scene embedding runs through the selected engine's batch interface.
      * Sync writes one Postgres transaction + one sidecar rewrite per dir.

    Triggered by the watcher when MEMORIA_BATCH_SIZE > 1 (see watcher.py).
    Per-image errors are isolated — one bad file doesn't kill the batch.
    """
    from concurrent.futures import ThreadPoolExecutor
    from constants import is_image, is_video

    if not file_paths:
        return

    # If memory pressure is critical, bail before allocating any decode buffers.
    # Re-enqueue residuals as single-image tasks so the safety-net path can
    # take over with its own breathing logic.
    if _monitor.pressure() == "critical":
        logger.warning(
            "[BATCH] RAM critical at start — re-enqueuing %d paths individually.",
            len(file_paths),
        )
        for p in file_paths:
            try:
                process_media(p)
            except Exception as exc:
                logger.warning("[BATCH] re-enqueue failed for %s: %s", p, exc)
        _gpu_cache_clear()
        gc.collect()
        return

    # Increment the global counter once per batch (not once per file).
    global _task_counter
    with _task_lock:
        _task_counter += 1
        count = _task_counter
    remaining = _get_queue_size()

    logger.info(
        "[BATCH] processing %d paths (task %d, %d remaining in queue)",
        len(file_paths), count, remaining,
    )

    # Signal a representative "current file" for the stats endpoint.
    try:
        _CURRENT_FILE_PATH.write_text(file_paths[0])
    except Exception:
        pass
    _heartbeat.touch(
        current_file=file_paths[0],
        current_started_ts=time.time(),
        paused_memory=False,
    )

    # ── Triage: video skip + already-synced + retry-DB-only ─────────────────
    cfg = processing_config.load_features()
    feats = cfg["features"]
    clock = throughput.StageClock()
    _scan_t0 = time.perf_counter()
    new_paths: list[str] = []
    triaged_completed = 0
    for fp in file_paths:
        # Video short-circuit (mirrors process_media).
        if is_video(fp):
            try:
                p = Path(fp)
                directory = str(p.parent)
                with _sidecar_lock(directory):
                    sidecar = _load_sidecar(directory)
                    sidecar[p.name] = {
                        "sidecar_state": "synced", "type": "video"
                    }
                    _save_sidecar(directory, sidecar)
                _upsert_processing_log(fp, "synced")
                triaged_completed += 1
            except Exception as exc:
                logger.warning("[BATCH] video skip sidecar write failed for %s: %s", fp, exc)
            continue

        if not is_image(fp):
            logger.info("[BATCH] skipping unknown extension: %s", fp)
            continue

        # Sidecar idempotency check.
        try:
            p = Path(fp)
            sidecar = _load_sidecar(str(p.parent))
            entry = sidecar.get(p.name)
            state = entry.get("sidecar_state") if entry else None
        except Exception:
            state = None
            entry = None

        if state == "synced":
            # Already complete — just regenerate the thumbnail (in case
            # the cache was swept) and move on.
            try:
                generate_thumbnail(fp)
            except Exception:
                pass
            triaged_completed += 1
            continue

        if state in ("pending", "failed") and entry is not None:
            # ML done previously, only DB sync needs retry. Use the
            # single-image sync_metadata for these — they're rare (only
            # show up after a crash) and not worth bundling.
            try:
                payload = {"file_path": fp, **_sidecar_entry_data(entry), **stat_file(fp)}
                generate_thumbnail(fp)
                sync_metadata(payload)
                triaged_completed += 1
            except Exception as exc:
                logger.warning("[BATCH] retry-sync failed for %s: %s", fp, exc)
            continue

        # New file → goes through the batched pipeline below.
        new_paths.append(fp)
        try:
            _upsert_processing_log(fp, "running")
        except Exception:
            pass

    if not new_paths:
        try:
            _CURRENT_FILE_PATH.write_text("")
        except Exception:
            pass
        if triaged_completed:
            _heartbeat._state["completed_since_start"] += triaged_completed
        heartbeat_updates = {
            "current_file": "",
            "current_started_ts": None,
            "paused_memory": False,
        }
        if triaged_completed:
            heartbeat_updates["last_completed_ts"] = time.time()
        _heartbeat.touch(
            **heartbeat_updates,
        )
        return

    # ── Build entries and run the staged pipeline ───────────────────────────
    clock.add("scan", time.perf_counter() - _scan_t0)
    entries = [_BatchEntry(file_path=p) for p in new_paths]

    # Pipeline: batches of >= 8 photos run as a 4-stage double-buffered
    # pipeline so CPU steps for the second half overlap with GPU steps for
    # the first half. Smaller batches run serially because the orchestration
    # overhead dwarfs the win.
    mode = "pipelined" if len(entries) >= 8 else "serial"
    _batch_t0 = time.perf_counter()
    with metrics.timer("worker.batch", mode=mode, n=len(entries)) as _t_batch:
        if mode == "pipelined":
            _run_batch_pipelined(entries, feats, clock)
        else:
            _run_batch_serial(entries, feats, clock)
    _batch_wall_sec = time.perf_counter() - _batch_t0

    # Cleanup.
    try:
        _CURRENT_FILE_PATH.write_text("")
    except Exception:
        pass

    n_ok = sum(1 for e in entries if not e.error)
    n_fail = len(entries) - n_ok
    completed_count = triaged_completed + n_ok
    _heartbeat._state["completed_since_start"] += completed_count
    heartbeat_updates = {
        "current_file": "",
        "current_started_ts": None,
        "paused_memory": False,
    }
    if completed_count:
        heartbeat_updates["last_completed_ts"] = time.time()
    _heartbeat.touch(**heartbeat_updates)
    metrics.count("worker.batch.photos_ok", n_ok, mode=mode)
    metrics.count("worker.batch.photos_failed", n_fail, mode=mode)
    metrics.record("worker.batch.size", len(entries), mode=mode)
    logger.info("[BATCH] done: %d ok, %d failed", n_ok, n_fail)

    # ── Persistent throughput record (one line per batch) ───────────────────
    try:
        total_bytes = sum(int(e.payload.get("file_size") or 0) for e in entries if e.payload)
        # Wall-clock = triage scan + batch run. Per-stage clock entries may sum
        # higher in pipelined mode (phases overlap); total_sec is authoritative.
        total_sec = clock.as_dict().get("scan", 0.0) + _batch_wall_sec
        throughput.log_run(throughput.build_record(
            mode=cfg["preset"],
            files=len(entries),
            total_bytes=total_bytes,
            total_sec=total_sec,
            enabled_features=feats,
            stage_sec=clock.as_dict(),
            config_load_error=cfg.get("config_load_error", False),
        ))
    except Exception as exc:
        logger.warning("[BATCH] throughput log failed: %s", exc)

    # Hardware safeguard at end-of-batch (mirrors process_media).
    if _monitor.pressure() == "critical":
        try:
            rss = _monitor.rss_percent() * 100
        except RuntimeError:
            rss = -1.0
        logger.warning(
            "[RAM SAFEGUARD] System RAM at %.1f%% post-batch — sleeping briefly.",
            rss,
        )
        metrics.count("worker.ram_safeguard.triggered", mode="batch")
        _gpu_cache_clear()
        gc.collect()
