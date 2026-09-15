"""Memoria – FastAPI Backend Entry Point.

In plain English: this file *is* the API server. It boots uvicorn on port
8000, registers every ``/api/v1/...`` route (delegating most to
``routers.py``), serves cached thumbnails and previews, and runs the
metrics middleware that times every request.

Notable startup choices:
  • A 10 GB on-disk cache of generated thumbnails / previews / face crops
    is swept on boot — oldest files are deleted until the total drops
    below the budget.
  • The pgvector type codec is registered on every new asyncpg
    connection so we can pass Python lists of floats into ``vector(N)``
    columns and pull them back out as the same.
  • A background task flushes the in-process metrics to
    ``~/.memoria/metrics/api.json`` every minute so the bench harness can
    pick them up across processes.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import sqlite3
import time as _time
from collections import deque as _deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncpg
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import metrics
from constants import is_image
from database_config import DATABASE_URL
from image_utils import (
    FACE_CROP_CACHE_DIR,
    FACE_CROP_MAX_DIM,
    PREVIEW_CACHE_DIR,
    THUMB_CACHE_DIR,
    make_face_crop as _generate_face_crop,
    make_preview as _generate_preview,
    make_thumbnail as _generate_thumbnail,
)
from runtime_paths import (
    CURRENT_FILE_PATH,
    QUEUE_DB_PATH,
    SCAN_SESSION_PATH,
    WORKER_HEARTBEAT_PATH,
    media_is_read_only,
)
from library_paths import validate_media_source


def _safe_media_source(path: str) -> str | None:
    try:
        return validate_media_source(path)
    except ValueError:
        return None


def _read_vram_usage() -> dict | None:
    """Return GPU memory usage as ``{backend, allocated_mb, driver_mb}``.

    On CUDA we use ``torch.cuda.mem_get_info()`` which queries the driver
    for *system-wide* free/total VRAM — the worker process holds the
    allocations, but the FastAPI process needs to report them. Per-process
    counters (``memory_allocated``/``memory_reserved``) would always show 0
    here because uvicorn doesn't touch the GPU itself.

    On MPS we keep the per-process call; ``torch.mps.driver_allocated_memory``
    is already system-wide on Apple Silicon's unified pool.

    Never raises — callers treat None as "unavailable".
    """
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = max(0, total_bytes - free_bytes)
            return {
                "backend": "cuda",
                "allocated_mb": used_bytes // (1024 * 1024),
                "driver_mb": total_bytes // (1024 * 1024),
            }
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "backend": "mps",
                "allocated_mb": torch.mps.current_allocated_memory() // (1024 * 1024),
                "driver_mb": torch.mps.driver_allocated_memory() // (1024 * 1024),
            }
    except Exception:
        pass
    return None


def _read_ram_usage() -> dict | None:
    """Return host RAM usage in MiB, or ``None`` when probing is unavailable."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        mib = 1024 * 1024
        return {
            "used_mb": int(memory.used // mib),
            "available_mb": int(memory.available // mib),
            "total_mb": int(memory.total // mib),
            "percent": float(memory.percent),
        }
    except Exception:
        return None

logger = logging.getLogger("memoria.api")
_metrics_logger = logging.getLogger("memoria.metrics")
_SCENE_ENV_FINGERPRINT_SETTING = "ml.scene.last_env_fingerprint"


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return a finite, bounded environment float or the safe default."""
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    if (
        not math.isfinite(value)
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        return default
    return value


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return a bounded environment integer, falling back when malformed."""
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    if minimum is not None and value < minimum:
        return default
    if maximum is not None and value > maximum:
        return default
    return value

# ── Progress endpoint state ──────────────────────────────────────────────────
# Module-level deque for ETA/rate calculation, and a small TTL cache for the
# done-count query (avoids hitting Postgres on every 5-second poll).
_progress_rate_samples: _deque = _deque(maxlen=120)  # (ts, done) pairs
_progress_done_cache: dict = {"ts": 0.0, "value": 0, "session_started_at": None}
_PROGRESS_DONE_TTL = 5.0  # seconds

# Stall detection thresholds (env-tunable).
_HEARTBEAT_STALE_S = _env_float("MEMORIA_HEARTBEAT_STALE_S", 120.0, minimum=1.0)
_STALL_TASK_S = _env_float("MEMORIA_STALL_TASK_S", 600.0, minimum=1.0)
_STALL_NO_COMPLETE_S = _env_float(
    "MEMORIA_STALL_NO_COMPLETE_S", 900.0, minimum=1.0
)

# ── /stats TTL cache ─────────────────────────────────────────────────────────
# The 7 COUNT queries in get_stats are expensive at scale (esp. COUNT DISTINCT
# over face_embeddings). Cache the Postgres-derived block for 30 s; queue size,
# current file, RAM, and VRAM stay live every call.
_STATS_TTL_S = _env_float("MEMORIA_STATS_TTL_S", 30.0, minimum=0.0)
_BLUR_THRESHOLD = _env_float(
    "MEMORIA_BLUR_THRESHOLD", 0.75, minimum=0.0, maximum=1.0
)

_stats_cache: dict = {"ts": 0.0, "data": None}
_stats_cache_lock: asyncio.Lock | None = None


def _get_stats_lock() -> asyncio.Lock:
    global _stats_cache_lock
    if _stats_cache_lock is None:
        _stats_cache_lock = asyncio.Lock()
    return _stats_cache_lock

# A startup sweep evicts oldest files once the total exceeds CACHE_MAX_BYTES.
# Sized for ~159k photos: thumbnails ≈ 30 KB × 159k = ~4.5 GB, plus a working
# set of 2048px previews. 10 GB lets both coexist without LRU thrashing.
# Override via MEMORIA_CACHE_MAX_GB env var.
_CACHE_MAX_GB = _env_float("MEMORIA_CACHE_MAX_GB", 10.0, minimum=0.0)
CACHE_MAX_BYTES = int(_CACHE_MAX_GB * 1024 * 1024 * 1024)


def _sweep_cache_if_over_budget() -> None:
    """Keep the three cache dirs under CACHE_MAX_BYTES, evicting oldest files
    by mtime. Called once on startup; no periodic loop."""
    files: list[tuple[Path, int, float]] = []
    for d in (THUMB_CACHE_DIR, PREVIEW_CACHE_DIR, FACE_CROP_CACHE_DIR):
        if not d.exists():
            continue
        for f in d.iterdir():
            if not f.is_file():
                continue
            try:
                st = f.stat()
                files.append((f, st.st_size, st.st_mtime))
            except OSError:
                pass
    total = sum(size for _, size, _ in files)
    if total <= CACHE_MAX_BYTES:
        return
    files.sort(key=lambda t: t[2])  # oldest first
    freed = 0
    for path, size, _ in files:
        try:
            path.unlink()
            freed += size
            total -= size
        except OSError:
            continue
        if total <= CACHE_MAX_BYTES:
            break
    logger.info(
        "Cache sweep freed %.1f MB (cap %d GB).",
        freed / (1024 * 1024), CACHE_MAX_BYTES // (1024**3),
    )


QUEUE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
HUEY_DB = str(QUEUE_DB_PATH)


async def _periodic_metrics_flush(interval_s: float = 60.0):
    """Emit ``metrics.log_summary`` every ``interval_s`` seconds.

    Started by lifespan; cancelled on shutdown. Catches every exception so a
    bad histogram can't kill the loop — observability must never crash the
    server.
    """
    try:
        while True:
            await asyncio.sleep(interval_s)
            try:
                metrics.log_summary(_metrics_logger)
            except Exception:
                logger.exception("metrics flush failed")
    except asyncio.CancelledError:
        # Final flush before shutdown.
        try:
            metrics.log_summary(_metrics_logger)
        except Exception:
            logger.exception("metrics shutdown flush failed")
        raise


async def _cancel_background_task(
    task: asyncio.Task | None,
    *,
    label: str,
) -> None:
    """Cancel and drain a lifespan-owned task without leaking its exception."""
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("%s task failed during shutdown", label)


async def _close_db_pool(pool) -> None:
    """Close an asyncpg pool, terminating it if checked-out sockets stall."""
    if pool is None:
        return
    try:
        await asyncio.wait_for(pool.close(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning(
            "Database pool did not close within 10 seconds; terminating sockets"
        )
        pool.terminate()
    except asyncio.CancelledError:
        pool.terminate()
        raise
    except Exception:
        logger.exception("Database pool shutdown failed; terminating sockets")
        pool.terminate()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    # Register the pgvector codec on every pooled connection so HTTP handlers
    # can read `vector` columns and bind Python lists to `<=>` parameters
    # (e.g. /api/v1/faces/{id}/similar, /api/v1/search/*). The worker registers
    # its own connection in workers/tasks.py:cluster_faces — this is the
    # asyncpg-side mirror.
    #
    # pgvector is a schema invariant as of migration 012. Fail pool creation
    # immediately if its codec cannot be registered; advertising a healthy API
    # whose core face/scene/dedup routes all fail is more dangerous than an
    # actionable startup error (including when uvicorn is run outside start.sh).
    from pgvector.asyncpg import register_vector

    async def _init_conn(conn):
        await register_vector(conn)

    app.state.db = await asyncpg.create_pool(
        DATABASE_URL,
        init=_init_conn,
        min_size=5,
        max_size=20,
        command_timeout=30,
    )

    # Register environment-selected model identities without loading weights.
    # If scene configuration changed, retain the previous active vector space
    # and enqueue a resumable worker migration for the desired model.
    async def _record_model_reconciliation_error(
        conn, kind: str, label: str, exc: Exception
    ) -> None:
        logger.exception("%s model reconciliation failed: %s", label, exc)
        try:
            await conn.execute(
                """
                UPDATE ml_model_state
                SET status = 'failed', last_error = $1, updated_at = NOW()
                WHERE kind = $2
                """,
                str(exc)[:4000],
                kind,
            )
        except Exception as state_exc:
            logger.warning(
                "Could not persist %s model reconciliation error: %s",
                label,
                state_exc,
            )

    async def _reconcile_ml_models() -> None:
        import model_store

        async with app.state.db.acquire() as conn:
            # Scene and face engines are separate failure domains. A malformed
            # setting or unavailable driver in one must not suppress recovery
            # and re-index scheduling for the other.
            try:
                from constants import IMAGE_EXTENSIONS
                from scene_engine import configured_model_spec

                scene_job_signal = None
                async with conn.transaction():
                    scene_spec = configured_model_spec()
                    scene_model = await model_store.register_model_async(conn, scene_spec)
                    prior_env_fingerprint = await conn.fetchval(
                        "SELECT value #>> '{}' FROM app_settings WHERE key = $1",
                        _SCENE_ENV_FINGERPRINT_SETTING,
                    )
                    scene_state = await model_store.get_model_state_async(conn, "scene")

                    # An unchanged boot environment must not undo a model
                    # selected and promoted through the UI.  A changed model
                    # fingerprint is an explicit environment transition and
                    # remains authoritative, preserving Phase 2 env control.
                    if (
                        scene_state is None
                        or prior_env_fingerprint != scene_model["fingerprint"]
                    ):
                        scene_state = await model_store.reconcile_model_state_async(
                            conn, "scene", scene_model["model_key"]
                        )
                    await conn.execute(
                        """
                        INSERT INTO app_settings (key, value, updated_at)
                        VALUES ($1, to_jsonb($2::text), NOW())
                        ON CONFLICT (key) DO UPDATE
                        SET value = EXCLUDED.value, updated_at = NOW()
                        """,
                        _SCENE_ENV_FINGERPRINT_SETTING,
                        scene_model["fingerprint"],
                    )

                    target_model = scene_state.get("desired_model") or scene_model
                    target_model_key = target_model["model_key"]
                    extensions = sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS)
                    missing_scene_items = await conn.fetchval(
                        """
                        SELECT COUNT(*)
                        FROM media m
                        LEFT JOIN scene_embeddings se
                          ON se.media_id = m.id AND se.model_key = $1
                        WHERE m.extension = ANY($2::text[]) AND se.media_id IS NULL
                        """,
                        target_model_key,
                        extensions,
                    )
                    needs_scene_reindex = (
                        scene_state.get("active_model_key") != target_model_key
                        or int(missing_scene_items or 0) > 0
                    )
                    auto_reindex = os.getenv(
                        "MEMORIA_SCENE_AUTO_REINDEX", "1"
                    ).strip().lower()
                    if (
                        needs_scene_reindex
                        and auto_reindex not in {"0", "false", "no", "off"}
                    ):
                        batch_size = _env_int(
                            "MEMORIA_SCENE_REINDEX_BATCH_SIZE",
                            16,
                            minimum=1,
                            maximum=64,
                        )
                        job, created = await model_store.create_reindex_job_async(
                            conn,
                            kind="scene",
                            model_key=target_model_key,
                            total_items=int(missing_scene_items or 0),
                            batch_size=batch_size,
                        )
                        scene_job_signal = (
                            job,
                            created,
                            scene_state.get("active_model_key"),
                            target_model_key,
                        )

                if scene_job_signal is not None:
                    # Enqueue only after the state/job transaction commits.
                    # Re-enqueuing an active job is a crash-recovery signal;
                    # the single worker makes duplicate continuations no-ops.
                    from workers.tasks import reindex_scene_embeddings

                    job, created, active_model_key, target_model_key = scene_job_signal
                    reindex_scene_embeddings(str(job["id"]))
                    logger.info(
                        "%s scene model migration %s -> %s (job %s)",
                        "Queued" if created else "Resumed",
                        active_model_key,
                        target_model_key,
                        job["id"],
                    )
            except Exception as exc:
                await _record_model_reconciliation_error(conn, "scene", "Scene", exc)

            try:
                from constants import IMAGE_EXTENSIONS
                from ml.face import (
                    FACENET_VGGFACE2_SPEC,
                    INSIGHTFACE_ARCFACE_SPEC,
                    FaceEngineConfig,
                )

                face_config = FaceEngineConfig.from_env()
                # Catalog the detector/embedding pipeline independently from
                # the vector space it writes. This keeps both runtime contract
                # and embedding provenance self-describing.
                await model_store.register_model_async(
                    conn, face_config.to_model_spec()
                )
                face_spec = (
                    FACENET_VGGFACE2_SPEC
                    if face_config.embedder == "facenet"
                    else INSIGHTFACE_ARCFACE_SPEC
                    if face_config.embedder == "insightface"
                    else None
                )
                if face_spec is not None:
                    face_model = await model_store.register_model_async(conn, face_spec)
                    face_state = await model_store.reconcile_model_state_async(
                        conn, "face-embedding", face_model["model_key"]
                    )
                    extensions = sorted(
                        ext.lstrip(".") for ext in IMAGE_EXTENSIONS
                    )
                    missing_face_items = await conn.fetchval(
                        """
                        SELECT COUNT(*)
                        FROM media m
                        LEFT JOIN ml_processing_state ps
                          ON ps.media_id = m.id
                         AND ps.kind = 'face-embedding'
                         AND ps.model_key = $1
                         AND ps.status = 'completed'
                        WHERE m.extension = ANY($2::text[])
                          AND ps.media_id IS NULL
                        """,
                        face_model["model_key"],
                        extensions,
                    )
                    needs_face_reprocess = (
                        face_state.get("active_model_key") != face_model["model_key"]
                        or int(missing_face_items or 0) > 0
                    )
                    if needs_face_reprocess:
                        # Reprocess is model-scoped and the one-worker queue
                        # makes duplicate restart signals harmless.
                        from workers.tasks import reprocess_faces

                        reprocess_faces()
                        logger.info(
                            "Queued face model migration %s -> %s (%d incomplete media)",
                            face_state.get("active_model_key"),
                            face_model["model_key"],
                            int(missing_face_items or 0),
                        )
            except Exception as exc:
                await _record_model_reconciliation_error(
                    conn, "face-embedding", "Face", exc
                )

    # Recover an interrupted clustering pass for the active vector namespace.
    # Legacy/alternate model clusters must not suppress this check.
    async def _maybe_cluster_faces():
        try:
            async with app.state.db.acquire() as conn:
                n_unclustered = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM face_embeddings fe
                    JOIN ml_model_state state
                      ON state.kind = 'face-embedding'
                     AND state.active_model_key = fe.embedding_model_key
                    WHERE fe.cluster_id IS NULL
                    """
                )
            if n_unclustered:
                logger.info(
                    "Startup: %d active-model faces remain unclustered — "
                    "triggering cluster_faces.",
                    n_unclustered,
                )
                from workers.tasks import cluster_faces
                cluster_faces()
        except Exception as exc:
            logger.warning("Startup cluster check failed: %s", exc)

    flush_task: asyncio.Task | None = None
    cluster_task: asyncio.Task | None = None
    try:
        await _reconcile_ml_models()
        for d in (THUMB_CACHE_DIR, PREVIEW_CACHE_DIR, FACE_CROP_CACHE_DIR):
            d.mkdir(parents=True, exist_ok=True)
        CURRENT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.get_running_loop().run_in_executor(
            None,
            _sweep_cache_if_over_budget,
        )
        flush_task = asyncio.create_task(
            _periodic_metrics_flush(),
            name="memoria-metrics-flush",
        )
        cluster_task = asyncio.create_task(
            _maybe_cluster_faces(),
            name="memoria-startup-face-cluster-check",
        )
        yield
    finally:
        await _cancel_background_task(
            cluster_task,
            label="startup face-cluster check",
        )
        await _cancel_background_task(
            flush_task,
            label="metrics flush",
        )
        pool = app.state.db
        app.state.db = None
        await _close_db_pool(pool)


app = FastAPI(title="Memoria", version="2.0.0", lifespan=lifespan)

_DEFAULT_ORIGINS = "http://localhost:5004,http://127.0.0.1:5004"
_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

_ALLOWED_HOSTS = {h.strip().lower().strip("[]") for h in os.getenv(
    "MEMORIA_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]"
).split(",") if h.strip()}


@app.middleware("http")
async def _request_boundary_middleware(request: Request, call_next):
    host = request.headers.get("host", "").split(":", 1)[0].lower().strip("[]")
    if host and host not in _ALLOWED_HOSTS:
        return JSONResponse(status_code=400, content={"detail": "untrusted host"})
    origin = request.headers.get("origin")
    if origin and origin not in _ALLOWED_ORIGINS:
        return JSONResponse(status_code=403, content={"detail": "untrusted origin"})
    if request.method in {"POST", "PUT", "PATCH"} and request.url.path.startswith("/api/"):
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            return JSONResponse(status_code=415, content={"detail": "application/json required"})
    return await call_next(request)


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """Time every HTTP request and tag by route template + method + status.

    Uses ``request.scope['route'].path`` (the template like ``/api/v1/photos/{id}``)
    when available so cardinality stays bounded even with high-id traffic.
    """
    loop = asyncio.get_running_loop()
    start = loop.time()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed_ms = (loop.time() - start) * 1000.0
        # Resolve route template post-dispatch so FastAPI has filled in
        # request.scope["route"]. Falls back to raw path for unmatched routes.
        route = request.scope.get("route")
        route_path = getattr(route, "path", None) or request.url.path
        try:
            metrics.record(
                "api.request.duration_ms",
                elapsed_ms,
                route=route_path,
                method=request.method,
            )
            metrics.count(
                "api.request.count",
                route=route_path,
                method=request.method,
                status=str(status_code),
            )
            if status_code >= 500:
                metrics.count("api.request.error", route=route_path)
        except Exception:
            # Never let metrics break the response.
            pass

# ── Register v1 API routers (all query Postgres / filesystem directly) ─────
from routers import ALL_ROUTERS  # noqa: E402
for _router in ALL_ROUTERS:
    app.include_router(_router)


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Metrics (live API snapshot + worker/watcher snapshot files) ───────────
@app.post("/api/v1/metrics/reset")
async def reset_metrics():
    """Wipe API-side metrics and enqueue a worker task to wipe its own.

    Used by ``backend/bench.py`` so each benchmark run gets a clean slate.
    No auth — intended for local-only use.
    """
    metrics.reset()
    # Stale on-disk snapshots would otherwise show up in the next /metrics
    # call until the periodic flush overwrites them.
    for name in ("worker", "watcher"):
        try:
            (metrics.SNAPSHOT_DIR / f"{name}.json").unlink(missing_ok=True)
        except OSError:
            pass
    try:
        from workers.tasks import reset_metrics as _worker_reset
        _worker_reset()  # enqueues the Huey task
        worker_queued = True
    except Exception as exc:
        logger.warning("Failed to enqueue worker reset_metrics: %s", exc)
        worker_queued = False
    return {"ok": True, "worker_reset_enqueued": worker_queued}


@app.get("/api/v1/metrics")
async def get_metrics():
    """Return live API-process metrics + the latest worker/watcher snapshots.

    Histograms expose ``n / mean / p50 / p95 / p99 / min / max``; counters
    expose cumulative ``value``. Series keys are flat strings of the form
    ``name|tag=value|tag=value`` with tags sorted alphabetically.

    The worker (huey_consumer) and watcher run in separate processes, so
    their timings can't share an in-memory store with this server. Each
    one writes a JSON snapshot to ``~/.memoria/metrics/<name>.json`` every
    30 s; this endpoint reads those files and returns them under
    ``processes``.
    """
    api_snapshot = metrics.snapshot()
    api_snapshot["_name"] = "api"
    processes: dict[str, dict] = {"api": api_snapshot}
    try:
        external = metrics.read_snapshots_dir()
    except Exception:
        logger.exception("failed to read external metrics snapshots")
        external = {}
    processes.update(external)
    # Combined view: union of every series across processes. For counters,
    # sum across processes (assume they don't double-publish the same key).
    # For histograms, keep them separate per-process — sliding-window
    # percentiles can't be merged.
    merged_counters: dict[str, dict] = {}
    merged_histograms: dict[str, dict] = {}
    for name, snap in processes.items():
        for key, val in snap.get("counters", {}).items():
            cur = merged_counters.get(key)
            if cur is None:
                merged_counters[key] = dict(val) | {"sources": [name]}
            else:
                cur["value"] += val.get("value", 0)
                cur["last_ts"] = max(cur.get("last_ts", 0), val.get("last_ts", 0))
                cur["sources"].append(name)
        for key, val in snap.get("histograms", {}).items():
            tagged_key = f"{key}|process={name}" if "|" in key else f"{key}|process={name}"
            merged_histograms[tagged_key] = val
    return {
        "processes": processes,
        "combined": {
            "counters": merged_counters,
            "histograms": merged_histograms,
        },
    }


# ── Stats ────────────────────────────────────────────────────────────────────
@app.get("/api/v1/stats")
async def get_stats(request: Request):
    """Return indexing stats plus live queue, file, RAM, and VRAM state.

    Postgres-derived counts are TTL-cached (default 30 s) to avoid 7 COUNT
    queries every 5 s dashboard poll. Queue and system-memory values stay live.
    """
    now = _time.time()

    # Postgres block — only run when the TTL has expired.
    async with _get_stats_lock():
        if now - _stats_cache["ts"] >= _STATS_TTL_S or _stats_cache["data"] is None:
            async with request.app.state.db.acquire() as conn:
                media_row = await conn.fetchrow(
                    "SELECT COUNT(*) AS n, COALESCE(SUM(file_size), 0) AS bytes FROM media"
                )
                total_indexed: int = (media_row["n"] if media_row else 0) or 0
                total_bytes_indexed: int = int((media_row["bytes"] if media_row else 0) or 0)
                # COUNT DISTINCT via subquery so the planner can use idx_face_media_id.
                faces_indexed: int = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT DISTINCT fe.media_id
                        FROM face_embeddings fe
                        JOIN ml_model_state state
                          ON state.kind = 'face-embedding'
                         AND state.active_model_key = fe.embedding_model_key
                    ) s
                    """
                ) or 0
                faces_total: int = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM face_embeddings fe
                    JOIN ml_model_state state
                      ON state.kind = 'face-embedding'
                     AND state.active_model_key = fe.embedding_model_key
                    """
                ) or 0
                face_clusters_count: int = await conn.fetchval(
                    """
                    SELECT COUNT(DISTINCT fe.cluster_id)
                    FROM face_embeddings fe
                    JOIN ml_model_state state
                      ON state.kind = 'face-embedding'
                     AND state.active_model_key = fe.embedding_model_key
                    JOIN face_clusters fc
                      ON fc.id = fe.cluster_id AND fc.is_hidden = FALSE
                    """
                ) or 0
                blur_scored: int = await conn.fetchval(
                    "SELECT COUNT(*) FROM media WHERE blur_score > 0"
                ) or 0
                blur_flagged: int = await conn.fetchval(
                    "SELECT COUNT(*) FROM media WHERE blur_score > $1",
                    _BLUR_THRESHOLD,
                ) or 0
                scene_row = await conn.fetchrow(
                    """
                    SELECT s.active_model_key,
                           COUNT(e.media_id) AS indexed
                    FROM ml_model_state s
                    LEFT JOIN scene_embeddings e
                      ON e.model_key = s.active_model_key
                    WHERE s.kind = 'scene'
                    GROUP BY s.active_model_key
                    """
                )
                scene_indexed: int = int(scene_row["indexed"] or 0) if scene_row else 0
                scene_model_key = scene_row["active_model_key"] if scene_row else None
                scene_job = await conn.fetchrow(
                    """
                    SELECT id, status, total_items, processed_items, failed_items,
                           skipped_items,
                           model_key AS target_model_key
                    FROM ml_reindex_jobs
                    WHERE kind = 'scene'
                    ORDER BY requested_at DESC
                    LIMIT 1
                    """
                )
            _stats_cache["data"] = {
                "total_indexed": total_indexed,
                "total_bytes_indexed": total_bytes_indexed,
                "faces_indexed": faces_indexed,
                "faces_total": faces_total,
                "face_clusters_count": face_clusters_count,
                "blur_scored": blur_scored,
                "blur_flagged": blur_flagged,
                "scene_indexed": scene_indexed,
                "scene_model_key": scene_model_key,
                "scene_reindex": dict(scene_job) if scene_job else None,
            }
            _stats_cache["ts"] = now
        cached = _stats_cache["data"]

    total_indexed = cached["total_indexed"]
    total_bytes_indexed = cached["total_bytes_indexed"]
    faces_indexed = cached["faces_indexed"]
    faces_total = cached["faces_total"]
    face_clusters_count = cached["face_clusters_count"]
    blur_scored = cached["blur_scored"]
    blur_flagged = cached["blur_flagged"]
    scene_indexed = cached["scene_indexed"]
    scene_model_key = cached.get("scene_model_key")
    scene_reindex = cached.get("scene_reindex")

    # Live data — queue size, current file, RAM, and VRAM.
    def _read_queue_size() -> int:
        try:
            with sqlite3.connect(HUEY_DB, timeout=5) as con:
                task_row = con.execute("SELECT COUNT(*) FROM task").fetchone()
                pending = int(task_row[0]) if task_row else 0
                try:
                    schedule_row = con.execute(
                        "SELECT COUNT(*) FROM schedule"
                    ).fetchone()
                    scheduled = int(schedule_row[0]) if schedule_row else 0
                except sqlite3.OperationalError:
                    scheduled = 0
                return pending + scheduled
        except Exception:
            return 0

    def _read_live_system_state() -> tuple[str, object, object]:
        cf = ""
        try:
            if CURRENT_FILE_PATH.exists():
                cf = CURRENT_FILE_PATH.read_text().strip()
        except Exception:
            pass
        return cf, _read_vram_usage(), _read_ram_usage()

    queue_size, (current_file, vram_usage, ram_usage) = await asyncio.gather(
        asyncio.to_thread(_read_queue_size),
        asyncio.to_thread(_read_live_system_state),
    )

    return {
        "total_indexed": total_indexed,
        "total_bytes_indexed": total_bytes_indexed,
        "queue_size": queue_size,
        "current_file": current_file,
        "vram_usage": vram_usage,
        "ram_usage": ram_usage,
        "faces_indexed": faces_indexed,
        "faces_total": faces_total,
        "face_clusters": face_clusters_count,
        "face_scan_active": queue_size > 0 and faces_indexed < total_indexed,
        "blur_scored": blur_scored,
        "blur_flagged": blur_flagged,
        "blur_scan_active": queue_size > 0 and blur_scored < total_indexed,
        "scene_indexed": scene_indexed,
        "scene_model_key": scene_model_key,
        "scene_reindex": scene_reindex,
        "scene_scan_active": (
            bool(
                scene_reindex
                and scene_reindex.get("status")
                in {"queued", "running", "indexing", "cancel_requested"}
            )
            or (queue_size > 0 and scene_indexed < total_indexed)
        ),
        "media_read_only": media_is_read_only(),
    }


# ── Media Endpoints ──────────────────────────────────────────────────────────
@app.get("/api/v1/media")
async def list_media(request: Request, offset: int = 0, limit: int = 100):
    """Return paginated media entries for the virtual grid."""
    async with request.app.state.db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, filename, directory FROM media
            WHERE file_path NOT LIKE '%/.[^/]%'
            ORDER BY directory ASC, indexed_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
        total: int = await conn.fetchval(
            "SELECT COUNT(*) FROM media WHERE file_path NOT LIKE '%/.[^/]%'"
        )
    items = [
        {
            "id": row["id"],
            "filename": row["filename"],
            "folder": row["directory"],
            "thumbnailUrl": f"/api/v1/media/{row['id']}/thumbnail",
        }
        for row in rows
    ]
    return {"items": items, "offset": offset, "limit": limit, "total": total}


@app.get("/api/v1/media/{media_id}")
async def get_media(media_id: int, request: Request):
    """Return metadata for a single media item."""
    async with request.app.state.db.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM media WHERE id = $1", media_id)
    if not row:
        return JSONResponse(status_code=404, content={"detail": "not found"})
    return dict(row)


@app.get("/api/v1/media/{media_id}/thumbnail")
async def get_thumbnail(media_id: int, request: Request):
    """Serve a cached 240×240 JPEG thumbnail.

    `make_thumbnail` (via `decode_pil`) handles RAW + HEIC + AVIF in addition
    to JPEG/PNG/etc., so the gate below uses the broader `is_image` set from
    constants — videos / unknown extensions return 404 (browser can't render
    raw bytes inline anyway).
    """
    async with request.app.state.db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT file_path FROM media WHERE id = $1", media_id
        )
    if not row:
        return JSONResponse(status_code=404, content={"detail": "not found"})
    file_path = _safe_media_source(row["file_path"])
    if not file_path:
        return JSONResponse(status_code=404, content={"detail": "file not found on disk"})

    # Only generate thumbnails for known image formats. Videos / unknown
    # extensions get 404 — the original bytes wouldn't render in <img> anyway.
    if not is_image(file_path):
        return JSONResponse(
            status_code=404,
            content={"detail": "not a supported image format"},
        )

    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(file_path.encode()).hexdigest()
    cached = THUMB_CACHE_DIR / f"{cache_key}.jpg"

    if cached.exists():
        metrics.count("cache.thumbnail.hit")
    else:
        metrics.count("cache.thumbnail.miss")
        try:
            loop = asyncio.get_running_loop()
            with metrics.timer("api.thumbnail.regen"):
                await loop.run_in_executor(None, _generate_thumbnail, file_path, cached)
        except Exception as exc:
            # Pillow / rawpy / pillow-heif raised — corrupted file or
            # unsupported sub-format. Return 500 so the frontend treats
            # it as a hard error (vs serving raw bytes the browser can't
            # render).
            logger.warning("Thumbnail generation failed for %s: %s", file_path, exc)
            return JSONResponse(
                status_code=500,
                content={"detail": "thumbnail decode failed"},
            )

    return FileResponse(
        cached,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/v1/media/{media_id}/face-crop")
async def get_face_crop(
    media_id: int,
    request: Request,
    x: int = Query(..., ge=-FACE_CROP_MAX_DIM, le=FACE_CROP_MAX_DIM),
    y: int = Query(..., ge=-FACE_CROP_MAX_DIM, le=FACE_CROP_MAX_DIM),
    w: int = Query(..., gt=0, le=FACE_CROP_MAX_DIM),
    h: int = Query(..., gt=0, le=FACE_CROP_MAX_DIM),
):
    """Serve a 160×160 JPEG crop of a face region with padding for context.

    Bbox params are bounds-checked by FastAPI (|x|,|y| <= FACE_CROP_MAX_DIM;
    0 < w,h <= FACE_CROP_MAX_DIM). Negative x/y are valid when a face sits
    near an edge and padding pushes the box outside the frame — make_face_crop
    clamps to the image dimensions before passing to Pillow.
    """
    async with request.app.state.db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT file_path FROM media WHERE id = $1", media_id
        )
    if not row:
        return JSONResponse(status_code=404, content={"detail": "not found"})
    file_path = _safe_media_source(row["file_path"])
    if not file_path:
        return JSONResponse(status_code=404, content={"detail": "file not found on disk"})
    if not is_image(file_path):
        return JSONResponse(status_code=400, content={"detail": "unsupported format for crop"})

    FACE_CROP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{file_path}:{x}:{y}:{w}:{h}".encode()).hexdigest()
    cached = FACE_CROP_CACHE_DIR / f"{cache_key}.jpg"

    if cached.exists():
        metrics.count("cache.face_crop.hit")
    else:
        metrics.count("cache.face_crop.miss")
        try:
            loop = asyncio.get_running_loop()
            with metrics.timer("api.face_crop.regen"):
                await loop.run_in_executor(
                    None, _generate_face_crop, file_path, cached, x, y, w, h
                )
        except Exception:
            return JSONResponse(status_code=500, content={"detail": "crop failed"})

    return FileResponse(
        cached,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ── Paginated Photos (v1) — Keyset Pagination (Spec 04) ──────────────────────

def _row_to_photo(row) -> dict:
    exif_raw = row.get("exif_data")
    # asyncpg returns JSONB as dict; only parse string for legacy compat.
    if isinstance(exif_raw, dict):
        exif = exif_raw
    elif isinstance(exif_raw, str):
        try:
            exif = json.loads(exif_raw)
        except Exception:
            exif = {}
    else:
        exif = {}

    return {
        "id": row["id"],
        "filename": row["filename"],
        "folder": row["directory"],
        "thumbnailUrl": f"/api/v1/media/{row['id']}/thumbnail",
        "previewUrl": f"/api/v1/preview/{row['id']}",
        "phash": row["phash"] or "",
        "blur_score": float(row["blur_score"] or 0),
        "exif": exif,
        "indexed_at": row["indexed_at"].isoformat() if row["indexed_at"] else None,
    }


def _row_to_photo_slim(row) -> dict:
    """Minimal list-view shape — no exif/phash (saves ~1-2 KB/item × 300k rows)."""
    return {
        "id": row["id"],
        "filename": row["filename"],
        "folder": row["directory"],
        "thumbnailUrl": f"/api/v1/media/{row['id']}/thumbnail",
        "previewUrl": f"/api/v1/preview/{row['id']}",
        "blur_score": float(row["blur_score"] or 0),
        "indexed_at": row["indexed_at"].isoformat() if row["indexed_at"] else None,
    }


@app.get("/api/v1/photos")
async def list_photos(
    request: Request,
    limit: int = 100,
    cursor_indexed_at: Optional[str] = Query(None),
    cursor_id: Optional[int] = Query(None),
    offset: Optional[int] = Query(None),
):
    """Return paginated photo entries using keyset (cursor) pagination.

    Backward compatibility: if ``offset`` is provided (and no cursor), falls
    back to the legacy limit/offset query with a deprecation warning.
    """
    limit = min(limit, 200)

    # Legacy fallback — offset-based pagination (deprecated).
    if offset is not None and cursor_indexed_at is None:
        logger.warning("DEPRECATED: /api/v1/photos called with offset=%d — migrate to cursor pagination.", offset)
        async with request.app.state.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, filename, directory, phash, blur_score, exif_data, indexed_at
                FROM media
                WHERE file_path NOT LIKE '%/.[^/]%%'
                ORDER BY indexed_at DESC, id DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
            total: int = await conn.fetchval(
                "SELECT COUNT(*) FROM media WHERE file_path NOT LIKE '%/.[^/]%%'"
            )
        items = []
        for r in rows:
            try:
                items.append(_row_to_photo(r))
            except Exception:
                logger.exception("Failed to convert row to photo: %r", dict(r))
                continue
        return {"items": items, "offset": offset, "limit": limit, "total": total}

    # Keyset pagination — slim payload (no exif/phash; detail endpoint has them).
    async with request.app.state.db.acquire() as conn:
        if cursor_indexed_at is not None and cursor_id is not None:
            rows = await conn.fetch(
                """
                SELECT id, filename, directory, blur_score, indexed_at
                FROM media
                WHERE file_path NOT LIKE '%/.[^/]%%'
                  AND (indexed_at, id) < ($2::timestamptz, $3::bigint)
                ORDER BY indexed_at DESC, id DESC
                LIMIT $1
                """,
                limit,
                datetime.fromisoformat(cursor_indexed_at),
                cursor_id,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, filename, directory, blur_score, indexed_at
                FROM media
                WHERE file_path NOT LIKE '%/.[^/]%%'
                ORDER BY indexed_at DESC, id DESC
                LIMIT $1
                """,
                limit,
            )

    items = []
    for r in rows:
        try:
            items.append(_row_to_photo_slim(r))
        except Exception:
            logger.exception("Failed to convert row to photo: %r", dict(r))
            continue
    has_more = len(rows) == limit
    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        next_cursor = {
            "indexed_at": last["indexed_at"].isoformat() if last["indexed_at"] else None,
            "id": last["id"],
        }

    return {"items": items, "next_cursor": next_cursor, "has_more": has_more}


@app.get("/api/v1/photos/{media_id}")
async def get_photo(request: Request, media_id: int):
    """Return a single photo item by its media ID."""
    async with request.app.state.db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, filename, directory, phash, blur_score, exif_data, indexed_at FROM media WHERE id = $1",
            media_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Photo not found")
    return _row_to_photo(row)


# ── Library Version ──────────────────────────────────────────────────────────
@app.get("/api/v1/library/version")
async def library_version(request: Request):
    """Cheap polling endpoint: returns total count + latest item cursor.

    The frontend polls this every 30 s instead of refetching all infinite-query
    pages. When count or latest.id changes, the UI shows a "New photos" pill.
    """
    async with request.app.state.db.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM media") or 0
        latest_row = await conn.fetchrow(
            "SELECT id, indexed_at FROM media ORDER BY indexed_at DESC, id DESC LIMIT 1"
        )
    latest = None
    if latest_row:
        latest = {
            "id": latest_row["id"],
            "indexed_at": latest_row["indexed_at"].isoformat() if latest_row["indexed_at"] else None,
        }
    return {"count": count, "latest": latest}


# ── Progress / Stall Detection ───────────────────────────────────────────────
@app.get("/api/v1/progress")
async def get_progress(request: Request):
    """Ingest progress with stall classification.

    Returns state: idle | scanning | paused_memory | stalled.
    Reads ~/.memoria/scan_session.json (denominator) and
    ~/.memoria/worker_heartbeat.json (worker liveness) atomically.
    """
    now = _time.time()

    # Read session file (denominator).
    session_data: dict | None = None
    session_path = SCAN_SESSION_PATH
    try:
        session_data = json.loads(await asyncio.to_thread(session_path.read_text))
    except Exception:
        pass

    # Read heartbeat file.
    hb: dict | None = None
    hb_path = WORKER_HEARTBEAT_PATH
    try:
        hb = json.loads(await asyncio.to_thread(hb_path.read_text))
    except Exception:
        pass

    # Done count (TTL-cached to avoid Postgres on every 5s poll).
    global _progress_done_cache
    session_started_at = session_data.get("started_at") if session_data else None
    done = 0
    if session_started_at:
        cache_hit = (
            _progress_done_cache["session_started_at"] == session_started_at
            and (now - _progress_done_cache["ts"]) < _PROGRESS_DONE_TTL
        )
        if cache_hit:
            done = _progress_done_cache["value"]
        else:
            try:
                async with request.app.state.db.acquire() as conn:
                    done = await conn.fetchval(
                        "SELECT COUNT(*) FROM processing_log"
                        " WHERE status IN ('synced','failed')"
                        " AND updated_at >= $1::timestamptz",
                        session_started_at,
                    ) or 0
                _progress_done_cache = {
                    "ts": now,
                    "value": done,
                    "session_started_at": session_started_at,
                }
            except Exception:
                done = _progress_done_cache.get("value", 0)

    total = int(session_data.get("enqueued", 0)) if session_data else 0
    percent = round(done / total * 100, 1) if total > 0 else 0.0

    # Rate / ETA.
    _progress_rate_samples.append((now, done))
    files_per_min: float | None = None
    eta_seconds: float | None = None
    if len(_progress_rate_samples) >= 2:
        oldest_ts, oldest_done = _progress_rate_samples[0]
        span_s = now - oldest_ts
        if span_s >= 60:
            rate = (done - oldest_done) / (span_s / 60)
            if rate > 0:
                files_per_min = round(rate, 1)
                eta_seconds = round((total - done) / rate * 60) if total > done else 0

    # Queue size (non-blocking).
    def _get_q() -> int:
        try:
            with sqlite3.connect(HUEY_DB, timeout=3) as con:
                task_row = con.execute("SELECT COUNT(*) FROM task").fetchone()
                pending = int(task_row[0]) if task_row else 0
                try:
                    schedule_row = con.execute(
                        "SELECT COUNT(*) FROM schedule"
                    ).fetchone()
                    scheduled = int(schedule_row[0]) if schedule_row else 0
                except sqlite3.OperationalError:
                    scheduled = 0
                return pending + scheduled
        except Exception:
            return 0

    queue_tasks = await asyncio.to_thread(_get_q)

    # Worker fields.
    worker_alive = False
    current_file = ""
    current_file_age_s: float | None = None
    last_completed_age_s: float | None = None
    paused_memory = False
    pressure = "ok"
    if hb:
        written_ts = hb.get("written_ts") or 0
        worker_alive = (now - written_ts) < _HEARTBEAT_STALE_S
        current_file = hb.get("current_file") or ""
        paused_memory = bool(hb.get("paused_memory"))
        pressure = hb.get("pressure") or "ok"
        if hb.get("current_started_ts"):
            current_file_age_s = round(now - hb["current_started_ts"], 1)
        if hb.get("last_completed_ts"):
            last_completed_age_s = round(now - hb["last_completed_ts"], 1)

    # State machine.
    reasons: list[str] = []
    if queue_tasks == 0 and not current_file:
        state = "idle"
    elif paused_memory and worker_alive:
        state = "paused_memory"
    elif queue_tasks > 0 or current_file:
        if hb and (now - (hb.get("written_ts") or 0)) > _HEARTBEAT_STALE_S:
            reasons.append("heartbeat_stale")
        if current_file_age_s is not None and current_file_age_s > _STALL_TASK_S:
            reasons.append("task_running_too_long")
        if (
            last_completed_age_s is None or last_completed_age_s > _STALL_NO_COMPLETE_S
        ) and (queue_tasks > 0 or current_file):
            if not current_file_age_s or current_file_age_s > _STALL_NO_COMPLETE_S:
                reasons.append("no_completions")
        state = "stalled" if reasons else "scanning"
    else:
        state = "idle"

    return {
        "state": state,
        "reasons": reasons,
        "session": {
            "id": session_data.get("session_id") if session_data else None,
            "started_at": session_started_at,
            "finished_at": session_data.get("finished_at") if session_data else None,
            "total": total,
            "done": done,
            "percent": percent,
        },
        "queue": {"tasks": queue_tasks},
        "worker": {
            "alive": worker_alive,
            "current_file": current_file,
            "current_file_age_s": current_file_age_s,
            "last_completed_age_s": last_completed_age_s,
            "paused_memory": paused_memory,
            "pressure": pressure,
        },
        "rate": {
            "files_per_min": files_per_min,
            "eta_seconds": eta_seconds,
        },
    }


# ── Queue Stats (Spec 05 C1) ────────────────────────────────────────────────
@app.get("/api/v1/queue/stats")
async def queue_stats(request: Request):
    """Return live Huey queued/scheduled depth plus active and failed counts."""
    def _read_queue_depths() -> tuple[int, int]:
        try:
            with sqlite3.connect(HUEY_DB, timeout=3) as con:
                task_row = con.execute("SELECT COUNT(*) FROM task").fetchone()
                pending = int(task_row[0]) if task_row else 0
                try:
                    schedule_row = con.execute(
                        "SELECT COUNT(*) FROM schedule"
                    ).fetchone()
                    scheduled = int(schedule_row[0]) if schedule_row else 0
                except sqlite3.OperationalError:
                    # Older/partially initialised Huey databases may not have
                    # created the delayed-task table yet.
                    scheduled = 0
                return pending, scheduled
        except Exception:
            return 0, 0

    def _read_running_task() -> int:
        try:
            return int(bool(CURRENT_FILE_PATH.read_text().strip()))
        except (FileNotFoundError, OSError):
            return 0

    queue_depths, running = await asyncio.gather(
        asyncio.to_thread(_read_queue_depths),
        asyncio.to_thread(_read_running_task),
    )
    pending, scheduled = queue_depths
    async with request.app.state.db.acquire() as conn:
        failed = await conn.fetchval(
            "SELECT COUNT(*) FROM processing_log WHERE status = 'failed'"
        ) or 0
    return {
        "pending": pending,
        "scheduled": scheduled,
        "running": running,
        "failed": failed,
    }


# ── 2048px Preview ───────────────────────────────────────────────────────────
@app.get("/api/v1/preview/{photo_id}")
async def get_preview(photo_id: int, request: Request):
    """Serve a cached 2048px WebP preview; generate on-the-fly if missing."""
    async with request.app.state.db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT file_path FROM media WHERE id = $1", photo_id
        )
    if not row:
        return JSONResponse(status_code=404, content={"detail": "not found"})
    file_path = _safe_media_source(row["file_path"])
    if not file_path:
        return JSONResponse(status_code=404, content={"detail": "file not found on disk"})

    # Videos / unknown formats: 404. `make_preview` handles JPEG/PNG/RAW/HEIC
    # via decode_pil; everything else can't render in <img> anyway.
    if not is_image(file_path):
        return JSONResponse(
            status_code=404,
            content={"detail": "not a supported image format"},
        )

    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(file_path.encode()).hexdigest()
    cached = PREVIEW_CACHE_DIR / f"{cache_key}.webp"

    if cached.exists():
        metrics.count("cache.preview.hit")
    else:
        metrics.count("cache.preview.miss")
        try:
            loop = asyncio.get_running_loop()
            with metrics.timer("api.preview.regen"):
                await loop.run_in_executor(None, _generate_preview, file_path, cached)
        except Exception as exc:
            logger.warning("Preview generation failed for %s: %s", file_path, exc)
            return JSONResponse(
                status_code=500,
                content={"detail": "preview decode failed"},
            )

    return FileResponse(
        cached,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
