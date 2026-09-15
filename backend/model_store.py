"""Persistence helpers for the pluggable ML model architecture.

This module is deliberately framework-free: importing it never imports torch,
ONNX Runtime, OpenCLIP, or Transformers.  Sync functions accept an existing
psycopg connection (the worker path); ``*_async`` functions accept an existing
asyncpg connection (the API path).

Model rows are immutable.  ``register_model`` is idempotent only when the
stored fingerprint matches the supplied descriptor.  Scene vectors are
validated before reaching PostgreSQL, and kNN SQL uses hard-coded predicates
only for known built-in indexes; arbitrary custom keys remain bound parameters
and use an exact-search fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from psycopg.rows import dict_row


logger = logging.getLogger("memoria.model_store")


DEFAULT_SCENE_MODEL_KEY = "scene/open_clip/ViT-L-14/laion2b_s32b_b82k"
OPENCLIP_SIGLIP_MODEL_KEY = (
    "scene/open_clip/ViT-SO400M-14-SigLIP-384/webli"
)
HF_SIGLIP2_MODEL_KEY = "scene/hf/google/siglip2-so400m-patch14-384"
DEFAULT_FACE_MODEL_KEY = "face/insightface-buffalo-l-arcface"
FACENET_MODEL_KEY = "face/facenet-vggface2"
LEGACY_FACE_MODEL_KEY = "face/legacy-unknown-512"


# These are the only model keys interpolated into a partial-index predicate.
# Every other key is a bind parameter and therefore cannot become SQL syntax.
_BUILTIN_SCENE_INDEX_DIMS: dict[str, int] = {
    DEFAULT_SCENE_MODEL_KEY: 768,
    OPENCLIP_SIGLIP_MODEL_KEY: 1152,
    HF_SIGLIP2_MODEL_KEY: 1152,
}
_BUILTIN_MODEL_SQL_LITERALS: dict[str, str] = {
    DEFAULT_SCENE_MODEL_KEY: "'scene/open_clip/ViT-L-14/laion2b_s32b_b82k'",
    OPENCLIP_SIGLIP_MODEL_KEY: (
        "'scene/open_clip/ViT-SO400M-14-SigLIP-384/webli'"
    ),
    HF_SIGLIP2_MODEL_KEY: "'scene/hf/google/siglip2-so400m-patch14-384'",
}

_TERMINAL_JOB_STATUSES = frozenset({"cancelled", "completed", "failed"})
_ACTIVE_JOB_STATUSES = frozenset(
    {"queued", "running", "indexing", "cancel_requested"}
)
_ALL_JOB_STATUSES = _ACTIVE_JOB_STATUSES | _TERMINAL_JOB_STATUSES
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ModelStoreError(RuntimeError):
    """Base error raised for invalid or inconsistent persisted ML state."""


class ModelIdentityConflict(ModelStoreError):
    """A model_key already exists with a different immutable fingerprint."""


class ModelNotFound(ModelStoreError):
    """The requested model identity is not registered."""


class ModelStateConflict(ModelStoreError):
    """The requested state transition conflicts with current desired state."""


def _decode_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _model_row(row: Any) -> dict[str, Any]:
    result = dict(row)
    for field in ("provider_preferences", "metadata"):
        if field in result:
            result[field] = _decode_json(result[field])
    return result


def _state_row(row: Any) -> dict[str, Any]:
    result = dict(row)
    for field in ("desired_model", "active_model"):
        value = _decode_json(result.get(field))
        result[field] = _model_row(value) if isinstance(value, Mapping) else value
    return result


def _job_row(row: Any) -> dict[str, Any]:
    result = dict(row)
    if "metrics" in result:
        result["metrics"] = _decode_json(result["metrics"])
    return result


def _benchmark_row(row: Any) -> dict[str, Any]:
    result = dict(row)
    for field in ("query_latency_ms", "metrics"):
        if field in result:
            result[field] = _decode_json(result[field])
    return result


def _as_mapping(spec: Any) -> dict[str, Any]:
    if isinstance(spec, Mapping):
        return dict(spec)
    as_dict = getattr(spec, "as_dict", None)
    if callable(as_dict):
        value = as_dict()
        if isinstance(value, Mapping):
            return dict(value)
    raise TypeError("model descriptor must be a mapping or expose as_dict()")


def _as_uuid(value: Any) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"invalid UUID: {value!r}") from exc


def _first(raw: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return default


def _positive_dimension(value: Any, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError("embedding dimension must be an integer")
    try:
        dimension = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding dimension must be an integer") from exc
    if not 1 <= dimension <= 16_000:
        raise ValueError("embedding dimension must be between 1 and 16000")
    return dimension


def _input_resolution(raw: Mapping[str, Any]) -> list[int] | None:
    value = raw.get("input_resolution")
    if value is None:
        width = raw.get("input_width")
        height = raw.get("input_height")
        if width is None and height is None:
            return None
        value = [width, height]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("input_resolution must contain width and height")
    if len(value) != 2:
        raise ValueError("input_resolution must contain width and height")
    if any(isinstance(v, bool) for v in value):
        raise ValueError("input resolution values must be integers")
    try:
        result = [int(value[0]), int(value[1])]
    except (TypeError, ValueError) as exc:
        raise ValueError("input resolution values must be integers") from exc
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError("input resolution values must be positive")
    return result


def _canonical_fingerprint_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror ``ml.base.ModelSpec.fingerprint`` without importing an engine."""
    return {
        "model_key": model["model_key"],
        "task": model["kind"],
        "driver": model["driver"],
        "model_id": model["model_id"],
        "weights": model.get("weights"),
        "revision": model.get("revision"),
        "input_resolution": model.get("input_resolution"),
        "embedding_dim": model.get("embedding_dimension"),
        "default_precision": model["default_precision"],
        "provider_preferences": model["provider_preferences"],
        "distance_metric": model["distance_metric"],
        "normalized": model["normalized"],
        "index_kind": model["index_kind"],
        "metadata": model["metadata"],
    }


def normalize_model_descriptor(spec: Any) -> dict[str, Any]:
    """Return the immutable database representation of a ModelSpec-like value."""
    raw = _as_mapping(spec)
    model_key = str(_first(raw, "model_key", "key", default="")).strip()
    kind = str(_first(raw, "kind", "task", default="")).strip()
    driver = str(raw.get("driver") or "").strip()
    model_id = str(_first(raw, "model_name", "model_id", "name", default="")).strip()
    if not all((model_key, kind, driver, model_id)):
        raise ValueError("model_key, kind/task, driver, and model_name/model_id are required")

    dimension = _positive_dimension(
        _first(raw, "embedding_dimension", "embedding_dim", "output_dimension"),
        allow_none=True,
    )
    if kind in {"scene", "face-embedding"} and dimension is None:
        raise ValueError(f"{kind} models must declare an embedding dimension")

    providers = _first(
        raw,
        "preferred_providers",
        "provider_preferences",
        default=["cuda", "mps", "cpu"],
    )
    if not isinstance(providers, Sequence) or isinstance(providers, (str, bytes)):
        raise ValueError("provider preferences must be an ordered sequence")
    provider_list = list(dict.fromkeys(str(p).strip().lower() for p in providers))
    if not provider_list or any(not p for p in provider_list):
        raise ValueError("provider preferences cannot be empty")

    metadata = dict(raw.get("metadata") or {})
    # Validate serializability before opening a transaction.
    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    distance_metric = str(raw.get("distance_metric") or "cosine").strip().lower()
    if distance_metric not in {"cosine", "inner_product", "l2"}:
        raise ValueError(f"unsupported distance metric: {distance_metric!r}")
    index_kind = str(raw.get("index_kind") or "exact").strip().lower()
    if index_kind not in {"hnsw", "exact", "none"}:
        raise ValueError(f"unsupported index kind: {index_kind!r}")

    model = {
        "model_key": model_key,
        "kind": kind,
        "driver": driver,
        "model_id": model_id,
        "weights": _first(raw, "weights", "pretrained"),
        "revision": raw.get("revision"),
        "input_resolution": _input_resolution(raw),
        "embedding_dimension": dimension,
        "default_precision": str(raw.get("default_precision") or "auto").lower(),
        "provider_preferences": provider_list,
        "distance_metric": distance_metric,
        "normalized": bool(raw.get("normalized", True)),
        "index_kind": index_kind,
        "metadata": metadata,
        "is_builtin": bool(raw.get("is_builtin", False)),
    }
    supplied_fingerprint = str(raw.get("fingerprint") or "").lower()
    if supplied_fingerprint and not _HEX64.fullmatch(supplied_fingerprint):
        raise ValueError("model fingerprint must be a lowercase SHA-256 hex string")
    canonical = json.dumps(
        _canonical_fingerprint_payload(model),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if supplied_fingerprint and supplied_fingerprint != fingerprint:
        raise ValueError(
            "model fingerprint does not match the normalized descriptor fields"
        )
    model["fingerprint"] = fingerprint
    return model


_MODEL_COLUMNS = """
    model_key, kind, driver, model_id, weights, revision, input_resolution,
    embedding_dimension, default_precision, provider_preferences,
    distance_metric, normalized, index_kind, fingerprint, metadata, is_builtin
"""


def register_model(conn: Any, spec: Any) -> dict[str, Any]:
    """Idempotently register an immutable model using a psycopg connection."""
    model = normalize_model_descriptor(spec)
    with conn.transaction():
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                INSERT INTO ml_models ({_MODEL_COLUMNS})
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s, %s, %s, %s, %s::jsonb, %s
                )
                ON CONFLICT (model_key) DO NOTHING
                RETURNING *
                """,
                (
                    model["model_key"], model["kind"], model["driver"],
                    model["model_id"], model["weights"], model["revision"],
                    model["input_resolution"], model["embedding_dimension"],
                    model["default_precision"],
                    json.dumps(model["provider_preferences"]),
                    model["distance_metric"], model["normalized"],
                    model["index_kind"], model["fingerprint"],
                    json.dumps(model["metadata"]), model["is_builtin"],
                ),
            )
            row = cur.fetchone()
            created = row is not None
            if row is None:
                cur.execute("SELECT * FROM ml_models WHERE model_key = %s", (model["model_key"],))
                row = cur.fetchone()
    if row is None:
        raise ModelStoreError("model registration did not return a row")
    result = _model_row(row)
    if result["fingerprint"] != model["fingerprint"]:
        logger.error(
            "Rejected model identity conflict for model_key=%s",
            model["model_key"],
        )
        raise ModelIdentityConflict(
            f"model_key {model['model_key']!r} already has a different fingerprint"
        )
    if created:
        logger.info(
            "Registered %s model %s (dimension=%s)",
            model["kind"],
            model["model_key"],
            model["embedding_dimension"],
        )
    return result


async def register_model_async(conn: Any, spec: Any) -> dict[str, Any]:
    """Asyncpg counterpart to :func:`register_model`."""
    model = normalize_model_descriptor(spec)
    async with conn.transaction():
        row = await conn.fetchrow(
            f"""
            INSERT INTO ml_models ({_MODEL_COLUMNS})
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9,
                $10::jsonb, $11, $12, $13, $14, $15::jsonb, $16
            )
            ON CONFLICT (model_key) DO NOTHING
            RETURNING *
            """,
            model["model_key"], model["kind"], model["driver"],
            model["model_id"], model["weights"], model["revision"],
            model["input_resolution"], model["embedding_dimension"],
            model["default_precision"], json.dumps(model["provider_preferences"]),
            model["distance_metric"], model["normalized"], model["index_kind"],
            model["fingerprint"], json.dumps(model["metadata"]), model["is_builtin"],
        )
        created = row is not None
        if row is None:
            row = await conn.fetchrow(
                "SELECT * FROM ml_models WHERE model_key = $1", model["model_key"]
            )
    if row is None:
        raise ModelStoreError("model registration did not return a row")
    result = _model_row(row)
    if result["fingerprint"] != model["fingerprint"]:
        logger.error(
            "Rejected model identity conflict for model_key=%s",
            model["model_key"],
        )
        raise ModelIdentityConflict(
            f"model_key {model['model_key']!r} already has a different fingerprint"
        )
    if created:
        logger.info(
            "Registered %s model %s (dimension=%s)",
            model["kind"],
            model["model_key"],
            model["embedding_dimension"],
        )
    return result


def get_model(conn: Any, model_key: str) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM ml_models WHERE model_key = %s", (model_key,))
        row = cur.fetchone()
    return _model_row(row) if row else None


async def get_model_async(conn: Any, model_key: str) -> dict[str, Any] | None:
    row = await conn.fetchrow("SELECT * FROM ml_models WHERE model_key = $1", model_key)
    return _model_row(row) if row else None


def list_models(conn: Any, kind: str | None = None) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        if kind is None:
            cur.execute("SELECT * FROM ml_models ORDER BY kind, model_key")
        else:
            cur.execute(
                "SELECT * FROM ml_models WHERE kind = %s ORDER BY model_key", (kind,)
            )
        return [_model_row(row) for row in cur.fetchall()]


async def list_models_async(conn: Any, kind: str | None = None) -> list[dict[str, Any]]:
    if kind is None:
        rows = await conn.fetch("SELECT * FROM ml_models ORDER BY kind, model_key")
    else:
        rows = await conn.fetch(
            "SELECT * FROM ml_models WHERE kind = $1 ORDER BY model_key", kind
        )
    return [_model_row(row) for row in rows]


_STATE_SELECT = """
    SELECT s.*,
           CASE WHEN desired.model_key IS NULL THEN NULL ELSE to_jsonb(desired) END
               AS desired_model,
           CASE WHEN active.model_key IS NULL THEN NULL ELSE to_jsonb(active) END
               AS active_model
    FROM ml_model_state s
    LEFT JOIN ml_models desired ON desired.model_key = s.desired_model_key
    LEFT JOIN ml_models active ON active.model_key = s.active_model_key
"""


def get_model_state(conn: Any, kind: str) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_STATE_SELECT + " WHERE s.kind = %s", (kind,))
        row = cur.fetchone()
    return _state_row(row) if row else None


async def get_model_state_async(conn: Any, kind: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(_STATE_SELECT + " WHERE s.kind = $1", kind)
    return _state_row(row) if row else None


def list_model_states(conn: Any) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_STATE_SELECT + " ORDER BY s.kind")
        return [_state_row(row) for row in cur.fetchall()]


async def list_model_states_async(conn: Any) -> list[dict[str, Any]]:
    rows = await conn.fetch(_STATE_SELECT + " ORDER BY s.kind")
    return [_state_row(row) for row in rows]


def get_active_model(conn: Any, kind: str) -> dict[str, Any] | None:
    state = get_model_state(conn, kind)
    return state.get("active_model") if state else None


async def get_active_model_async(conn: Any, kind: str) -> dict[str, Any] | None:
    state = await get_model_state_async(conn, kind)
    return state.get("active_model") if state else None


def _assert_model_kind(model: Mapping[str, Any] | None, model_key: str, kind: str) -> None:
    if model is None:
        raise ModelNotFound(f"model {model_key!r} is not registered")
    if model["kind"] != kind:
        raise ModelStateConflict(
            f"model {model_key!r} has kind {model['kind']!r}, not {kind!r}"
        )


def reconcile_model_state(conn: Any, kind: str, desired_model_key: str) -> dict[str, Any]:
    """Make ``desired`` match configuration while preserving the ready active model."""
    model = get_model(conn, desired_model_key)
    _assert_model_kind(model, desired_model_key, kind)
    with conn.transaction():
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM ml_model_state WHERE kind = %s FOR UPDATE", (kind,)
            )
            current = cur.fetchone()
            if current is None:
                cur.execute(
                    """
                    INSERT INTO ml_model_state (
                        kind, desired_model_key, active_model_key, status, generation
                    ) VALUES (%s, %s, %s, 'ready', 1)
                    """,
                    (kind, desired_model_key, desired_model_key),
                )
            else:
                changed = current["desired_model_key"] != desired_model_key
                status = (
                    "ready"
                    if current["active_model_key"] == desired_model_key
                    else "building"
                )
                cur.execute(
                    """
                    UPDATE ml_model_state
                    SET desired_model_key = %s,
                        status = %s,
                        generation = generation + %s,
                        last_error = NULL,
                        updated_at = NOW()
                    WHERE kind = %s
                    """,
                    (desired_model_key, status, int(changed), kind),
                )
    state = get_model_state(conn, kind)
    if state is None:
        raise ModelStoreError("model state reconciliation did not persist")
    return state


async def reconcile_model_state_async(
    conn: Any, kind: str, desired_model_key: str
) -> dict[str, Any]:
    model = await get_model_async(conn, desired_model_key)
    _assert_model_kind(model, desired_model_key, kind)
    async with conn.transaction():
        current = await conn.fetchrow(
            "SELECT * FROM ml_model_state WHERE kind = $1 FOR UPDATE", kind
        )
        if current is None:
            await conn.execute(
                """
                INSERT INTO ml_model_state (
                    kind, desired_model_key, active_model_key, status, generation
                ) VALUES ($1, $2, $2, 'ready', 1)
                """,
                kind, desired_model_key,
            )
        else:
            changed = current["desired_model_key"] != desired_model_key
            status = "ready" if current["active_model_key"] == desired_model_key else "building"
            await conn.execute(
                """
                UPDATE ml_model_state
                SET desired_model_key = $1,
                    status = $2,
                    generation = generation + $3,
                    last_error = NULL,
                    updated_at = NOW()
                WHERE kind = $4
                """,
                desired_model_key, status, int(changed), kind,
            )
    state = await get_model_state_async(conn, kind)
    if state is None:
        raise ModelStoreError("model state reconciliation did not persist")
    return state


def activate_model(
    conn: Any, kind: str, model_key: str, *, require_desired: bool = True
) -> dict[str, Any]:
    model = get_model(conn, model_key)
    _assert_model_kind(model, model_key, kind)
    with conn.transaction():
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM ml_model_state WHERE kind = %s FOR UPDATE", (kind,)
            )
            state = cur.fetchone()
            if state is None:
                cur.execute(
                    """
                    INSERT INTO ml_model_state
                        (kind, desired_model_key, active_model_key, status, generation)
                    VALUES (%s, %s, %s, 'ready', 1)
                    """,
                    (kind, model_key, model_key),
                )
            else:
                if require_desired and state["desired_model_key"] != model_key:
                    raise ModelStateConflict(
                        f"cannot activate stale model {model_key!r}; desired is "
                        f"{state['desired_model_key']!r}"
                    )
                cur.execute(
                    """
                    UPDATE ml_model_state
                    SET active_model_key = %s, status = 'ready', last_error = NULL,
                        updated_at = NOW()
                    WHERE kind = %s
                    """,
                    (model_key, kind),
                )
    result = get_model_state(conn, kind)
    if result is None:
        raise ModelStoreError("model activation did not persist")
    return result


async def activate_model_async(
    conn: Any, kind: str, model_key: str, *, require_desired: bool = True
) -> dict[str, Any]:
    model = await get_model_async(conn, model_key)
    _assert_model_kind(model, model_key, kind)
    async with conn.transaction():
        state = await conn.fetchrow(
            "SELECT * FROM ml_model_state WHERE kind = $1 FOR UPDATE", kind
        )
        if state is None:
            await conn.execute(
                """
                INSERT INTO ml_model_state
                    (kind, desired_model_key, active_model_key, status, generation)
                VALUES ($1, $2, $2, 'ready', 1)
                """,
                kind, model_key,
            )
        else:
            if require_desired and state["desired_model_key"] != model_key:
                raise ModelStateConflict(
                    f"cannot activate stale model {model_key!r}; desired is "
                    f"{state['desired_model_key']!r}"
                )
            await conn.execute(
                """
                UPDATE ml_model_state
                SET active_model_key = $1, status = 'ready', last_error = NULL,
                    updated_at = NOW()
                WHERE kind = $2
                """,
                model_key, kind,
            )
    result = await get_model_state_async(conn, kind)
    if result is None:
        raise ModelStoreError("model activation did not persist")
    return result


def mark_model_state_failed(
    conn: Any, kind: str, model_key: str, error: str
) -> dict[str, Any] | None:
    """Record a failed build only if the model is still the desired target."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ml_model_state
            SET status = CASE
                    WHEN active_model_key = desired_model_key
                    THEN 'ready' ELSE 'failed'
                END,
                last_error = %s,
                updated_at = NOW()
            WHERE kind = %s AND desired_model_key = %s
            """,
            (str(error), kind, model_key),
        )
    return get_model_state(conn, kind)


async def mark_model_state_failed_async(
    conn: Any, kind: str, model_key: str, error: str
) -> dict[str, Any] | None:
    """Async counterpart of :func:`mark_model_state_failed`."""
    await conn.execute(
        """
        UPDATE ml_model_state
        SET status = CASE
                WHEN active_model_key = desired_model_key
                THEN 'ready' ELSE 'failed'
            END,
            last_error = $1,
            updated_at = NOW()
        WHERE kind = $2 AND desired_model_key = $3
        """,
        str(error), kind, model_key,
    )
    return await get_model_state_async(conn, kind)


def create_reindex_job(
    conn: Any,
    *,
    kind: str,
    model_key: str,
    total_items: int = 0,
    force_reindex: bool = False,
    batch_size: int | None = None,
) -> tuple[dict[str, Any], bool]:
    """Create or return the one active job for ``(kind, model_key)``.

    Returns ``(job, created)``. The partial unique index is the concurrency
    guard, so simultaneous API requests remain idempotent.
    """
    model = get_model(conn, model_key)
    _assert_model_kind(model, model_key, kind)
    total_items = max(0, int(total_items))
    if batch_size is not None and int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    with conn.transaction():
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO ml_reindex_jobs (
                    kind, model_key, total_items, force_reindex, batch_size
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING *
                """,
                (kind, model_key, total_items, force_reindex, batch_size),
            )
            row = cur.fetchone()
            created = row is not None
            if row is None:
                cur.execute(
                    """
                    SELECT * FROM ml_reindex_jobs
                    WHERE kind = %s AND model_key = %s
                      AND status IN ('queued','running','indexing','cancel_requested')
                    ORDER BY requested_at DESC LIMIT 1
                    """,
                    (kind, model_key),
                )
                row = cur.fetchone()
    if row is None:
        raise ModelStoreError("reindex job conflict occurred without an active job")
    return _job_row(row), created


async def create_reindex_job_async(
    conn: Any,
    *,
    kind: str,
    model_key: str,
    total_items: int = 0,
    force_reindex: bool = False,
    batch_size: int | None = None,
) -> tuple[dict[str, Any], bool]:
    model = await get_model_async(conn, model_key)
    _assert_model_kind(model, model_key, kind)
    total_items = max(0, int(total_items))
    if batch_size is not None and int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    async with conn.transaction():
        row = await conn.fetchrow(
            """
            INSERT INTO ml_reindex_jobs (
                kind, model_key, total_items, force_reindex, batch_size
            ) VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT DO NOTHING
            RETURNING *
            """,
            kind, model_key, total_items, force_reindex, batch_size,
        )
        created = row is not None
        if row is None:
            row = await conn.fetchrow(
                """
                SELECT * FROM ml_reindex_jobs
                WHERE kind = $1 AND model_key = $2
                  AND status IN ('queued','running','indexing','cancel_requested')
                ORDER BY requested_at DESC LIMIT 1
                """,
                kind, model_key,
            )
    if row is None:
        raise ModelStoreError("reindex job conflict occurred without an active job")
    return _job_row(row), created


def get_reindex_job(conn: Any, job_id: Any) -> dict[str, Any] | None:
    job_id = _as_uuid(job_id)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM ml_reindex_jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    return _job_row(row) if row else None


async def get_reindex_job_async(conn: Any, job_id: Any) -> dict[str, Any] | None:
    job_id = _as_uuid(job_id)
    row = await conn.fetchrow("SELECT * FROM ml_reindex_jobs WHERE id = $1", job_id)
    return _job_row(row) if row else None


def list_reindex_jobs(
    conn: Any,
    *,
    kind: str | None = None,
    model_key: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    clauses: list[str] = []
    params: list[Any] = []
    if kind is not None:
        clauses.append("kind = %s")
        params.append(kind)
    if model_key is not None:
        clauses.append("model_key = %s")
        params.append(model_key)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(limit)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM ml_reindex_jobs"
            f"{where} ORDER BY requested_at DESC LIMIT %s",
            params,
        )
        return [_job_row(row) for row in cur.fetchall()]


async def list_reindex_jobs_async(
    conn: Any,
    *,
    kind: str | None = None,
    model_key: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    clauses: list[str] = []
    params: list[Any] = []

    def bind(value: Any) -> str:
        params.append(value)
        return f"${len(params)}"

    if kind is not None:
        clauses.append(f"kind = {bind(kind)}")
    if model_key is not None:
        clauses.append(f"model_key = {bind(model_key)}")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    limit_param = bind(limit)
    rows = await conn.fetch(
        "SELECT * FROM ml_reindex_jobs"
        f"{where} ORDER BY requested_at DESC LIMIT {limit_param}",
        *params,
    )
    return [_job_row(row) for row in rows]


_JOB_VALUE_FIELDS = {
    "status", "cursor_media_id", "total_items", "processed_items",
    "failed_items", "skipped_items", "batch_size", "metrics", "error",
}


def _normalize_job_updates(updates: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(updates) - _JOB_VALUE_FIELDS
    if unknown:
        raise ValueError(f"unsupported reindex job fields: {sorted(unknown)}")
    result = dict(updates)
    if "status" in result and result["status"] not in _ALL_JOB_STATUSES:
        raise ValueError(f"invalid reindex status: {result['status']!r}")
    for key in (
        "cursor_media_id", "total_items", "processed_items", "failed_items",
        "skipped_items",
    ):
        if key in result:
            result[key] = int(result[key])
            if result[key] < 0:
                raise ValueError(f"{key} cannot be negative")
    if result.get("batch_size") is not None:
        result["batch_size"] = int(result["batch_size"])
        if result["batch_size"] <= 0:
            raise ValueError("batch_size must be positive")
    if "metrics" in result:
        result["metrics"] = dict(result["metrics"] or {})
    return result


def update_reindex_job(conn: Any, job_id: Any, **updates: Any) -> dict[str, Any]:
    job_id = _as_uuid(job_id)
    values = _normalize_job_updates(updates)
    if not values:
        job = get_reindex_job(conn, job_id)
        if job is None:
            raise ModelStoreError(f"reindex job {job_id!r} not found")
        return job
    assignments: list[str] = []
    params: list[Any] = []
    for field, value in values.items():
        assignments.append(f"{field} = %s" + ("::jsonb" if field == "metrics" else ""))
        params.append(json.dumps(value) if field == "metrics" else value)
    if values.get("status") == "running":
        assignments.append("started_at = COALESCE(started_at, NOW())")
    if values.get("status") in _TERMINAL_JOB_STATUSES:
        assignments.append("finished_at = COALESCE(finished_at, NOW())")
    assignments.append("updated_at = NOW()")
    params.append(job_id)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"UPDATE ml_reindex_jobs SET {', '.join(assignments)} WHERE id = %s RETURNING *",
            params,
        )
        row = cur.fetchone()
    if row is None:
        raise ModelStoreError(f"reindex job {job_id!r} not found")
    return _job_row(row)


async def update_reindex_job_async(
    conn: Any, job_id: Any, **updates: Any
) -> dict[str, Any]:
    job_id = _as_uuid(job_id)
    values = _normalize_job_updates(updates)
    if not values:
        job = await get_reindex_job_async(conn, job_id)
        if job is None:
            raise ModelStoreError(f"reindex job {job_id!r} not found")
        return job
    assignments: list[str] = []
    params: list[Any] = []
    for field, value in values.items():
        params.append(json.dumps(value) if field == "metrics" else value)
        cast = "::jsonb" if field == "metrics" else ""
        assignments.append(f"{field} = ${len(params)}{cast}")
    if values.get("status") == "running":
        assignments.append("started_at = COALESCE(started_at, NOW())")
    if values.get("status") in _TERMINAL_JOB_STATUSES:
        assignments.append("finished_at = COALESCE(finished_at, NOW())")
    assignments.append("updated_at = NOW()")
    params.append(job_id)
    row = await conn.fetchrow(
        f"UPDATE ml_reindex_jobs SET {', '.join(assignments)} "
        f"WHERE id = ${len(params)} RETURNING *",
        *params,
    )
    if row is None:
        raise ModelStoreError(f"reindex job {job_id!r} not found")
    return _job_row(row)


def start_reindex_job(conn: Any, job_id: Any) -> dict[str, Any]:
    job_id = _as_uuid(job_id)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE ml_reindex_jobs
            SET status = 'running', started_at = COALESCE(started_at, NOW()),
                updated_at = NOW()
            WHERE id = %s AND status = 'queued'
            RETURNING *
            """,
            (job_id,),
        )
        row = cur.fetchone()
    if row is not None:
        return _job_row(row)
    job = get_reindex_job(conn, job_id)
    if job is None:
        raise ModelStoreError(f"reindex job {job_id!r} not found")
    return job


async def start_reindex_job_async(conn: Any, job_id: Any) -> dict[str, Any]:
    job_id = _as_uuid(job_id)
    row = await conn.fetchrow(
        """
        UPDATE ml_reindex_jobs
        SET status = 'running', started_at = COALESCE(started_at, NOW()),
            updated_at = NOW()
        WHERE id = $1 AND status = 'queued'
        RETURNING *
        """,
        job_id,
    )
    if row is not None:
        return _job_row(row)
    job = await get_reindex_job_async(conn, job_id)
    if job is None:
        raise ModelStoreError(f"reindex job {job_id!r} not found")
    return job


def finish_reindex_job(
    conn: Any, job_id: Any, *, metrics: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    job_id = _as_uuid(job_id)
    metrics_json = json.dumps(dict(metrics)) if metrics is not None else None
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE ml_reindex_jobs
            SET status = 'completed', error = NULL,
                metrics = COALESCE(%s::jsonb, metrics),
                finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
            WHERE id = %s AND status IN ('running', 'indexing')
            RETURNING *
            """,
            (metrics_json, job_id),
        )
        row = cur.fetchone()
    if row is not None:
        return _job_row(row)
    job = get_reindex_job(conn, job_id)
    if job is None:
        raise ModelStoreError(f"reindex job {job_id!r} not found")
    return job


async def finish_reindex_job_async(
    conn: Any, job_id: Any, *, metrics: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    job_id = _as_uuid(job_id)
    metrics_json = json.dumps(dict(metrics)) if metrics is not None else None
    row = await conn.fetchrow(
        """
        UPDATE ml_reindex_jobs
        SET status = 'completed', error = NULL,
            metrics = COALESCE($1::jsonb, metrics),
            finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
        WHERE id = $2 AND status IN ('running', 'indexing')
        RETURNING *
        """,
        metrics_json, job_id,
    )
    if row is not None:
        return _job_row(row)
    job = await get_reindex_job_async(conn, job_id)
    if job is None:
        raise ModelStoreError(f"reindex job {job_id!r} not found")
    return job


def fail_reindex_job(conn: Any, job_id: Any, error: str) -> dict[str, Any]:
    job_id = _as_uuid(job_id)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE ml_reindex_jobs
            SET status = 'failed', error = %s,
                finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
            WHERE id = %s AND status IN ('queued', 'running', 'indexing')
            RETURNING *
            """,
            (str(error), job_id),
        )
        row = cur.fetchone()
    if row is not None:
        return _job_row(row)
    job = get_reindex_job(conn, job_id)
    if job is None:
        raise ModelStoreError(f"reindex job {job_id!r} not found")
    return job


async def fail_reindex_job_async(conn: Any, job_id: Any, error: str) -> dict[str, Any]:
    job_id = _as_uuid(job_id)
    row = await conn.fetchrow(
        """
        UPDATE ml_reindex_jobs
        SET status = 'failed', error = $1,
            finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
        WHERE id = $2 AND status IN ('queued', 'running', 'indexing')
        RETURNING *
        """,
        str(error), job_id,
    )
    if row is not None:
        return _job_row(row)
    job = await get_reindex_job_async(conn, job_id)
    if job is None:
        raise ModelStoreError(f"reindex job {job_id!r} not found")
    return job


def request_reindex_cancel(conn: Any, job_id: Any) -> dict[str, Any]:
    job_id = _as_uuid(job_id)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE ml_reindex_jobs
            SET status = 'cancel_requested', updated_at = NOW()
            WHERE id = %s AND status IN ('queued','running','indexing')
            RETURNING *
            """,
            (job_id,),
        )
        row = cur.fetchone()
    if row is not None:
        return _job_row(row)
    job = get_reindex_job(conn, job_id)
    if job is None:
        raise ModelStoreError(f"reindex job {job_id!r} not found")
    return job


async def request_reindex_cancel_async(conn: Any, job_id: Any) -> dict[str, Any]:
    job_id = _as_uuid(job_id)
    row = await conn.fetchrow(
        """
        UPDATE ml_reindex_jobs
        SET status = 'cancel_requested', updated_at = NOW()
        WHERE id = $1 AND status IN ('queued','running','indexing')
        RETURNING *
        """,
        job_id,
    )
    if row is not None:
        return _job_row(row)
    job = await get_reindex_job_async(conn, job_id)
    if job is None:
        raise ModelStoreError(f"reindex job {job_id!r} not found")
    return job


def cancel_reindex_job(conn: Any, job_id: Any) -> dict[str, Any]:
    """Mark a job cancelled after its worker has acknowledged the request."""
    job_id = _as_uuid(job_id)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE ml_reindex_jobs
            SET status = 'cancelled',
                finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
            WHERE id = %s
              AND status IN ('queued', 'running', 'indexing', 'cancel_requested')
            RETURNING *
            """,
            (job_id,),
        )
        row = cur.fetchone()
    if row is not None:
        return _job_row(row)
    job = get_reindex_job(conn, job_id)
    if job is None:
        raise ModelStoreError(f"reindex job {job_id!r} not found")
    return job


async def cancel_reindex_job_async(conn: Any, job_id: Any) -> dict[str, Any]:
    job_id = _as_uuid(job_id)
    row = await conn.fetchrow(
        """
        UPDATE ml_reindex_jobs
        SET status = 'cancelled',
            finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
        WHERE id = $1
          AND status IN ('queued', 'running', 'indexing', 'cancel_requested')
        RETURNING *
        """,
        job_id,
    )
    if row is not None:
        return _job_row(row)
    job = await get_reindex_job_async(conn, job_id)
    if job is None:
        raise ModelStoreError(f"reindex job {job_id!r} not found")
    return job


def finalize_reindex_and_activate(
    conn: Any,
    job_id: Any,
    *,
    metrics: Mapping[str, Any] | None = None,
    eligible_extensions: Sequence[str] | None = None,
    benchmark: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bool, int]:
    """Atomically validate coverage and cut over unless cancellation won.

    The job row is locked before the model-state row. A concurrent cancellation
    that commits first is acknowledged without changing the active model; a
    cancellation that arrives after this transaction commits observes an
    already-completed job. Scene and face cutovers briefly take a SHARE lock
    on ``media`` so coverage and promotion cannot straddle an ingest commit.

    Returns ``(job, activated, remaining_items)``. A non-forced job with
    late missing rows is rewound to cursor zero for its queued successor.
    When supplied, ``benchmark`` is inserted before this transaction commits,
    so a benchmark failure also rolls back the promotion and completion.
    """

    job_id = _as_uuid(job_id)
    metrics_json = json.dumps(dict(metrics)) if metrics is not None else None
    with conn.transaction():
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM ml_reindex_jobs WHERE id = %s FOR UPDATE",
                (job_id,),
            )
            job = cur.fetchone()
            if job is None:
                raise ModelStoreError(f"reindex job {job_id!r} not found")

            status = job["status"]
            if status in _TERMINAL_JOB_STATUSES:
                return _job_row(job), status == "completed", 0

            if status == "cancel_requested":
                cur.execute(
                    """
                    UPDATE ml_reindex_jobs
                    SET status = 'cancelled',
                        finished_at = COALESCE(finished_at, NOW()),
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (job_id,),
                )
                cancelled = cur.fetchone()
                cur.execute(
                    """
                    UPDATE ml_model_state
                    SET status = CASE
                            WHEN active_model_key = desired_model_key
                            THEN 'ready' ELSE 'failed'
                        END,
                        last_error = CASE
                            WHEN active_model_key = desired_model_key
                            THEN NULL ELSE 're-index cancelled'
                        END,
                        updated_at = NOW()
                    WHERE kind = %s AND desired_model_key = %s
                    """,
                    (job["kind"], job["model_key"]),
                )
                if cancelled is None:
                    raise ModelStoreError("cancelled reindex job did not return a row")
                return _job_row(cancelled), False, 0

            if status not in {"running", "indexing"}:
                raise ModelStateConflict(
                    f"cannot finalize reindex job {job_id!s} from status {status!r}"
                )

            cur.execute(
                "SELECT * FROM ml_model_state WHERE kind = %s FOR UPDATE",
                (job["kind"],),
            )
            state = cur.fetchone()
            if state is None:
                error = f"model state {job['kind']!r} is missing at re-index cutover"
                cur.execute(
                    """
                    UPDATE ml_reindex_jobs
                    SET status = 'failed', error = %s,
                        finished_at = COALESCE(finished_at, NOW()),
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (error, job_id),
                )
                failed = cur.fetchone()
                if failed is None:
                    raise ModelStoreError("failed reindex job did not return a row")
                return _job_row(failed), False, 0

            if state["desired_model_key"] != job["model_key"]:
                error = "configured model changed before re-index cutover"
                cur.execute(
                    """
                    UPDATE ml_reindex_jobs
                    SET status = 'cancelled', error = %s,
                        finished_at = COALESCE(finished_at, NOW()),
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (error, job_id),
                )
                cancelled = cur.fetchone()
                if cancelled is None:
                    raise ModelStoreError("cancelled reindex job did not return a row")
                return _job_row(cancelled), False, 0

            remaining = 0
            if job["kind"] in {"scene", "face-embedding"}:
                extensions = [str(value) for value in (eligible_extensions or ())]
                if not extensions:
                    raise ValueError(
                        "eligible_extensions are required for scene/face cutover"
                    )
                # Briefly block media INSERT/UPDATE transactions while checking
                # coverage and promoting the namespace. Readers remain online,
                # and an in-flight ingest either commits before this count or
                # waits until the new model is active.
                cur.execute("LOCK TABLE media IN SHARE MODE")
                if job["kind"] == "scene":
                    coverage_sql = """
                        SELECT COUNT(*) AS remaining
                        FROM media m
                        LEFT JOIN scene_embeddings se
                          ON se.media_id = m.id AND se.model_key = %s
                        WHERE m.extension = ANY(%s) AND se.media_id IS NULL
                    """
                else:
                    coverage_sql = """
                        SELECT COUNT(*) AS remaining
                        FROM media m
                        LEFT JOIN ml_processing_state ps
                          ON ps.media_id = m.id
                         AND ps.kind = 'face-embedding'
                         AND ps.model_key = %s
                         AND ps.status = 'completed'
                        WHERE m.extension = ANY(%s) AND ps.media_id IS NULL
                    """
                cur.execute(coverage_sql, (job["model_key"], extensions))
                coverage = cur.fetchone()
                if coverage:
                    remaining = int(coverage["remaining"] or 0)

            failed_items = int(job["failed_items"] or 0)
            if failed_items:
                error = (
                    f"{failed_items} media item(s) failed; "
                    f"{remaining} target item(s) remain missing"
                )
                cur.execute(
                    """
                    UPDATE ml_reindex_jobs
                    SET status = 'failed', error = %s,
                        metrics = COALESCE(%s::jsonb, metrics),
                        finished_at = COALESCE(finished_at, NOW()),
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (error, metrics_json, job_id),
                )
                failed = cur.fetchone()
                cur.execute(
                    """
                    UPDATE ml_model_state
                    SET status = CASE
                            WHEN active_model_key = desired_model_key
                            THEN 'ready' ELSE 'failed'
                        END,
                        last_error = %s,
                        updated_at = NOW()
                    WHERE kind = %s AND desired_model_key = %s
                    """,
                    (error, job["kind"], job["model_key"]),
                )
                if failed is None:
                    raise ModelStoreError("failed reindex job did not return a row")
                return _job_row(failed), False, remaining

            if remaining:
                if job["force_reindex"]:
                    error = (
                        f"forced re-index finished with {remaining} target "
                        "item(s) missing"
                    )
                    cur.execute(
                        """
                        UPDATE ml_reindex_jobs
                        SET status = 'failed', error = %s,
                            metrics = COALESCE(%s::jsonb, metrics),
                            finished_at = COALESCE(finished_at, NOW()),
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING *
                        """,
                        (error, metrics_json, job_id),
                    )
                    incomplete = cur.fetchone()
                    cur.execute(
                        """
                        UPDATE ml_model_state
                        SET status = CASE
                                WHEN active_model_key = desired_model_key
                                THEN 'ready' ELSE 'failed'
                            END,
                            last_error = %s,
                            updated_at = NOW()
                        WHERE kind = %s AND desired_model_key = %s
                        """,
                        (error, job["kind"], job["model_key"]),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE ml_reindex_jobs
                        SET cursor_media_id = 0,
                            total_items = total_items + %s,
                            metrics = COALESCE(%s::jsonb, metrics),
                            error = NULL,
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING *
                        """,
                        (remaining, metrics_json, job_id),
                    )
                    incomplete = cur.fetchone()
                if incomplete is None:
                    raise ModelStoreError("incomplete reindex job did not return a row")
                return _job_row(incomplete), False, remaining

            processed_items = int(job["processed_items"] or 0)
            failed_items = int(job["failed_items"] or 0)
            skipped_items = int(job["skipped_items"] or 0)
            total_items = int(job["total_items"] or 0)
            if not job["force_reindex"]:
                # Normal ingestion may satisfy a target vector while this job
                # is scanning. Preserve encoder throughput separately and
                # report those externally completed rows as skipped.
                skipped_items = max(
                    skipped_items,
                    total_items - processed_items - failed_items,
                    0,
                )
            total_items = max(
                total_items,
                processed_items + failed_items + skipped_items,
            )

            cur.execute(
                """
                UPDATE ml_model_state
                SET active_model_key = %s, status = 'ready', last_error = NULL,
                    updated_at = NOW()
                WHERE kind = %s
                """,
                (job["model_key"], job["kind"]),
            )
            if benchmark is not None:
                benchmark_payload = dict(benchmark)
                benchmark_payload["model_key"] = job["model_key"]
                benchmark_payload["reindex_job_id"] = job_id
                # record_model_benchmark does not open or commit a transaction;
                # called here it participates in this cutover transaction. Any
                # insert failure therefore rolls back both promotion and job
                # completion, allowing the Huey retry to repeat the cutover.
                record_model_benchmark(conn, **benchmark_payload)
            cur.execute(
                """
                UPDATE ml_reindex_jobs
                SET status = 'completed', error = NULL,
                    metrics = COALESCE(%s::jsonb, metrics),
                    total_items = %s, skipped_items = %s,
                    finished_at = COALESCE(finished_at, NOW()),
                    updated_at = NOW()
                WHERE id = %s
                RETURNING *
                """,
                (metrics_json, total_items, skipped_items, job_id),
            )
            completed = cur.fetchone()
            if completed is None:
                raise ModelStoreError("completed reindex job did not return a row")
            return _job_row(completed), True, 0


async def finalize_reindex_and_activate_async(
    conn: Any,
    job_id: Any,
    *,
    metrics: Mapping[str, Any] | None = None,
    eligible_extensions: Sequence[str] | None = None,
    benchmark: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bool, int]:
    """Async counterpart to :func:`finalize_reindex_and_activate`."""

    job_id = _as_uuid(job_id)
    metrics_json = json.dumps(dict(metrics)) if metrics is not None else None
    async with conn.transaction():
        job = await conn.fetchrow(
            "SELECT * FROM ml_reindex_jobs WHERE id = $1 FOR UPDATE",
            job_id,
        )
        if job is None:
            raise ModelStoreError(f"reindex job {job_id!r} not found")

        status = job["status"]
        if status in _TERMINAL_JOB_STATUSES:
            return _job_row(job), status == "completed", 0

        if status == "cancel_requested":
            cancelled = await conn.fetchrow(
                """
                UPDATE ml_reindex_jobs
                SET status = 'cancelled',
                    finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
                WHERE id = $1
                RETURNING *
                """,
                job_id,
            )
            await conn.execute(
                """
                UPDATE ml_model_state
                SET status = CASE
                        WHEN active_model_key = desired_model_key
                        THEN 'ready' ELSE 'failed'
                    END,
                    last_error = CASE
                        WHEN active_model_key = desired_model_key
                        THEN NULL ELSE 're-index cancelled'
                    END,
                    updated_at = NOW()
                WHERE kind = $1 AND desired_model_key = $2
                """,
                job["kind"], job["model_key"],
            )
            if cancelled is None:
                raise ModelStoreError("cancelled reindex job did not return a row")
            return _job_row(cancelled), False, 0

        if status not in {"running", "indexing"}:
            raise ModelStateConflict(
                f"cannot finalize reindex job {job_id!s} from status {status!r}"
            )

        state = await conn.fetchrow(
            "SELECT * FROM ml_model_state WHERE kind = $1 FOR UPDATE",
            job["kind"],
        )
        if state is None:
            error = f"model state {job['kind']!r} is missing at re-index cutover"
            failed = await conn.fetchrow(
                """
                UPDATE ml_reindex_jobs
                SET status = 'failed', error = $1,
                    finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
                WHERE id = $2
                RETURNING *
                """,
                error, job_id,
            )
            if failed is None:
                raise ModelStoreError("failed reindex job did not return a row")
            return _job_row(failed), False, 0

        if state["desired_model_key"] != job["model_key"]:
            error = "configured model changed before re-index cutover"
            cancelled = await conn.fetchrow(
                """
                UPDATE ml_reindex_jobs
                SET status = 'cancelled', error = $1,
                    finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
                WHERE id = $2
                RETURNING *
                """,
                error, job_id,
            )
            if cancelled is None:
                raise ModelStoreError("cancelled reindex job did not return a row")
            return _job_row(cancelled), False, 0

        remaining = 0
        if job["kind"] in {"scene", "face-embedding"}:
            extensions = [str(value) for value in (eligible_extensions or ())]
            if not extensions:
                raise ValueError(
                    "eligible_extensions are required for scene/face cutover"
                )
            await conn.execute("LOCK TABLE media IN SHARE MODE")
            if job["kind"] == "scene":
                coverage_sql = """
                    SELECT COUNT(*) AS remaining
                    FROM media m
                    LEFT JOIN scene_embeddings se
                      ON se.media_id = m.id AND se.model_key = $1
                    WHERE m.extension = ANY($2::text[]) AND se.media_id IS NULL
                """
            else:
                coverage_sql = """
                    SELECT COUNT(*) AS remaining
                    FROM media m
                    LEFT JOIN ml_processing_state ps
                      ON ps.media_id = m.id
                     AND ps.kind = 'face-embedding'
                     AND ps.model_key = $1
                     AND ps.status = 'completed'
                    WHERE m.extension = ANY($2::text[]) AND ps.media_id IS NULL
                """
            coverage = await conn.fetchrow(
                coverage_sql, job["model_key"], extensions,
            )
            if coverage:
                remaining = int(coverage["remaining"] or 0)

        failed_items = int(job["failed_items"] or 0)
        if failed_items:
            error = (
                f"{failed_items} media item(s) failed; "
                f"{remaining} target item(s) remain missing"
            )
            failed = await conn.fetchrow(
                """
                UPDATE ml_reindex_jobs
                SET status = 'failed', error = $1,
                    metrics = COALESCE($2::jsonb, metrics),
                    finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
                WHERE id = $3
                RETURNING *
                """,
                error, metrics_json, job_id,
            )
            await conn.execute(
                """
                UPDATE ml_model_state
                SET status = CASE
                        WHEN active_model_key = desired_model_key
                        THEN 'ready' ELSE 'failed'
                    END,
                    last_error = $1,
                    updated_at = NOW()
                WHERE kind = $2 AND desired_model_key = $3
                """,
                error, job["kind"], job["model_key"],
            )
            if failed is None:
                raise ModelStoreError("failed reindex job did not return a row")
            return _job_row(failed), False, remaining

        if remaining:
            if job["force_reindex"]:
                error = (
                    f"forced re-index finished with {remaining} target "
                    "item(s) missing"
                )
                incomplete = await conn.fetchrow(
                    """
                    UPDATE ml_reindex_jobs
                    SET status = 'failed', error = $1,
                        metrics = COALESCE($2::jsonb, metrics),
                        finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
                    WHERE id = $3
                    RETURNING *
                    """,
                    error, metrics_json, job_id,
                )
                await conn.execute(
                    """
                    UPDATE ml_model_state
                    SET status = CASE
                            WHEN active_model_key = desired_model_key
                            THEN 'ready' ELSE 'failed'
                        END,
                        last_error = $1,
                        updated_at = NOW()
                    WHERE kind = $2 AND desired_model_key = $3
                    """,
                    error, job["kind"], job["model_key"],
                )
            else:
                incomplete = await conn.fetchrow(
                    """
                    UPDATE ml_reindex_jobs
                    SET cursor_media_id = 0,
                        total_items = total_items + $1,
                        metrics = COALESCE($2::jsonb, metrics),
                        error = NULL,
                        updated_at = NOW()
                    WHERE id = $3
                    RETURNING *
                    """,
                    remaining, metrics_json, job_id,
                )
            if incomplete is None:
                raise ModelStoreError("incomplete reindex job did not return a row")
            return _job_row(incomplete), False, remaining

        processed_items = int(job["processed_items"] or 0)
        failed_items = int(job["failed_items"] or 0)
        skipped_items = int(job["skipped_items"] or 0)
        total_items = int(job["total_items"] or 0)
        if not job["force_reindex"]:
            skipped_items = max(
                skipped_items,
                total_items - processed_items - failed_items,
                0,
            )
        total_items = max(
            total_items,
            processed_items + failed_items + skipped_items,
        )

        await conn.execute(
            """
            UPDATE ml_model_state
            SET active_model_key = $1, status = 'ready', last_error = NULL,
                updated_at = NOW()
            WHERE kind = $2
            """,
            job["model_key"], job["kind"],
        )
        if benchmark is not None:
            benchmark_payload = dict(benchmark)
            benchmark_payload["model_key"] = job["model_key"]
            benchmark_payload["reindex_job_id"] = job_id
            await record_model_benchmark_async(conn, **benchmark_payload)
        completed = await conn.fetchrow(
            """
            UPDATE ml_reindex_jobs
            SET status = 'completed', error = NULL,
                metrics = COALESCE($1::jsonb, metrics),
                total_items = $2, skipped_items = $3,
                finished_at = COALESCE(finished_at, NOW()), updated_at = NOW()
            WHERE id = $4
            RETURNING *
            """,
            metrics_json, total_items, skipped_items, job_id,
        )
        if completed is None:
            raise ModelStoreError("completed reindex job did not return a row")
        return _job_row(completed), True, 0


def _validated_vector(values: Iterable[float], expected_dimension: int) -> list[float]:
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding contains a non-numeric value") from exc
    if len(result) != expected_dimension:
        raise ValueError(
            f"embedding has {len(result)} dimensions; expected {expected_dimension}"
        )
    if not all(math.isfinite(value) for value in result):
        raise ValueError("embedding contains NaN or infinity")
    return result


def _vector_text(values: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in values) + "]"


def _scene_model(conn: Any, model_key: str) -> dict[str, Any]:
    model = get_model(conn, model_key)
    _assert_model_kind(model, model_key, "scene")
    return model  # type: ignore[return-value]


async def _scene_model_async(conn: Any, model_key: str) -> dict[str, Any]:
    model = await get_model_async(conn, model_key)
    _assert_model_kind(model, model_key, "scene")
    return model  # type: ignore[return-value]


def upsert_scene_embedding(
    conn: Any,
    *,
    media_id: int,
    model_key: str,
    embedding: Iterable[float],
    runtime_provider: str = "unknown",
    runtime_precision: str = "unknown",
) -> dict[str, Any]:
    model = _scene_model(conn, model_key)
    dimension = _positive_dimension(model["embedding_dimension"])
    vector = _validated_vector(embedding, dimension)  # type: ignore[arg-type]
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO scene_embeddings (
                media_id, model_key, embedding, dimension,
                runtime_provider, runtime_precision
            ) VALUES (%s, %s, (%s::text)::vector, %s, %s, %s)
            ON CONFLICT (media_id, model_key) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                dimension = EXCLUDED.dimension,
                runtime_provider = EXCLUDED.runtime_provider,
                runtime_precision = EXCLUDED.runtime_precision,
                updated_at = NOW()
            RETURNING media_id, model_key, dimension, runtime_provider,
                      runtime_precision, created_at, updated_at
            """,
            (
                int(media_id), model_key, _vector_text(vector), dimension,
                runtime_provider or "unknown", runtime_precision or "unknown",
            ),
        )
        row = cur.fetchone()
    if row is None:
        raise ModelStoreError("scene embedding upsert did not return a row")
    return dict(row)


async def upsert_scene_embedding_async(
    conn: Any,
    *,
    media_id: int,
    model_key: str,
    embedding: Iterable[float],
    runtime_provider: str = "unknown",
    runtime_precision: str = "unknown",
) -> dict[str, Any]:
    model = await _scene_model_async(conn, model_key)
    dimension = _positive_dimension(model["embedding_dimension"])
    vector = _validated_vector(embedding, dimension)  # type: ignore[arg-type]
    row = await conn.fetchrow(
        """
        INSERT INTO scene_embeddings (
            media_id, model_key, embedding, dimension,
            runtime_provider, runtime_precision
        ) VALUES ($1, $2, ($3::text)::vector, $4, $5, $6)
        ON CONFLICT (media_id, model_key) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            dimension = EXCLUDED.dimension,
            runtime_provider = EXCLUDED.runtime_provider,
            runtime_precision = EXCLUDED.runtime_precision,
            updated_at = NOW()
        RETURNING media_id, model_key, dimension, runtime_provider,
                  runtime_precision, created_at, updated_at
        """,
        int(media_id), model_key, _vector_text(vector), dimension,
        runtime_provider or "unknown", runtime_precision or "unknown",
    )
    if row is None:
        raise ModelStoreError("scene embedding upsert did not return a row")
    return dict(row)


def _parse_vector_text(value: str) -> list[float]:
    raw = value.strip()
    if not raw.startswith("[") or not raw.endswith("]"):
        raise ModelStoreError("database returned an invalid vector literal")
    inner = raw[1:-1]
    return [] if not inner else [float(part) for part in inner.split(",")]


def get_active_scene_embedding(conn: Any, media_id: int) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT se.media_id, se.model_key, se.embedding::text AS embedding,
                   se.dimension, se.runtime_provider, se.runtime_precision
            FROM ml_model_state state
            JOIN scene_embeddings se ON se.model_key = state.active_model_key
            WHERE state.kind = 'scene' AND se.media_id = %s
            """,
            (int(media_id),),
        )
        row = cur.fetchone()
    if row is None:
        return None
    result = dict(row)
    result["embedding"] = _parse_vector_text(result["embedding"])
    return result


async def get_active_scene_embedding_async(
    conn: Any, media_id: int
) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT se.media_id, se.model_key, se.embedding::text AS embedding,
               se.dimension, se.runtime_provider, se.runtime_precision
        FROM ml_model_state state
        JOIN scene_embeddings se ON se.model_key = state.active_model_key
        WHERE state.kind = 'scene' AND se.media_id = $1
        """,
        int(media_id),
    )
    if row is None:
        return None
    result = dict(row)
    result["embedding"] = _parse_vector_text(result["embedding"])
    return result


@dataclass(frozen=True)
class SceneKnnPlan:
    model_key: str
    dimension: int
    uses_hnsw: bool
    distance_metric: str


def scene_knn_plan(model: Mapping[str, Any]) -> SceneKnnPlan:
    """Describe whether a model's query matches a built-in partial HNSW index."""
    model_key = str(model["model_key"])
    dimension = _positive_dimension(model.get("embedding_dimension"))
    uses_hnsw = (
        model.get("index_kind") == "hnsw"
        and _BUILTIN_SCENE_INDEX_DIMS.get(model_key) == dimension
        and model.get("distance_metric", "cosine") == "cosine"
    )
    return SceneKnnPlan(
        model_key=model_key,
        dimension=dimension,  # type: ignore[arg-type]
        uses_hnsw=uses_hnsw,
        distance_metric=str(model.get("distance_metric") or "cosine"),
    )


def _distance_operator(metric: str) -> str:
    try:
        return {"cosine": "<=>", "inner_product": "<#>", "l2": "<->"}[metric]
    except KeyError as exc:
        raise ValueError(f"unsupported distance metric: {metric!r}") from exc


def query_scene_neighbors(
    conn: Any,
    embedding: Iterable[float],
    *,
    model_key: str | None = None,
    limit: int = 50,
    exclude_media_id: int | None = None,
) -> list[dict[str, Any]]:
    model = _scene_model(conn, model_key) if model_key else get_active_model(conn, "scene")
    if model is None:
        raise ModelStateConflict("no active scene model is configured")
    plan = scene_knn_plan(model)
    vector = _validated_vector(embedding, plan.dimension)
    limit = max(1, min(int(limit), 1000))
    op = _distance_operator(plan.distance_metric)
    distance = (
        f"(se.embedding::vector({plan.dimension})) {op} "
        f"((%s::text)::vector({plan.dimension}))"
    )
    vector_text = _vector_text(vector)
    params: list[Any] = [vector_text]
    if plan.uses_hnsw:
        predicate = f"se.model_key = {_BUILTIN_MODEL_SQL_LITERALS[plan.model_key]}"
    else:
        predicate = "se.model_key = %s"
        params.append(plan.model_key)
    if exclude_media_id is not None:
        predicate += " AND se.media_id <> %s"
        params.append(int(exclude_media_id))
    # The distance expression occurs once in SELECT and once in ORDER BY.
    params.append(vector_text)
    params.append(limit)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT se.media_id, {distance} AS distance
            FROM scene_embeddings se
            WHERE {predicate}
            ORDER BY {distance}
            LIMIT %s
            """,
            params,
        )
        rows = cur.fetchall()
    return [dict(row) for row in rows]


async def query_scene_neighbors_async(
    conn: Any,
    embedding: Iterable[float],
    *,
    model_key: str | None = None,
    limit: int = 50,
    exclude_media_id: int | None = None,
) -> list[dict[str, Any]]:
    model = (
        await _scene_model_async(conn, model_key)
        if model_key
        else await get_active_model_async(conn, "scene")
    )
    if model is None:
        raise ModelStateConflict("no active scene model is configured")
    plan = scene_knn_plan(model)
    vector_text = _vector_text(_validated_vector(embedding, plan.dimension))
    limit = max(1, min(int(limit), 1000))
    op = _distance_operator(plan.distance_metric)
    params: list[Any] = []

    def bind(value: Any) -> str:
        params.append(value)
        return f"${len(params)}"

    if plan.uses_hnsw:
        predicate = f"se.model_key = {_BUILTIN_MODEL_SQL_LITERALS[plan.model_key]}"
    else:
        predicate = f"se.model_key = {bind(plan.model_key)}"
    vector_param = bind(vector_text)
    distance = (
        f"(se.embedding::vector({plan.dimension})) {op} "
        f"(({vector_param}::text)::vector({plan.dimension}))"
    )
    if exclude_media_id is not None:
        predicate += f" AND se.media_id <> {bind(int(exclude_media_id))}"
    limit_param = bind(limit)
    rows = await conn.fetch(
        f"""
        SELECT se.media_id, {distance} AS distance
        FROM scene_embeddings se
        WHERE {predicate}
        ORDER BY {distance}
        LIMIT {limit_param}
        """,
        *params,
    )
    return [dict(row) for row in rows]


def upsert_processing_state(
    conn: Any,
    *,
    media_id: int,
    kind: str,
    model_key: str,
    status: str,
    item_count: int = 0,
    duration_ms: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if status not in {"completed", "failed"}:
        raise ValueError("processing status must be 'completed' or 'failed'")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO ml_processing_state (
                media_id, kind, model_key, status, item_count, duration_ms, error
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (media_id, kind, model_key) DO UPDATE SET
                status = EXCLUDED.status,
                item_count = EXCLUDED.item_count,
                duration_ms = EXCLUDED.duration_ms,
                error = EXCLUDED.error,
                updated_at = NOW()
            RETURNING *
            """,
            (media_id, kind, model_key, status, item_count, duration_ms, error),
        )
        row = cur.fetchone()
    if row is None:
        raise ModelStoreError("processing state upsert did not return a row")
    return dict(row)


async def upsert_processing_state_async(
    conn: Any,
    *,
    media_id: int,
    kind: str,
    model_key: str,
    status: str,
    item_count: int = 0,
    duration_ms: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if status not in {"completed", "failed"}:
        raise ValueError("processing status must be 'completed' or 'failed'")
    row = await conn.fetchrow(
        """
        INSERT INTO ml_processing_state (
            media_id, kind, model_key, status, item_count, duration_ms, error
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (media_id, kind, model_key) DO UPDATE SET
            status = EXCLUDED.status,
            item_count = EXCLUDED.item_count,
            duration_ms = EXCLUDED.duration_ms,
            error = EXCLUDED.error,
            updated_at = NOW()
        RETURNING *
        """,
        int(media_id), kind, model_key, status, int(item_count), duration_ms, error,
    )
    if row is None:
        raise ModelStoreError("processing state upsert did not return a row")
    return dict(row)


def record_model_benchmark(
    conn: Any,
    *,
    model_key: str,
    benchmark_type: str = "scene",
    sample_count: int = 0,
    batch_size: int | None = None,
    duration_ms: float = 0.0,
    items_per_second: float = 0.0,
    query_count: int = 0,
    query_latency_ms: Mapping[str, Any] | None = None,
    peak_vram_mb: float | None = None,
    runtime_provider: str = "unknown",
    runtime_precision: str = "unknown",
    metrics: Mapping[str, Any] | None = None,
    reindex_job_id: Any | None = None,
) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO ml_model_benchmarks (
                model_key, reindex_job_id, benchmark_type, sample_count,
                batch_size, duration_ms, items_per_second, query_count,
                query_latency_ms, peak_vram_mb, runtime_provider,
                runtime_precision, metrics
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s,
                %s::jsonb
            ) RETURNING *
            """,
            (
                model_key,
                (_as_uuid(reindex_job_id) if reindex_job_id is not None else None),
                benchmark_type, sample_count,
                batch_size, duration_ms, items_per_second, query_count,
                json.dumps(dict(query_latency_ms or {})), peak_vram_mb,
                runtime_provider, runtime_precision, json.dumps(dict(metrics or {})),
            ),
        )
        row = cur.fetchone()
    if row is None:
        raise ModelStoreError("benchmark insert did not return a row")
    return _benchmark_row(row)


async def record_model_benchmark_async(
    conn: Any,
    *,
    model_key: str,
    benchmark_type: str = "scene",
    sample_count: int = 0,
    batch_size: int | None = None,
    duration_ms: float = 0.0,
    items_per_second: float = 0.0,
    query_count: int = 0,
    query_latency_ms: Mapping[str, Any] | None = None,
    peak_vram_mb: float | None = None,
    runtime_provider: str = "unknown",
    runtime_precision: str = "unknown",
    metrics: Mapping[str, Any] | None = None,
    reindex_job_id: Any | None = None,
) -> dict[str, Any]:
    row = await conn.fetchrow(
        """
        INSERT INTO ml_model_benchmarks (
            model_key, reindex_job_id, benchmark_type, sample_count,
            batch_size, duration_ms, items_per_second, query_count,
            query_latency_ms, peak_vram_mb, runtime_provider,
            runtime_precision, metrics
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11, $12,
            $13::jsonb
        ) RETURNING *
        """,
        model_key, (_as_uuid(reindex_job_id) if reindex_job_id is not None else None),
        benchmark_type, int(sample_count), batch_size, float(duration_ms),
        float(items_per_second), int(query_count),
        json.dumps(dict(query_latency_ms or {})), peak_vram_mb, runtime_provider,
        runtime_precision, json.dumps(dict(metrics or {})),
    )
    if row is None:
        raise ModelStoreError("benchmark insert did not return a row")
    return _benchmark_row(row)


def list_model_benchmarks(
    conn: Any,
    *,
    model_key: str | None = None,
    benchmark_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    with conn.cursor(row_factory=dict_row) as cur:
        if model_key is None and benchmark_type is None:
            cur.execute(
                "SELECT * FROM ml_model_benchmarks ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
        elif benchmark_type is None:
            cur.execute(
                """
                SELECT * FROM ml_model_benchmarks
                WHERE model_key = %s ORDER BY created_at DESC LIMIT %s
                """,
                (model_key, limit),
            )
        elif model_key is None:
            cur.execute(
                """
                SELECT * FROM ml_model_benchmarks
                WHERE benchmark_type = %s ORDER BY created_at DESC LIMIT %s
                """,
                (benchmark_type, limit),
            )
        else:
            cur.execute(
                """
                SELECT * FROM ml_model_benchmarks
                WHERE model_key = %s AND benchmark_type = %s
                ORDER BY created_at DESC LIMIT %s
                """,
                (model_key, benchmark_type, limit),
            )
        return [_benchmark_row(row) for row in cur.fetchall()]


async def list_model_benchmarks_async(
    conn: Any,
    *,
    model_key: str | None = None,
    benchmark_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    if model_key is None and benchmark_type is None:
        rows = await conn.fetch(
            "SELECT * FROM ml_model_benchmarks ORDER BY created_at DESC LIMIT $1",
            limit,
        )
    elif benchmark_type is None:
        rows = await conn.fetch(
            """
            SELECT * FROM ml_model_benchmarks
            WHERE model_key = $1 ORDER BY created_at DESC LIMIT $2
            """,
            model_key, limit,
        )
    elif model_key is None:
        rows = await conn.fetch(
            """
            SELECT * FROM ml_model_benchmarks
            WHERE benchmark_type = $1 ORDER BY created_at DESC LIMIT $2
            """,
            benchmark_type, limit,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT * FROM ml_model_benchmarks
            WHERE model_key = $1 AND benchmark_type = $2
            ORDER BY created_at DESC LIMIT $3
            """,
            model_key, benchmark_type, limit,
        )
    return [_benchmark_row(row) for row in rows]


__all__ = [
    "DEFAULT_SCENE_MODEL_KEY", "OPENCLIP_SIGLIP_MODEL_KEY",
    "HF_SIGLIP2_MODEL_KEY", "DEFAULT_FACE_MODEL_KEY", "FACENET_MODEL_KEY",
    "LEGACY_FACE_MODEL_KEY", "ModelStoreError", "ModelIdentityConflict",
    "ModelNotFound", "ModelStateConflict", "SceneKnnPlan",
    "normalize_model_descriptor", "register_model", "register_model_async",
    "get_model", "get_model_async", "list_models", "list_models_async",
    "get_model_state", "get_model_state_async", "list_model_states",
    "list_model_states_async", "get_active_model",
    "get_active_model_async", "reconcile_model_state",
    "reconcile_model_state_async", "activate_model", "activate_model_async",
    "mark_model_state_failed", "mark_model_state_failed_async",
    "create_reindex_job", "create_reindex_job_async", "get_reindex_job",
    "get_reindex_job_async", "list_reindex_jobs", "list_reindex_jobs_async",
    "update_reindex_job", "update_reindex_job_async", "start_reindex_job",
    "start_reindex_job_async", "finish_reindex_job",
    "finish_reindex_job_async", "fail_reindex_job", "fail_reindex_job_async",
    "request_reindex_cancel", "request_reindex_cancel_async",
    "cancel_reindex_job", "cancel_reindex_job_async",
    "finalize_reindex_and_activate", "finalize_reindex_and_activate_async",
    "upsert_scene_embedding", "upsert_scene_embedding_async",
    "get_active_scene_embedding", "get_active_scene_embedding_async",
    "scene_knn_plan", "query_scene_neighbors", "query_scene_neighbors_async",
    "upsert_processing_state", "upsert_processing_state_async",
    "record_model_benchmark", "record_model_benchmark_async",
    "list_model_benchmarks", "list_model_benchmarks_async",
]
