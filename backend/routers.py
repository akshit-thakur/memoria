"""Memoria — FastAPI Routers.

In plain English: this is where the API's *feature* routes live (faces,
blurry, dedup, map, editor, search). Each section is its own
``APIRouter`` that the main FastAPI app (in ``main.py``) mounts under
``/api/v1``. Most handlers run SQL straight against Postgres via asyncpg;
the editor routes also touch the filesystem (writing edited photos into
``~/.memoria/edits/``).

The most subtle thing here is the duplicate-grouping rebuild in
``_refresh_duplicate_groups`` — it uses pgvector's HNSW index to find
near-duplicate edges in O(log n) per photo, then does union-find on those
edges in Python to compute groups. The rest of the file is mostly thin
SQL wrappers + a handful of Pillow image transformations.

Hosts the /api/v1/<module> routes served by the primary FastAPI app. All
routers (faces, blurry, dedup, map, editor) query PostgreSQL directly or
write non-destructive outputs to ~/.memoria/."""

import asyncio
import json
import logging
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import metrics
import processing_config
import throughput
from runtime_paths import EDITS_DIR, TRASH_DIR, media_is_read_only
from library_paths import validate_media_source

try:
    _HNSW_EF_SEARCH = int(os.getenv("MEMORIA_HNSW_EF_SEARCH", "80"))
except ValueError:
    _HNSW_EF_SEARCH = 80
_HNSW_EF_SEARCH = max(1, min(_HNSW_EF_SEARCH, 1000))


# ── Dedup (real Postgres-backed) ─────────────────────────────────────────────
dedup_router = APIRouter()
_dedup_logger = logging.getLogger("memoria.dedup")

_TRASH_DIR = TRASH_DIR

# Watermark: embedding count at the time of the last successful rebuild.
# -1 means "never built". On a clean process start the tables may already
# hold data from a previous run, so we compare against a real count before
# deciding to skip. Thread-safety: asyncio is single-threaded; no lock needed.
_dedup_embedding_watermark: int = -1
# Background rebuild state — set to True while a rebuild asyncio.Task is running.
_dedup_rebuilding: bool = False
_dedup_rebuild_task: "asyncio.Task | None" = None

# ORDER BY fragment that puts the most "original-looking" member first.
# Tier 0 — camera-named files:  IMG_*, DSC_*, DSCF_*, etc.
# Tier 1 — date-prefixed files: 20231201_* / 20231201T* (any 8-digit date)
# Tier 2 — files that carry EXIF metadata (camera model present)
# Tier 3 — everything else (generic, unnamed, "copy", WhatsApp, etc.)
# Within each tier: larger file wins (uncompressed original > re-save),
# then earlier creation date, then lowest id as tiebreaker.
_MEMBER_ORDER_SQL = """
  CASE
    WHEN m.filename ~* '^(IMG|DSC|DSCF|MVI|MOV|VID|PICT|RAW|CR2|NEF|DNG|RW2|ORF|ARW)[-_ ]?\\d' THEN 0
    WHEN m.filename ~* '^2\\d{7}[-_T]' THEN 1
    WHEN m.exif_data IS NOT NULL
         AND m.exif_data::text NOT IN ('{}', 'null', '') THEN 2
    ELSE 3
  END ASC,
  m.file_size DESC NULLS LAST,
  m.file_created_at ASC NULLS LAST,
  m.id ASC
"""


def _read_only_media_guard() -> JSONResponse | None:
    """Return the shared mutation error when source media is read-only."""
    if not media_is_read_only():
        return None
    return JSONResponse(
        status_code=409,
        content={
            "code": "media_read_only",
            "detail": "source media is read-only; this operation is disabled",
        },
    )


async def _refresh_duplicate_groups(conn) -> None:
    """Rebuild duplicate_groups + duplicate_members from SSCD dedup embeddings.

    Builds a near-duplicate edge list via an HNSW lateral join (pgvector
    cosine distance), then runs union-find to compute connected components.
    Each component with >= 2 members becomes one group.

    Threshold (cosine distance ceiling) is tunable via MEMORIA_DEDUP_THRESHOLD;
    default 0.10 covers near-identical shots with light edits (re-encodes,
    crops, watermarks, color tweaks). Per-member ``cos_distance`` is the
    nearest-neighbor distance within the group, surfaced for UI display.

    Rows whose ``dedup_embedding IS NULL`` (e.g. before SSCD backfill runs)
    are silently excluded — those will join groups once their embeddings
    are filled in and the endpoint is hit again.

    The rebuild streams the edge list and replaces the previous snapshot in
    one transaction, so memory use is proportional to the union-find state
    rather than the number of pgvector edges.
    """
    threshold = float(os.getenv("MEMORIA_DEDUP_THRESHOLD", "0.10"))
    knn = int(os.getenv("MEMORIA_DEDUP_KNN", "8"))

    # Snapshot the highest existing group ID. Everything with id <= this
    # value is from a previous rebuild and will be removed at the end.
    old_max_id: int | None = await conn.fetchval(
        "SELECT MAX(id) FROM duplicate_groups"
    )

    # Classic union-find ("disjoint set union") over a streamed edge list.
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        # Path-compressing find.
        # First pass: walk up to the root of the tree containing `x`.
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        # Second pass: re-point every node on the way up directly at the
        # root, so future find() calls are O(1) instead of O(depth).
        while parent.get(x, x) != root:
            nxt = parent.get(x, x)
            parent[x] = root
            x = nxt
        return root

    def union(a: int, b: int) -> None:
        # Merge by re-pointing one group's root at the other's root.
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    best_dist: dict[int, float] = {}
    edge_sql = """
        SELECT m1.id AS a,
               m2.id AS b,
               (m1.dedup_embedding <=> m2.dedup_embedding)::float AS d
        FROM media m1
        JOIN LATERAL (
            SELECT id, dedup_embedding FROM media
            WHERE dedup_embedding IS NOT NULL AND id <> m1.id
            ORDER BY dedup_embedding <=> m1.dedup_embedding
            LIMIT $2
        ) m2 ON TRUE
        WHERE m1.dedup_embedding IS NOT NULL
          AND (m1.dedup_embedding <=> m2.dedup_embedding) <= $1
          AND m1.id < m2.id
        """

    async with conn.transaction():
        async for r in conn.cursor(edge_sql, threshold, knn, prefetch=1000):
            a, b, distance = int(r["a"]), int(r["b"]), float(r["d"])
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            union(a, b)
            if best_dist.get(a, float("inf")) > distance:
                best_dist[a] = distance
            if best_dist.get(b, float("inf")) > distance:
                best_dist[b] = distance

        components: dict[int, list[int]] = {}
        for node in parent:
            components.setdefault(find(node), []).append(node)
        grouped_members = [
            sorted(members) for members in components.values() if len(members) >= 2
        ]

        if grouped_members:
            group_rows = await conn.fetch(
                """
                INSERT INTO duplicate_groups (phash)
                SELECT NULL FROM generate_series(1, $1)
                RETURNING id
                """,
                len(grouped_members),
            )
            member_args = (
                (group["id"], media_id, best_dist.get(media_id))
                for group, members in zip(group_rows, grouped_members)
                for media_id in members
            )
            await conn.executemany(
                "INSERT INTO duplicate_members (group_id, media_id, cos_distance) "
                "VALUES ($1, $2, $3)",
                member_args,
            )

        if old_max_id is not None:
            await conn.execute(
                "DELETE FROM duplicate_groups WHERE id <= $1", old_max_id
            )


async def _run_rebuild_background(pool, target_count: int) -> None:
    """Acquire a fresh connection and run a full dedup rebuild without blocking requests."""
    global _dedup_embedding_watermark, _dedup_rebuilding
    try:
        async with pool.acquire() as conn:
            with metrics.timer("db.dedup_rebuild"):
                await _refresh_duplicate_groups(conn)
        _dedup_embedding_watermark = target_count
        _dedup_logger.info(
            "dedup rebuild complete: %d embeddings → watermark updated", target_count
        )
    except Exception as exc:
        _dedup_logger.error("Background dedup rebuild failed: %s", exc)
    finally:
        _dedup_rebuilding = False


def _media_row_to_dict(r) -> dict:
    exif = r["exif_data"]
    if isinstance(exif, str):
        try:
            exif = json.loads(exif)
        except json.JSONDecodeError:
            exif = {}
    elif not isinstance(exif, dict):
        exif = {}
    out = {
        "id": r["id"],
        "filename": r["filename"],
        "folder": r["directory"],
        "thumbnailUrl": f"/api/v1/media/{r['id']}/thumbnail",
        "previewUrl": f"/api/v1/preview/{r['id']}",
        "phash": r["phash"] or "",
        "blur_score": float(r["blur_score"] or 0),
        "exif": exif,
        "indexed_at": r["indexed_at"].isoformat() if r["indexed_at"] else None,
        "file_size": r["file_size"] or 0,
        "created_at": r["file_created_at"].isoformat() if r["file_created_at"] else "",
        "source_device": exif.get("Model", "") if isinstance(exif, dict) else "",
    }
    if "cos_distance" in r.keys():
        cd = r["cos_distance"]
        out["cos_distance"] = float(cd) if cd is not None else None
    return out


@dedup_router.get("/api/v1/dedup/groups")
async def dedup_list_groups(request: Request, page: int = 1, page_size: int = 20):
    """Return near-dup groups.

    Rebuild is triggered in the background when the embedding count changes.
    The response is returned immediately from whatever groups are already in
    the DB, so the first ~200 groups appear as soon as they are inserted
    rather than waiting for the full rebuild to finish. Clients should poll
    while ``rebuilding`` is true (recommended interval: 3 s).
    """
    global _dedup_embedding_watermark, _dedup_rebuilding, _dedup_rebuild_task
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    offset = (page - 1) * page_size

    pool = request.app.state.db
    async with pool.acquire() as conn:
        current_count: int = await conn.fetchval(
            "SELECT COUNT(*) FROM media WHERE dedup_embedding IS NOT NULL"
        ) or 0

        if current_count != _dedup_embedding_watermark and not _dedup_rebuilding:
            _dedup_rebuilding = True
            _dedup_rebuild_task = asyncio.create_task(
                _run_rebuild_background(pool, current_count)
            )
            _dedup_logger.info(
                "dedup rebuild scheduled in background: %d embeddings (watermark was %d)",
                current_count, _dedup_embedding_watermark,
            )

        total_groups: int = await conn.fetchval(
            "SELECT COUNT(*) FROM duplicate_groups"
        ) or 0
        group_rows = await conn.fetch(
            "SELECT id, phash FROM duplicate_groups ORDER BY id LIMIT $1 OFFSET $2",
            page_size, offset,
        )
        group_ids = [r["id"] for r in group_rows]
        member_rows = await conn.fetch(
            f"""
            SELECT dm.group_id, dm.cos_distance,
                   m.id, m.filename, m.directory, m.phash, m.blur_score,
                   m.exif_data, m.indexed_at, m.file_size, m.file_created_at
            FROM duplicate_members dm
            JOIN media m ON m.id = dm.media_id
            WHERE dm.group_id = ANY($1::int[])
            ORDER BY dm.group_id, {_MEMBER_ORDER_SQL}
            """,
            group_ids,
        ) if group_ids else []

    members_by_group: dict[int, list] = {}
    for r in member_rows:
        members_by_group.setdefault(r["group_id"], []).append(_media_row_to_dict(r))

    groups = [
        {"id": r["id"], "phash": r["phash"], "members": members_by_group.get(r["id"], [])}
        for r in group_rows
        if members_by_group.get(r["id"])
    ]
    return {
        "groups": groups,
        "total": total_groups,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total_groups / page_size) if total_groups > 0 else 1,
        "rebuilding": _dedup_rebuilding,
    }


@dedup_router.get("/api/v1/dedup/groups/{group_id}")
async def dedup_get_group(group_id: int, request: Request):
    pool = request.app.state.db
    async with pool.acquire() as conn:
        g = await conn.fetchrow(
            "SELECT id, phash FROM duplicate_groups WHERE id = $1", group_id
        )
        if g is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        member_rows = await conn.fetch(
            f"""
            SELECT dm.cos_distance,
                   m.id, m.filename, m.directory, m.phash, m.blur_score,
                   m.exif_data, m.indexed_at, m.file_size, m.file_created_at
            FROM duplicate_members dm
            JOIN media m ON m.id = dm.media_id
            WHERE dm.group_id = $1
            ORDER BY {_MEMBER_ORDER_SQL}
            """,
            group_id,
        )
    return {
        "id": g["id"],
        "phash": g["phash"],
        "members": [_media_row_to_dict(r) for r in member_rows],
    }


@dedup_router.post("/api/v1/dedup/groups/{group_id}/delete-duplicates")
async def dedup_delete_duplicates(group_id: int, request: Request):
    """Move all members except the best-quality original to
    ``~/.memoria/trash/`` and cascade-delete their media rows.
    "Best-quality" is determined by _MEMBER_ORDER_SQL: camera-named files
    (IMG_*, DSC_*) beat date-prefixed names, which beat EXIF-carrying files,
    which beat everything else; ties broken by file size then creation date."""
    if response := _read_only_media_guard():
        return response
    pool = request.app.state.db
    _TRASH_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT m.id, m.file_path, m.filename, m.exif_data, m.file_size, m.file_created_at
            FROM duplicate_members dm
            JOIN media m ON m.id = dm.media_id
            WHERE dm.group_id = $1
            ORDER BY {_MEMBER_ORDER_SQL}
            """,
            group_id,
        )
        if not rows:
            return JSONResponse({"detail": "group not found"}, status_code=404)

        keep = rows[0]
        remove = rows[1:]
        deleted_ids: list[int] = []
        failed_ids: list[int] = []

        for r in remove:
            src = Path(r["file_path"])
            if src.exists():
                dest = _TRASH_DIR / f"{ts}_{r['id']}_{src.name}"
                try:
                    shutil.move(str(src), str(dest))
                except OSError as exc:
                    _dedup_logger.error("Trash move failed for %s: %s", src, exc)
                    failed_ids.append(r["id"])
                    # Preserve the database row when the source file could not
                    # be moved. Deleting only the row would make the operation
                    # look successful until the watcher indexed the same file
                    # again on its next pass.
                    continue
            deleted_ids.append(r["id"])

        if deleted_ids:
            await conn.execute(
                "DELETE FROM media WHERE id = ANY($1::bigint[])", deleted_ids
            )
        # Keep the group visible when one or more files could not be moved, so
        # the UI can retry instead of silently hiding unresolved duplicates.
        if not failed_ids:
            await conn.execute("DELETE FROM duplicate_groups WHERE id = $1", group_id)

    return {
        "status": "ok",
        "kept_id": keep["id"],
        "deleted_ids": deleted_ids,
        "failed_ids": failed_ids,
        "trash_dir": str(_TRASH_DIR),
    }


# ── Faces ────────────────────────────────────────────────────────────────────
faces_router = APIRouter()
_faces_logger = logging.getLogger("memoria.faces")


@faces_router.get("/api/v1/faces/clusters")
async def faces_list_clusters(request: Request, limit: int = 100, offset: int = 0):
    limit = min(limit, 500)
    pool = request.app.state.db
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT fc.id, fc.label, COUNT(fe.id)::integer AS face_count,
                   AVG(fe.confidence)::double precision AS confidence_score,
                   fc.representative_media_id, fc.created_at
            FROM face_clusters fc
            JOIN face_embeddings fe ON fe.cluster_id = fc.id
            JOIN ml_model_state state
              ON state.kind = 'face-embedding'
             AND state.active_model_key = fe.embedding_model_key
            WHERE fc.is_hidden = FALSE
            GROUP BY fc.id, fc.label, fc.representative_media_id, fc.created_at
            ORDER BY COUNT(fe.id) DESC, fc.id
            LIMIT $1 OFFSET $2
            """,
            limit, offset,
        )
        total = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT fc.id)
            FROM face_clusters fc
            JOIN face_embeddings fe ON fe.cluster_id = fc.id
            JOIN ml_model_state state
              ON state.kind = 'face-embedding'
             AND state.active_model_key = fe.embedding_model_key
            WHERE fc.is_hidden = FALSE
            """
        ) or 0
    clusters = [
        {
            "id": r["id"],
            "label": r["label"],
            "face_count": r["face_count"],
            "confidence_score": r["confidence_score"],
            "representative_media_id": r["representative_media_id"],
            "sample_thumbnail": f"/api/v1/media/{r['representative_media_id']}/thumbnail"
                if r["representative_media_id"] else None,
            "count": r["face_count"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]
    return {"clusters": clusters, "total": total, "limit": limit, "offset": offset}


@faces_router.get("/api/v1/faces/clusters/{cluster_id}")
async def faces_get_cluster(cluster_id: int, request: Request):
    pool = request.app.state.db
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT fc.id, fc.label, COUNT(fe.id)::integer AS face_count,
                   AVG(fe.confidence)::double precision AS confidence_score,
                   fc.representative_media_id, fc.created_at
            FROM face_clusters fc
            JOIN face_embeddings fe ON fe.cluster_id = fc.id
            JOIN ml_model_state state
              ON state.kind = 'face-embedding'
             AND state.active_model_key = fe.embedding_model_key
            WHERE fc.id = $1 AND fc.is_hidden = FALSE
            GROUP BY fc.id, fc.label, fc.representative_media_id, fc.created_at
            """,
            cluster_id,
        )
        if row is None:
            return JSONResponse(status_code=404, content={"detail": "not found"})

        members = await conn.fetch(
            "SELECT fe.id, fe.media_id, fe.bbox, fe.confidence, "
            "m.file_path "
            "FROM face_embeddings fe "
            "JOIN media m ON m.id = fe.media_id "
            "JOIN ml_model_state state "
            "  ON state.kind = 'face-embedding' "
            " AND state.active_model_key = fe.embedding_model_key "
            "WHERE fe.cluster_id = $1 "
            "ORDER BY fe.confidence DESC",
            cluster_id,
        )

    cluster = {
        "id": row["id"],
        "label": row["label"],
        "face_count": row["face_count"],
        "confidence_score": row["confidence_score"],
        "representative_media_id": row["representative_media_id"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "members": [
            {
                "face_id": m["id"],
                "media_id": m["media_id"],
                "bbox": json.loads(m["bbox"]) if isinstance(m["bbox"], str) else m["bbox"],
                "confidence": m["confidence"],
                "file_path": m["file_path"],
            }
            for m in members
        ],
    }
    return cluster


@faces_router.get("/api/v1/faces/photo/{photo_id}")
async def faces_for_photo(photo_id: int, request: Request):
    pool = request.app.state.db
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT fe.id, fe.bbox, fe.cluster_id, fe.confidence, "
            "fe.detection_model, fc.label "
            "FROM face_embeddings fe "
            "LEFT JOIN face_clusters fc ON fc.id = fe.cluster_id "
            "JOIN ml_model_state state "
            "  ON state.kind = 'face-embedding' "
            " AND state.active_model_key = fe.embedding_model_key "
            "WHERE fe.media_id = $1",
            photo_id,
        )
    faces = [
        {
            "face_id": r["id"],
            "bbox": json.loads(r["bbox"]) if isinstance(r["bbox"], str) else r["bbox"],
            "cluster_id": r["cluster_id"],
            "confidence": r["confidence"],
            "detection_model": r["detection_model"],
            "cluster_label": r["label"],
        }
        for r in rows
    ]
    return {"faces": faces}


@faces_router.patch("/api/v1/faces/clusters/{cluster_id}/label")
async def faces_update_label(cluster_id: int, request: Request):
    body = await request.json()
    label = body.get("label")
    if label is None:
        return JSONResponse(status_code=400, content={"detail": "missing 'label' field"})
    pool = request.app.state.db
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Check the target cluster exists and is not hidden.
            exists = await conn.fetchval(
                "SELECT id FROM face_clusters WHERE id = $1 AND is_hidden = FALSE",
                cluster_id,
            )
            if not exists:
                return JSONResponse(status_code=404, content={"detail": "not found"})

            # Set the label.
            await conn.execute(
                "UPDATE face_clusters SET label = $1 WHERE id = $2",
                label, cluster_id,
            )

            # Auto-merge: find visible clusters with the same label (case-insensitive).
            siblings = await conn.fetch(
                "SELECT id FROM face_clusters "
                "WHERE LOWER(label) = LOWER($1) AND id != $2 AND is_hidden = FALSE",
                label, cluster_id,
            )
            merged_total = 0
            for row in siblings:
                src = row["id"]
                moved = await conn.fetchval(
                    "SELECT COUNT(*) FROM face_embeddings WHERE cluster_id = $1", src
                )
                await conn.execute(
                    "UPDATE face_embeddings SET cluster_id = $1 WHERE cluster_id = $2",
                    cluster_id, src,
                )
                await conn.execute(
                    "INSERT INTO face_cluster_merges (source_cluster, target_cluster, merged_count) "
                    "VALUES ($1, $2, $3)",
                    src, cluster_id, moved,
                )
                await conn.execute("DELETE FROM face_clusters WHERE id = $1", src)
                merged_total += moved

            # Update the target cluster's face_count.
            new_count = await conn.fetchval(
                "SELECT COUNT(*) FROM face_embeddings WHERE cluster_id = $1", cluster_id
            )
            await conn.execute(
                "UPDATE face_clusters SET face_count = $1 WHERE id = $2",
                new_count, cluster_id,
            )
    return {
        "status": "ok",
        "cluster_id": cluster_id,
        "label": label,
        "merged_clusters": len(siblings),
        "merged_faces": merged_total,
    }


@faces_router.post("/api/v1/faces/clusters/{cluster_id}/merge")
async def faces_merge(cluster_id: int, request: Request):
    body = await request.json()
    source_id = body.get("source_cluster_id")
    if source_id is None:
        return JSONResponse(status_code=400, content={"detail": "missing 'source_cluster_id'"})
    if int(source_id) == cluster_id:
        return JSONResponse(status_code=400, content={"detail": "cannot merge cluster into itself"})

    pool = request.app.state.db
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Refuse merge if either cluster is hidden.
            hidden_check = await conn.fetchval(
                "SELECT COUNT(*) FROM face_clusters "
                "WHERE id = ANY($1) AND is_hidden = TRUE",
                [cluster_id, int(source_id)],
            )
            if hidden_check:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "cannot merge hidden clusters"},
                )
            moved = await conn.fetchval(
                "SELECT COUNT(*) FROM face_embeddings WHERE cluster_id = $1",
                source_id,
            )
            await conn.execute(
                "UPDATE face_embeddings SET cluster_id = $1 WHERE cluster_id = $2",
                cluster_id, source_id,
            )
            new_count = await conn.fetchval(
                "SELECT COUNT(*) FROM face_embeddings WHERE cluster_id = $1",
                cluster_id,
            )
            await conn.execute(
                "UPDATE face_clusters SET face_count = $1 WHERE id = $2",
                new_count, cluster_id,
            )
            await conn.execute(
                "INSERT INTO face_cluster_merges (source_cluster, target_cluster, merged_count) "
                "VALUES ($1, $2, $3)",
                source_id, cluster_id, moved,
            )
            await conn.execute(
                "DELETE FROM face_clusters WHERE id = $1", source_id,
            )
    return {"status": "ok", "target_cluster_id": cluster_id, "merged_count": moved}


@faces_router.delete("/api/v1/faces/clusters/{cluster_id}")
async def faces_hide_cluster(cluster_id: int, request: Request):
    """Soft-delete a face cluster. Hidden clusters are invisible in the UI
    and new faces of the same person are silently absorbed into the hidden
    cluster by the clustering worker, preventing respawn."""
    pool = request.app.state.db
    async with pool.acquire() as conn:
        updated = await conn.fetchval(
            "UPDATE face_clusters SET is_hidden = TRUE, hidden_at = NOW() "
            "WHERE id = $1 AND is_hidden = FALSE RETURNING id",
            cluster_id,
        )
        if updated is None:
            return JSONResponse(status_code=404, content={"detail": "not found or already hidden"})
    return {"ok": True, "cluster_id": cluster_id}


@faces_router.post("/api/v1/faces/index")
async def faces_trigger_indexing():
    """Trigger face detection on unscanned media, then cluster."""
    from workers.tasks import reprocess_faces, cluster_faces
    reprocess_faces()
    cluster_faces()
    return {"status": "queued", "tasks": ["reprocess_faces", "cluster_faces"]}


@faces_router.delete("/api/v1/faces/clear")
async def faces_clear_all(request: Request):
    """Delete face data and completion markers so a later scan can rebuild it."""
    pool = request.app.state.db
    async with pool.acquire() as conn:
        async with conn.transaction():
            emb = await conn.fetchval("SELECT COUNT(*) FROM face_embeddings")
            cls = await conn.fetchval("SELECT COUNT(*) FROM face_clusters")
            completed = await conn.fetchval(
                "SELECT COUNT(*) FROM ml_processing_state "
                "WHERE kind = 'face-embedding'"
            )
            await conn.execute("DELETE FROM face_cluster_merges")
            await conn.execute("DELETE FROM face_embeddings")
            await conn.execute("DELETE FROM face_clusters")
            # Successful zero-face passes also leave completion rows. Keeping
            # those rows would make POST /faces/index believe the namespace is
            # already complete and turn the UI's promised re-scan into a no-op.
            await conn.execute(
                "DELETE FROM ml_processing_state WHERE kind = 'face-embedding'"
            )
    _faces_logger.info(
        "Cleared all face data: %d embeddings, %d clusters, %d completion records",
        emb,
        cls,
        completed,
    )
    return {
        "status": "ok",
        "deleted_embeddings": emb,
        "deleted_clusters": cls,
        "deleted_completion_records": completed,
    }


@faces_router.get("/api/v1/faces/{face_id}/similar")
async def faces_similar(
    face_id: int,
    request: Request,
    limit: int = 24,
    threshold: float = 0.6,
):
    """Return faces nearest to ``face_id`` by cosine distance.

    Uses the active model's partial HNSW index (or an exact scan for a custom
    model) and never compares embeddings from different model namespaces.
    Useful when DBSCAN left the source face as noise (cluster_id IS NULL) or
    split it across near-duplicate clusters.

    ``threshold`` is the maximum cosine distance (0..2). 0.6 is permissive
    enough to surface likely matches DBSCAN missed.
    """
    limit = min(max(limit, 1), 200)
    threshold = threshold if math.isfinite(threshold) else 0.6
    threshold = max(0.0, min(threshold, 2.0))
    import model_store

    pool = request.app.state.db
    async with pool.acquire() as conn:
        src = await conn.fetchrow(
            """
            SELECT fe.embedding, fe.embedding_model_key
            FROM face_embeddings fe
            JOIN ml_model_state state
              ON state.kind = 'face-embedding'
             AND state.active_model_key = fe.embedding_model_key
            WHERE fe.id = $1
            """,
            face_id,
        )
        if src is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "face not found in the active model namespace"},
            )

        # Literal predicates are required for PostgreSQL to match the three
        # built-in partial HNSW indexes. Only canonical constants can become
        # SQL syntax; a future custom face model remains a bound parameter and
        # therefore uses a correct exact scan until its own index is declared.
        indexed_literals = {
            model_store.DEFAULT_FACE_MODEL_KEY: (
                "'face/insightface-buffalo-l-arcface'"
            ),
            model_store.FACENET_MODEL_KEY: "'face/facenet-vggface2'",
            model_store.LEGACY_FACE_MODEL_KEY: "'face/legacy-unknown-512'",
        }
        model_key = src["embedding_model_key"]
        literal = indexed_literals.get(model_key)
        if literal is not None:
            model_predicate = f"fe.embedding_model_key = {literal}"
            query_args = (src["embedding"], face_id, threshold, limit)
        else:
            model_predicate = "fe.embedding_model_key = $5"
            query_args = (
                src["embedding"], face_id, threshold, limit, model_key,
            )
        async with conn.transaction():
            # SET LOCAL takes effect for the current transaction only — bumps
            # HNSW recall for this single ANN query without polluting the
            # pooled connection's session state.
            await conn.execute(
                f"SET LOCAL hnsw.ef_search = {_HNSW_EF_SEARCH}"
            )
            with metrics.timer("db.vector_knn", route="/api/v1/faces/{face_id}/similar"):
                rows = await conn.fetch(
                    f"""
                    SELECT fe.id, fe.media_id, fe.bbox, fe.cluster_id, fe.confidence,
                           m.filename, m.directory, fc.label,
                           fe.embedding <=> $1 AS distance
                    FROM face_embeddings fe
                    JOIN media m ON m.id = fe.media_id
                    LEFT JOIN face_clusters fc ON fc.id = fe.cluster_id
                    WHERE fe.id <> $2
                      AND {model_predicate}
                      AND fe.embedding <=> $1 <= $3
                    ORDER BY fe.embedding <=> $1
                    LIMIT $4
                    """,
                    *query_args,
                )
            metrics.record("db.vector_knn.hits", len(rows), route="/api/v1/faces/{face_id}/similar")
    faces = [
        {
            "face_id": r["id"],
            "media_id": r["media_id"],
            "bbox": json.loads(r["bbox"]) if isinstance(r["bbox"], str) else r["bbox"],
            "cluster_id": r["cluster_id"],
            "confidence": r["confidence"],
            "cluster_label": r["label"],
            "filename": r["filename"],
            "folder": r["directory"],
            "distance": float(r["distance"]),
        }
        for r in rows
    ]
    return {"faces": faces}


# ── Blurry ───────────────────────────────────────────────────────────────────
blurry_router = APIRouter()

_BLUR_THRESHOLD = float(__import__("os").environ.get("MEMORIA_BLUR_THRESHOLD", "0.75"))


@blurry_router.get("/api/v1/blurry/photos")
async def blurry_list(request: Request, threshold: float = _BLUR_THRESHOLD, limit: int = 50, offset: int = 0):
    limit = min(limit, 200)
    pool = request.app.state.db
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, file_path, filename, blur_score, mime_type, file_size
            FROM media
            WHERE blur_score > $1
            ORDER BY blur_score DESC
            LIMIT $2 OFFSET $3
            """,
            threshold, limit, offset,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM media WHERE blur_score > $1",
            threshold,
        )
    items = [
        {
            "id": r["id"],
            "file_path": r["file_path"],
            "filename": r["filename"],
            "blur_score": round(r["blur_score"], 4),
            "mime_type": r["mime_type"],
            "file_size": r["file_size"],
        }
        for r in rows
    ]
    return {"items": items, "total": total, "threshold": threshold}


@blurry_router.post("/api/v1/blurry/scan")
async def blurry_scan():
    from workers.tasks import rescore_blur
    result = rescore_blur()
    return {"status": "queued", "task_id": result.id}


_BLURRY_PILLOW_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"})


@blurry_router.post("/api/v1/blurry/photos/{photo_id}/sharpen")
async def blurry_sharpen(photo_id: int, request: Request):
    """Run Pillow UnsharpMask on the original, save to ~/.memoria/edits/,
    and record the edit. Does not modify the original file."""
    pool = request.app.state.db
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, file_path FROM media WHERE id = $1", photo_id
        )
        if row is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        file_path = row["file_path"]
        try:
            file_path = validate_media_source(file_path)
        except ValueError:
            return JSONResponse({"detail": "file not found on disk"}, status_code=404)
        if Path(file_path).suffix.lower() not in _BLURRY_PILLOW_EXTS:
            return JSONResponse({"detail": "unsupported format"}, status_code=400)
        if not Path(file_path).exists():
            return JSONResponse({"detail": "file not found on disk"}, status_code=404)

        try:
            from PIL import Image, ImageFilter, ImageOps
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            img = Image.open(file_path)
            img = ImageOps.exif_transpose(img).convert("RGB")
            sharpened = img.filter(
                ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
            )
            out_dir = _safe_output_dir(_EDITS_DIR, photo_id)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            output = out_dir / f"sharpened_{ts}.jpg"
            sharpened.save(output, "JPEG", quality=88)
            img.close()
        except Exception as exc:
            _editor_logger.error("blurry_sharpen failed for %s: %s", file_path, exc)
            return JSONResponse({"detail": "sharpen failed"}, status_code=500)

        await conn.execute(
            "INSERT INTO media_edits (media_id, op, params, output_path) "
            "VALUES ($1, 'filter', $2::jsonb, $3)",
            photo_id, json.dumps({"filter_name": "sharpen", "source": "blurry"}),
            str(output),
        )

    return {
        "status": "ok", "photo_id": photo_id, "action": "sharpen",
        "output_path": str(output),
    }


@blurry_router.delete("/api/v1/blurry/photos/{photo_id}")
async def blurry_delete(photo_id: int, request: Request):
    if response := _read_only_media_guard():
        return response
    """Move the file to ~/.memoria/trash/ and cascade-delete its media row."""
    pool = request.app.state.db
    _TRASH_DIR.mkdir(parents=True, exist_ok=True)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, file_path FROM media WHERE id = $1", photo_id
        )
        if row is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        file_path = row["file_path"]
        try:
            file_path = validate_media_source(file_path)
        except ValueError:
            return JSONResponse({"detail": "file not found on disk"}, status_code=404)
        src = Path(file_path)
        trashed: str | None = None
        if src.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            dest = _TRASH_DIR / f"{ts}_{photo_id}_{src.name}"
            try:
                shutil.move(str(src), str(dest))
                trashed = str(dest)
            except OSError as exc:
                _dedup_logger.error("Trash move failed for %s: %s", src, exc)
                return JSONResponse(
                    {"detail": f"trash move failed: {exc}"}, status_code=500
                )

        await conn.execute("DELETE FROM media WHERE id = $1", photo_id)

    return {
        "status": "ok", "photo_id": photo_id, "action": "deleted",
        "trashed_to": trashed,
    }


# ── Map (real Postgres-backed, reads GPS from exif_data JSONB) ───────────────
map_router = APIRouter()


@map_router.get("/api/v1/map/locations")
async def map_list_locations(request: Request, limit: int = 2000, offset: int = 0):
    """Return photos whose EXIF contained GPSInfo, slim payload (id, lat, lng only).

    MapLibre's GeoJSON cluster source handles tens-of-thousands of points; we
    cap at 2000 per page and let the frontend paginate or stream as needed.
    """
    limit = min(limit, 5000)
    pool = request.app.state.db
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id,
                   (exif_data->>'_gps_lat')::float AS lat,
                   (exif_data->>'_gps_lng')::float AS lng
            FROM media
            WHERE exif_data ? '_gps_lat' AND exif_data ? '_gps_lng'
            ORDER BY indexed_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit, offset,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM media WHERE exif_data ? '_gps_lat' AND exif_data ? '_gps_lng'"
        ) or 0
    points = [
        {
            "media_id": r["id"],
            "lat": r["lat"],
            "lng": r["lng"],
            "thumbnail": f"/api/v1/media/{r['id']}/thumbnail",
        }
        for r in rows
    ]
    return {"points": points, "total": total, "limit": limit, "offset": offset}


@map_router.get("/api/v1/map/locations/{media_id}")
async def map_get_location(media_id: int, request: Request):
    pool = request.app.state.db
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id,
                   (exif_data->>'_gps_lat')::float AS lat,
                   (exif_data->>'_gps_lng')::float AS lng,
                   filename
            FROM media
            WHERE id = $1
              AND exif_data ? '_gps_lat' AND exif_data ? '_gps_lng'
            """,
            media_id,
        )
    if row is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    return {
        "media_id": row["id"],
        "lat": row["lat"],
        "lng": row["lng"],
        "thumbnail": f"/api/v1/media/{row['id']}/thumbnail",
        "label": row["filename"],
    }


# ── Editor (non-destructive; writes to ~/.memoria/edits/) ────────────────────
editor_router = APIRouter()
_editor_logger = logging.getLogger("memoria.editor")

_EDITS_DIR = EDITS_DIR
_EDIT_PILLOW_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"})


def _safe_output_dir(root: Path, media_id: int) -> Path:
    root = root.expanduser()
    if root.exists() and root.is_symlink():
        raise ValueError("output directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    out = root / str(media_id)
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve().parent != root.resolve():
        raise ValueError("invalid output destination")
    return out


def _editor_load_image(file_path: str):
    """Open an image with Pillow, registering the HEIC opener on demand."""
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    img = Image.open(file_path)
    # Normalize EXIF orientation so crop/rotate params match what the user sees.
    img = ImageOps.exif_transpose(img)
    return img


async def _resolve_media(conn, photo_id: int):
    row = await conn.fetchrow(
        "SELECT id, file_path FROM media WHERE id = $1", photo_id
    )
    if row is not None:
        try:
            path = validate_media_source(row["file_path"])
        except (ValueError, TypeError):
            return None
        row = {"id": row["id"], "file_path": path}
    return row


def _save_edit_output(media_id: int, op: str, suffix: str) -> Path:
    """Return a unique output path inside ~/.memoria/edits/{media_id}/."""
    out_dir = _safe_output_dir(_EDITS_DIR, media_id)
    # Refuse symlinked output roots and keep every generated file below the
    # configured edits directory.
    root = _EDITS_DIR.expanduser().resolve()
    if _EDITS_DIR.exists() and _EDITS_DIR.is_symlink():
        raise ValueError("edits directory must not be a symlink")
    # directory is created and validated by _safe_output_dir
    if out_dir.resolve().parent != root:
        raise ValueError("invalid edits destination")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return out_dir / f"{op}_{ts}{suffix}"


async def _record_edit(conn, media_id: int, op: str, params: dict, output_path: Path):
    await conn.execute(
        "INSERT INTO media_edits (media_id, op, params, output_path) "
        "VALUES ($1, $2, $3::jsonb, $4)",
        media_id, op, json.dumps(params), str(output_path),
    )


@editor_router.post("/api/v1/editor/crop")
async def editor_crop(request: Request):
    """Crop a region of the original image and save to ~/.memoria/edits/.

    Body: {photo_id: int, x?: int, y?: int, w?: int, h?: int}
    Missing params → center 80% crop.
    """
    body = await request.json()
    photo_id = body.get("photo_id")
    if not isinstance(photo_id, int):
        return JSONResponse({"detail": "photo_id required"}, status_code=400)

    pool = request.app.state.db
    async with pool.acquire() as conn:
        row = await _resolve_media(conn, photo_id)
        if row is None:
            return JSONResponse({"detail": "photo not found"}, status_code=404)
        file_path = row["file_path"]
        if Path(file_path).suffix.lower() not in _EDIT_PILLOW_EXTS:
            return JSONResponse({"detail": "unsupported format"}, status_code=400)

        try:
            img = _editor_load_image(file_path)
        except Exception as exc:
            _editor_logger.error("editor_crop load failed for %s: %s", file_path, exc)
            return JSONResponse({"detail": "image load failed"}, status_code=500)

        iw, ih = img.size
        x = max(0, int(body.get("x", iw // 10)))
        y = max(0, int(body.get("y", ih // 10)))
        w = int(body.get("w", iw - 2 * x))
        h = int(body.get("h", ih - 2 * y))
        right = min(iw, x + w)
        bottom = min(ih, y + h)
        if right <= x or bottom <= y:
            return JSONResponse({"detail": "invalid crop bounds"}, status_code=400)

        output = _save_edit_output(photo_id, "crop", ".jpg")
        params = {"x": x, "y": y, "w": right - x, "h": bottom - y}
        try:
            cropped = img.crop((x, y, right, bottom)).convert("RGB")
            cropped.save(output, "JPEG", quality=88)
        except Exception as exc:
            _editor_logger.error("editor_crop save failed for %s: %s", file_path, exc)
            return JSONResponse({"detail": "crop failed"}, status_code=500)
        finally:
            img.close()

        await _record_edit(conn, photo_id, "crop", params, output)

    return {"status": "ok", "op": "crop", "photo_id": photo_id,
            "output_path": str(output), "params": params}


@editor_router.post("/api/v1/editor/rotate")
async def editor_rotate(request: Request):
    """Rotate the image by angle degrees (default 90 CW) and save to edits/."""
    body = await request.json()
    photo_id = body.get("photo_id")
    if not isinstance(photo_id, int):
        return JSONResponse({"detail": "photo_id required"}, status_code=400)
    angle = int(body.get("angle", 90))
    if angle not in (90, 180, 270):
        return JSONResponse({"detail": "angle must be 90, 180, or 270"}, status_code=400)

    pool = request.app.state.db
    async with pool.acquire() as conn:
        row = await _resolve_media(conn, photo_id)
        if row is None:
            return JSONResponse({"detail": "photo not found"}, status_code=404)
        file_path = row["file_path"]
        if Path(file_path).suffix.lower() not in _EDIT_PILLOW_EXTS:
            return JSONResponse({"detail": "unsupported format"}, status_code=400)

        try:
            img = _editor_load_image(file_path)
        except Exception as exc:
            _editor_logger.error("editor_rotate load failed for %s: %s", file_path, exc)
            return JSONResponse({"detail": "image load failed"}, status_code=500)

        output = _save_edit_output(photo_id, f"rotate_{angle}", ".jpg")
        params = {"angle": angle}
        try:
            # Pillow rotates counter-clockwise; negate for "90 CW" intuition.
            rotated = img.rotate(-angle, expand=True).convert("RGB")
            rotated.save(output, "JPEG", quality=88)
        except Exception as exc:
            _editor_logger.error("editor_rotate save failed for %s: %s", file_path, exc)
            return JSONResponse({"detail": "rotate failed"}, status_code=500)
        finally:
            img.close()

        await _record_edit(conn, photo_id, "rotate", params, output)

    return {"status": "ok", "op": "rotate", "photo_id": photo_id,
            "output_path": str(output), "params": params}


_SUPPORTED_FILTERS = frozenset({"grayscale", "invert", "sepia", "blur", "sharpen"})


def _apply_filter(img, name: str):
    from PIL import ImageFilter, ImageOps
    rgb = img.convert("RGB")
    if name == "grayscale":
        return ImageOps.grayscale(rgb).convert("RGB")
    if name == "invert":
        return ImageOps.invert(rgb)
    if name == "blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=3))
    if name == "sharpen":
        return rgb.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    if name == "sepia":
        # Classic sepia: grayscale → tinted LUT
        gray = ImageOps.grayscale(rgb)
        sepia = ImageOps.colorize(gray, black="#3B2412", white="#FFE9C4")
        return sepia.convert("RGB")
    return rgb


@editor_router.post("/api/v1/editor/filter")
async def editor_filter(request: Request):
    """Apply a named filter to the image and save to edits/.

    Body: {photo_id: int, filter_name?: str}
    Supported: grayscale (default), invert, sepia, blur, sharpen.
    """
    body = await request.json()
    photo_id = body.get("photo_id")
    if not isinstance(photo_id, int):
        return JSONResponse({"detail": "photo_id required"}, status_code=400)
    filter_name = str(body.get("filter_name", "grayscale"))
    if filter_name not in _SUPPORTED_FILTERS:
        return JSONResponse(
            {"detail": f"filter must be one of {sorted(_SUPPORTED_FILTERS)}"},
            status_code=400,
        )

    pool = request.app.state.db
    async with pool.acquire() as conn:
        row = await _resolve_media(conn, photo_id)
        if row is None:
            return JSONResponse({"detail": "photo not found"}, status_code=404)
        file_path = row["file_path"]
        if Path(file_path).suffix.lower() not in _EDIT_PILLOW_EXTS:
            return JSONResponse({"detail": "unsupported format"}, status_code=400)

        try:
            img = _editor_load_image(file_path)
        except Exception as exc:
            _editor_logger.error("editor_filter load failed for %s: %s", file_path, exc)
            return JSONResponse({"detail": "image load failed"}, status_code=500)

        output = _save_edit_output(photo_id, f"filter_{filter_name}", ".jpg")
        params = {"filter_name": filter_name}
        try:
            out_img = _apply_filter(img, filter_name)
            out_img.save(output, "JPEG", quality=88)
        except Exception as exc:
            _editor_logger.error("editor_filter save failed for %s: %s", file_path, exc)
            return JSONResponse({"detail": "filter failed"}, status_code=500)
        finally:
            img.close()

        await _record_edit(conn, photo_id, "filter", params, output)

    return {"status": "ok", "op": "filter", "photo_id": photo_id,
            "output_path": str(output), "params": params}


# ── Search (CLIP semantic + image-similar, Phase D) ──────────────────────────
search_router = APIRouter()


def _search_row_to_dict(r, distance: float | None = None) -> dict:
    """Like _media_row_to_dict but for search rows (carries a `distance` column)."""
    exif = r["exif_data"]
    if isinstance(exif, str):
        try:
            exif = json.loads(exif)
        except json.JSONDecodeError:
            exif = {}
    elif not isinstance(exif, dict):
        exif = {}
    return {
        "id": r["id"],
        "filename": r["filename"],
        "folder": r["directory"],
        "thumbnailUrl": f"/api/v1/media/{r['id']}/thumbnail",
        "previewUrl": f"/api/v1/preview/{r['id']}",
        "phash": r["phash"] or "",
        "blur_score": float(r["blur_score"] or 0),
        "exif": exif,
        "indexed_at": r["indexed_at"].isoformat() if r["indexed_at"] else None,
        "distance": float(r["distance"] if distance is None else distance),
    }


async def _hydrate_scene_neighbors(conn, neighbors: list[dict]) -> list[dict]:
    """Join model-store kNN results back to the stable photo API shape.

    ``model_store.query_scene_neighbors_async`` intentionally returns only
    ``media_id`` and ``distance`` so it can stay focused on dimension-safe
    vector dispatch.  Hydrate those IDs in one query, then restore the kNN
    ordering here rather than relying on PostgreSQL's ``ANY`` row order.
    """
    if not neighbors:
        return []
    media_ids = [int(item["media_id"]) for item in neighbors]
    rows = await conn.fetch(
        """
        SELECT id, filename, directory, phash, blur_score, exif_data, indexed_at
        FROM media
        WHERE id = ANY($1::bigint[])
        """,
        media_ids,
    )
    by_id = {int(row["id"]): row for row in rows}
    return [
        _search_row_to_dict(by_id[media_id], float(item["distance"]))
        for item, media_id in zip(neighbors, media_ids)
        if media_id in by_id
    ]


@search_router.get("/api/v1/search/text")
async def search_text(q: str, request: Request, limit: int = 50):
    """Encode text in the active model space and query that namespace only."""
    if not q or not q.strip():
        return {"items": [], "query": q}
    limit = min(max(limit, 1), 200)
    import model_store

    pool = request.app.state.db
    async with pool.acquire() as conn:
        active_model = await model_store.get_active_model_async(conn, "scene")
    if active_model is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "no active scene model is configured"},
        )

    try:
        from scene_engine import get_engine

        api_device = os.getenv("MEMORIA_API_ML_DEVICE", "cpu").strip() or "cpu"

        def _encode_active_text() -> list[float]:
            engine = get_engine(active_model, device=api_device)
            return engine.encode_text(q.strip())

        # First use may load/download a large model. Do not hold a pooled DB
        # connection while that blocking work runs.
        with metrics.timer("clip.encode_text"):
            qvec = await asyncio.to_thread(_encode_active_text)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"detail": f"encoder unavailable: {exc}"},
        )

    async with pool.acquire() as conn:
        async with conn.transaction():
            # SET LOCAL bumps HNSW recall for this query only.
            await conn.execute(
                f"SET LOCAL hnsw.ef_search = {_HNSW_EF_SEARCH}"
            )
            with metrics.timer("db.vector_knn", route="/api/v1/search/text"):
                neighbors = await model_store.query_scene_neighbors_async(
                    conn,
                    qvec,
                    model_key=active_model["model_key"],
                    limit=limit,
                )
            metrics.record(
                "db.vector_knn.hits",
                len(neighbors),
                route="/api/v1/search/text",
            )
        items = await _hydrate_scene_neighbors(conn, neighbors)
    return {"items": items, "query": q}


@search_router.get("/api/v1/search/similar/{media_id}")
async def search_similar(media_id: int, request: Request, limit: int = 50):
    """Return neighbors from the source photo's active scene namespace."""
    limit = min(max(limit, 1), 200)
    import model_store

    pool = request.app.state.db
    async with pool.acquire() as conn:
        src = await model_store.get_active_scene_embedding_async(conn, media_id)
        if src is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "media has no active scene embedding"},
            )
        async with conn.transaction():
            # SET LOCAL bumps HNSW recall for this query only.
            await conn.execute(
                f"SET LOCAL hnsw.ef_search = {_HNSW_EF_SEARCH}"
            )
            with metrics.timer("db.vector_knn", route="/api/v1/search/similar"):
                neighbors = await model_store.query_scene_neighbors_async(
                    conn,
                    src["embedding"],
                    model_key=src["model_key"],
                    limit=limit,
                    exclude_media_id=media_id,
                )
            metrics.record(
                "db.vector_knn.hits",
                len(neighbors),
                route="/api/v1/search/similar",
            )
        items = await _hydrate_scene_neighbors(conn, neighbors)
    return {"items": items}


# ── Processing config (per-stage feature toggles + presets) ──────────────────
processing_router = APIRouter()
_processing_logger = logging.getLogger("memoria.processing")


async def _read_processing_config(conn) -> dict:
    """Read the stored processing config, falling back to all-on defaults."""
    raw = await conn.fetchval(
        "SELECT value FROM app_settings WHERE key = $1",
        processing_config.SETTINGS_KEY,
    )
    if raw is None:
        features = processing_config.coerce_features({})
        return {"preset": "full", "features": features}
    if isinstance(raw, str):
        raw = json.loads(raw)
    features = processing_config.coerce_features(raw.get("features", {}))
    preset = raw.get("preset") or "custom"
    return {"preset": preset, "features": features}


def _model_profile(record) -> dict | None:
    """Return one persisted/spec model in the frontend's compatibility shape."""
    if record is None:
        return None
    if hasattr(record, "as_dict"):
        payload = dict(record.as_dict())
    else:
        payload = dict(record)

    model_name = payload.get("model_name") or payload.get("model_id")
    embedding_dim = payload.get("embedding_dim")
    if embedding_dim is None:
        embedding_dim = payload.get("embedding_dimension")
    resolution = payload.get("input_resolution")
    if isinstance(resolution, tuple):
        resolution = list(resolution)
    if resolution is None:
        width = payload.get("input_width")
        height = payload.get("input_height")
        if width is not None or height is not None:
            resolution = [width if width is not None else height,
                          height if height is not None else width]
    providers = (
        payload.get("preferred_providers")
        or payload.get("provider_preferences")
        or []
    )
    if isinstance(providers, tuple):
        providers = list(providers)

    payload.update({
        "model_name": model_name,
        "model_id": payload.get("model_id") or model_name,
        "embedding_dim": embedding_dim,
        "embedding_dimension": payload.get("embedding_dimension", embedding_dim),
        "input_resolution": resolution,
        "preferred_providers": providers,
        "provider_preferences": payload.get("provider_preferences") or providers,
        "precision": payload.get("precision") or payload.get("default_precision"),
    })
    return payload


async def _processing_model_status(conn) -> dict:
    """Build the scene/face model catalog and durable lifecycle snapshot.

    This reads descriptors and jobs only; it deliberately never loads model
    weights or probes an accelerator merely because Settings was opened.
    """
    import model_store
    from ml.face import (
        FACENET_VGGFACE2_SPEC,
        INSIGHTFACE_ARCFACE_SPEC,
        FaceEngineConfig,
    )
    from scene_engine import builtin_profiles, configured_model_spec

    # asyncpg permits one in-flight operation per connection, so keep these
    # small catalog reads sequential while holding a single pooled connection.
    scene_state = await model_store.get_model_state_async(conn, "scene")
    face_state = await model_store.get_model_state_async(conn, "face-embedding")
    scene_rows = await model_store.list_models_async(conn, "scene")
    face_rows = await model_store.list_models_async(conn, "face-embedding")
    scene_jobs = await model_store.list_reindex_jobs_async(
        conn, kind="scene", limit=1
    )

    scene_profiles = {
        row["model_key"]: _model_profile(row) for row in scene_rows
    }
    for spec in builtin_profiles():
        scene_profiles.setdefault(spec.model_key, _model_profile(spec))

    configured_scene = None
    scene_config_error = None
    try:
        configured_scene = _model_profile(configured_model_spec())
        if configured_scene:
            scene_profiles.setdefault(configured_scene["model_key"], configured_scene)
    except Exception as exc:
        scene_config_error = str(exc)

    configured_face = None
    configured_face_embedding = None
    face_config_error = None
    try:
        face_config = FaceEngineConfig.from_env()
        configured_face = _model_profile(face_config.to_model_spec())
        face_spec = (
            FACENET_VGGFACE2_SPEC
            if face_config.embedder == "facenet"
            else INSIGHTFACE_ARCFACE_SPEC
            if face_config.embedder == "insightface"
            else None
        )
        configured_face_embedding = _model_profile(face_spec)
    except Exception as exc:
        face_config_error = str(exc)

    scene_status = {
        "configured": configured_scene,
        "desired": _model_profile(scene_state.get("desired_model")) if scene_state else None,
        "active": _model_profile(scene_state.get("active_model")) if scene_state else None,
        "status": scene_state.get("status", "unconfigured") if scene_state else "unconfigured",
        "generation": scene_state.get("generation", 0) if scene_state else 0,
        "last_error": (
            scene_state.get("last_error") if scene_state else None
        ) or scene_config_error,
        "reindex_job": scene_jobs[0] if scene_jobs else None,
        "profiles": list(scene_profiles.values()),
    }
    face_status = {
        "configured": configured_face_embedding or configured_face,
        "configured_embedding": configured_face_embedding,
        "pipeline": configured_face,
        "desired": _model_profile(face_state.get("desired_model")) if face_state else None,
        "active": _model_profile(face_state.get("active_model")) if face_state else None,
        "status": face_state.get("status", "unconfigured") if face_state else "unconfigured",
        "generation": face_state.get("generation", 0) if face_state else 0,
        "last_error": (
            face_state.get("last_error") if face_state else None
        ) or face_config_error,
        "profiles": [_model_profile(row) for row in face_rows],
    }
    return {"scene": scene_status, "face": face_status}


@processing_router.get("/api/v1/processing/config")
async def processing_get_config(request: Request):
    """Return the current processing config + the catalog of presets."""
    pool = request.app.state.db
    async with pool.acquire() as conn:
        cfg = await _read_processing_config(conn)
        models = await _processing_model_status(conn)
    return {
        "preset": cfg["preset"],
        "features": cfg["features"],
        "feature_keys": list(processing_config.FEATURE_KEYS),
        "presets": processing_config.PRESETS,
        "models": models,
    }


@processing_router.put("/api/v1/processing/config")
async def processing_put_config(request: Request):
    """Update the processing config from ``{preset?, features?}``.

    SSCD-on forces thumbnail-on (see processing_config.normalize). Unknown
    presets/feature keys → 400.
    """
    body = await request.json()
    try:
        resolved = processing_config.normalize(body)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    pool = request.app.state.db
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES ($1, $2::jsonb, NOW())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """,
                processing_config.SETTINGS_KEY,
                json.dumps(resolved),
            )
    _processing_logger.info("processing config updated -> %s", resolved)
    return {"status": "ok", **resolved}


@processing_router.get("/api/v1/processing/models")
async def processing_get_models(request: Request):
    """Return immutable model descriptors plus desired/active lifecycle state."""
    pool = request.app.state.db
    async with pool.acquire() as conn:
        return await _processing_model_status(conn)


@processing_router.post("/api/v1/processing/reindex/scene", status_code=202)
async def processing_start_scene_reindex(
    request: Request,
    profile: str | None = None,
    force: bool = False,
):
    """Stage a scene descriptor and idempotently queue its durable re-index."""
    import model_store
    from constants import IMAGE_EXTENSIONS
    from scene_engine import configured_model_spec, model_spec

    pool = request.app.state.db
    try:
        async with pool.acquire() as conn:
            # Descriptor registration, desired-state transition, coverage
            # snapshot, and job creation are one durable API operation. The
            # store helpers use nested savepoints when called in this outer
            # transaction, so any failure leaves the previous desired state
            # intact instead of stranding it without a job.
            async with conn.transaction():
                target = None
                if profile and profile.strip():
                    selector = profile.strip()
                    target = await model_store.get_model_async(conn, selector)
                    if target is None:
                        try:
                            selected_spec = model_spec(selector)
                        except KeyError:
                            return JSONResponse(
                                status_code=404,
                                content={
                                    "detail": f"unknown scene profile: {selector}"
                                },
                            )
                        if selected_spec.task != "scene":
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "detail": "selected model is not a scene model"
                                },
                            )
                        target = await model_store.register_model_async(
                            conn, selected_spec
                        )
                else:
                    current = await model_store.get_model_state_async(conn, "scene")
                    target = current.get("desired_model") if current else None
                    if target is None:
                        target = await model_store.register_model_async(
                            conn, configured_model_spec()
                        )

                if target.get("kind") != "scene":
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "selected model is not a scene model"},
                    )

                await model_store.reconcile_model_state_async(
                    conn, "scene", target["model_key"]
                )
                extensions = sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS)
                if force:
                    total_items = await conn.fetchval(
                        "SELECT COUNT(*) FROM media WHERE extension = ANY($1::text[])",
                        extensions,
                    )
                else:
                    total_items = await conn.fetchval(
                        """
                        SELECT COUNT(*)
                        FROM media m
                        LEFT JOIN scene_embeddings se
                          ON se.media_id = m.id AND se.model_key = $1
                        WHERE m.extension = ANY($2::text[])
                          AND se.media_id IS NULL
                        """,
                        target["model_key"],
                        extensions,
                    )
                try:
                    batch_size = int(
                        os.getenv("MEMORIA_SCENE_REINDEX_BATCH_SIZE", "16")
                    )
                except ValueError:
                    batch_size = 16
                batch_size = max(1, min(batch_size, 64))
                job, created = await model_store.create_reindex_job_async(
                    conn,
                    kind="scene",
                    model_key=target["model_key"],
                    total_items=int(total_items or 0),
                    force_reindex=bool(force),
                    batch_size=batch_size,
                )
    except (ValueError, model_store.ModelStoreError) as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # Queue only after every state/job write above is committed. Re-dispatching
    # an existing active job is a safe crash-recovery signal for the one-worker
    # consumer; its durable cursor makes repeated continuations idempotent.
    from workers.tasks import reindex_scene_embeddings

    reindex_scene_embeddings(str(job["id"]))
    _processing_logger.info(
        "%s scene re-index job=%s model=%s force=%s items=%s",
        "queued" if created else "resumed",
        job["id"],
        job["model_key"],
        bool(force),
        job["total_items"],
    )
    return {"job": job, "created": created}


@processing_router.get("/api/v1/processing/reindex/{job_id}")
async def processing_get_reindex_job(job_id: str, request: Request):
    """Inspect one durable scene/face re-index job."""
    import model_store

    pool = request.app.state.db
    try:
        async with pool.acquire() as conn:
            job = await model_store.get_reindex_job_async(conn, job_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "re-index job not found"})
    return {"job": job}


@processing_router.post("/api/v1/processing/reindex/{job_id}/cancel")
async def processing_cancel_reindex_job(job_id: str, request: Request):
    """Cooperatively request cancellation; the worker acknowledges it."""
    import model_store

    pool = request.app.state.db
    try:
        async with pool.acquire() as conn:
            job = await model_store.get_reindex_job_async(conn, job_id)
            if job is None:
                return JSONResponse(
                    status_code=404,
                    content={"detail": "re-index job not found"},
                )
            job = await model_store.request_reindex_cancel_async(conn, job_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    return {"job": job}


@processing_router.get("/api/v1/processing/benchmarks")
async def processing_get_benchmarks(
    request: Request,
    limit: int = 50,
    model_key: str | None = None,
    benchmark_type: str | None = None,
):
    """Return recent persisted model measurements, newest first."""
    import model_store

    pool = request.app.state.db
    async with pool.acquire() as conn:
        benchmarks = await model_store.list_model_benchmarks_async(
            conn,
            model_key=model_key,
            benchmark_type=benchmark_type,
            limit=limit,
        )
    return {"benchmarks": benchmarks}


@processing_router.get("/api/v1/processing/runs")
async def processing_get_runs(limit: int = 20):
    """Return recent persistent throughput records (newest first)."""
    return throughput.read_recent(limit)


# ── Aggregated list for main.py ──────────────────────────────────────────────

# -- Library on-demand scan ---------------------------------------------------
library_router = APIRouter()
_library_logger = logging.getLogger("memoria.library")


@library_router.get("/api/v1/library/browse")
async def library_browse_directory(path: str | None = None):
    """Browse configured library roots from the server side."""
    from library_paths import browse_directory, discover_library_mounts

    # On-demand browsing has a broader, explicit read-only bind scope than the
    # automatic watcher. Using CRAWL_ROOTS here made MEMORIA_HOST_MOUNT folders
    # impossible to browse even though the scan resolver could see them.
    roots = await asyncio.to_thread(discover_library_mounts)
    try:
        return await asyncio.to_thread(browse_directory, path, roots)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


@library_router.post("/api/v1/library/scan", status_code=202)
async def library_scan_folder(request: Request):
    """Queue an incremental scan for a server-visible folder."""
    from library_paths import (
        MAX_LIBRARY_SCAN_BODY_BYTES,
        discover_library_mounts,
        resolve_picked_scan_root,
        validate_picked_scan_request,
        validate_scan_path,
    )
    from workers.tasks import scan_library_folder

    body_chunks: list[bytes] = []
    body_size = 0
    async for chunk in request.stream():
        body_size += len(chunk)
        if body_size > MAX_LIBRARY_SCAN_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "library scan request body is too large"},
            )
        body_chunks.append(chunk)
    raw_body = b"".join(body_chunks)
    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(status_code=400, content={"detail": "invalid JSON body"})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"detail": "invalid request body"})

    mounts = await asyncio.to_thread(discover_library_mounts)
    requested_path = body.get("path")
    if isinstance(requested_path, str) and requested_path.strip():
        try:
            path = await asyncio.to_thread(
                validate_scan_path, requested_path.strip(), mounts
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

        await asyncio.to_thread(scan_library_folder, path)
        folder_name = os.path.basename(path.rstrip(os.sep)) or path
        _library_logger.info("queued on-demand library scan for explicit path %r", path)
        return {
            "status": "queued",
            "folder_name": folder_name,
            "path": path,
            "matched_files": None,
        }

    try:
        folder_name, files, total_files = validate_picked_scan_request(
            body.get("folder_name"),
            body.get("files"),
            body.get("total_files"),
        )
        path = await asyncio.to_thread(
            resolve_picked_scan_root,
            folder_name,
            files,
            mounts,
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    await asyncio.to_thread(scan_library_folder, path)
    _library_logger.info(
        "queued on-demand library scan for picked folder %r -> %s",
        folder_name,
        path,
    )
    return {
        "status": "queued",
        "folder_name": folder_name,
        "path": path,
        "matched_files": total_files,
        "sampled_files": len(files),
    }
ALL_ROUTERS = [
    dedup_router,
    faces_router,
    blurry_router,
    map_router,
    editor_router,
    search_router,
    processing_router,
    library_router,
]
