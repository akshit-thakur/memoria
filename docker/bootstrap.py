"""Wait for PostgreSQL, enable required extensions, and migrate the schema."""

from __future__ import annotations

import logging
import os
import subprocess
import time

import psycopg


logging.basicConfig(level=logging.INFO, format="[bootstrap] %(message)s")
logger = logging.getLogger("memoria.bootstrap")

ALEMBIC_HEAD = "012_vector_schema_repair"

# Alembic revisions predating 012 could be recorded even when pgvector DDL was
# skipped.  This catalog-only check is intentionally cheap enough to run on
# every container boot and prevents a partially repaired database from being
# advertised as ready.
SCHEMA_POSTCONDITION_SQL = """
WITH required_extensions(extension_name) AS (
    VALUES ('vector'), ('uuid-ossp')
),
required_relations(relation_name) AS (
    VALUES
        ('public.media'),
        ('public.scene_embeddings'),
        ('public.ml_models'),
        ('public.ml_model_state'),
        ('public.ml_reindex_jobs'),
        ('public.ml_processing_state'),
        ('public.ml_model_benchmarks'),
        ('public.face_embeddings'),
        ('public.face_clusters'),
        ('public.face_cluster_merges'),
        ('public.duplicate_groups'),
        ('public.duplicate_members'),
        ('public.media_edits'),
        ('public.app_settings'),
        ('public.processing_log'),
        ('public.media_scene_hnsw'),
        ('public.media_dedup_hnsw'),
        ('public.idx_scene_embeddings_model_media'),
        ('public.scene_openclip_vitl14_hnsw'),
        ('public.scene_openclip_siglip1152_hnsw'),
        ('public.scene_hf_siglip2_1152_hnsw'),
        ('public.face_arcface_512_hnsw'),
        ('public.face_facenet_512_hnsw'),
        ('public.face_legacy_512_hnsw')
),
required_columns(relation_name, column_name, expected_type) AS (
    VALUES
        ('public.face_embeddings', 'embedding', 'vector(512)'),
        ('public.media', 'scene_embedding', 'vector(768)'),
        ('public.media', 'dedup_embedding', 'vector(512)'),
        ('public.scene_embeddings', 'embedding', 'vector')
),
required_constraints(relation_name, constraint_name, required_fragment) AS (
    VALUES
        (
            'public.face_embeddings',
            'face_embedding_vector_dimension_ck',
            'vector_dims'
        ),
        (
            'public.face_embeddings',
            'fk_face_embedding_model_dimension',
            NULL::TEXT
        ),
        ('public.scene_embeddings', 'scene_embeddings_pkey', NULL::TEXT),
        (
            'public.scene_embeddings',
            'scene_embeddings_media_id_fkey',
            NULL::TEXT
        ),
        (
            'public.scene_embeddings',
            'scene_embeddings_model_key_fkey',
            NULL::TEXT
        ),
        (
            'public.scene_embeddings',
            'fk_scene_embedding_model_dimension',
            NULL::TEXT
        ),
        (
            'public.scene_embeddings',
            'scene_embeddings_dimension_ck',
            'vector_dims'
        )
),
problems(problem) AS (
    SELECT 'extension:' || extension_name
    FROM required_extensions required
    WHERE NOT EXISTS (
        SELECT 1 FROM pg_extension
        WHERE extname = required.extension_name
    )

    UNION ALL

    SELECT 'relation:' || relation_name
    FROM required_relations
    WHERE to_regclass(relation_name) IS NULL

    UNION ALL

    SELECT
        'column:' || relation_name || '.' || column_name
        || ' expected ' || expected_type
    FROM required_columns required
    WHERE NOT EXISTS (
        SELECT 1
        FROM pg_attribute attribute
        WHERE attribute.attrelid = to_regclass(required.relation_name)
          AND attribute.attname = required.column_name
          AND NOT attribute.attisdropped
          AND format_type(attribute.atttypid, attribute.atttypmod)
              = required.expected_type
    )

    UNION ALL

    SELECT 'constraint:' || relation_name || '.' || constraint_name
    FROM required_constraints required
    WHERE NOT EXISTS (
        SELECT 1
        FROM pg_constraint constraint_row
        WHERE constraint_row.conrelid = to_regclass(required.relation_name)
          AND constraint_row.conname = required.constraint_name
          AND (
              required.required_fragment IS NULL
              OR position(
                  required.required_fragment
                  IN pg_get_constraintdef(constraint_row.oid)
              ) > 0
          )
    )

    UNION ALL

    SELECT 'relation:public.face_embeddings_hnsw must be absent'
    WHERE to_regclass('public.face_embeddings_hnsw') IS NOT NULL

    UNION ALL

    SELECT 'alembic:expected single head ' || %s
    WHERE (SELECT COUNT(*) FROM alembic_version) <> 1
       OR NOT EXISTS (
           SELECT 1 FROM alembic_version WHERE version_num = %s
       )
)
SELECT COALESCE(
    array_agg(problem ORDER BY problem),
    ARRAY[]::TEXT[]
)
FROM problems
"""


def schema_postcondition_problems(conn) -> list[str]:
    """Return missing or incompatible schema invariants for a live connection."""

    row = conn.execute(
        SCHEMA_POSTCONDITION_SQL,
        (ALEMBIC_HEAD, ALEMBIC_HEAD),
    ).fetchone()
    if row is None:
        return ["schema postcondition query returned no row"]
    return [str(problem) for problem in (row[0] or [])]


def validate_schema(database_url: str) -> None:
    """Fail container startup unless migrations produced the complete schema."""

    with psycopg.connect(
        database_url,
        connect_timeout=3,
    ) as conn:
        problems = schema_postcondition_problems(conn)
    if problems:
        raise RuntimeError(
            "database migrations completed but schema postconditions failed: "
            + ", ".join(problems)
        )


def prepare_database() -> None:
    database_url = os.environ["DATABASE_URL"]
    wait_seconds = max(1, int(os.getenv("MEMORIA_DB_WAIT_SECONDS", "90")))
    deadline = time.monotonic() + wait_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with psycopg.connect(
                database_url,
                connect_timeout=3,
            ) as conn:
                conn.execute("SELECT 1")
            break
        except Exception as exc:  # PostgreSQL may still be starting.
            last_error = exc
            logger.info("waiting for PostgreSQL at %s:%s", os.getenv("DB_HOST"), os.getenv("DB_PORT"))
            time.sleep(2)
    else:
        raise RuntimeError(
            f"PostgreSQL did not become ready within {wait_seconds}s"
        ) from last_error

    try:
        with psycopg.connect(
            database_url,
            autocommit=True,
            connect_timeout=3,
        ) as conn:
            # The extension shipped by pgvector is named "vector". Run these
            # on every boot so pre-existing volumes are repaired too.
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    except Exception as exc:
        raise RuntimeError(
            "PostgreSQL is reachable, but the configured database user cannot "
            "enable the required vector and uuid-ossp extensions"
        ) from exc

    logger.info("database ready; applying Alembic migrations")
    subprocess.run(
        ["/opt/venv/bin/alembic", "-c", "/app/backend/alembic.ini", "upgrade", "head"],
        cwd="/app/backend",
        check=True,
    )
    validate_schema(database_url)
    logger.info("schema is at Alembic head and passed postcondition checks")


if __name__ == "__main__":
    if os.getenv("MEMORIA_SCHEMA_VALIDATE_ONLY", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        validate_schema(os.environ["DATABASE_URL"])
        logger.info("schema passed postcondition checks")
    else:
        prepare_database()
