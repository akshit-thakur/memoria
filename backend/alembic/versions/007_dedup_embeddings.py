"""media.dedup_embedding (SSCD) + HNSW; duplicate_members.cos_distance

What this migration does, in plain English:
  Adds the SSCD copy-detection embedding column (512 floats per photo)
  plus its HNSW index. Together with the union-find logic in
  ``routers.py::_refresh_duplicate_groups`` this is what finds *near*
  duplicates — photos that share content but aren't byte-identical
  (different crops, lower quality, watermarked, etc.). Also adds a
  ``cos_distance`` column on ``duplicate_members`` so the UI can show
  how similar each member is to the original.

Adds a nullable ``dedup_embedding`` column to the ``media`` table plus an HNSW
cosine index so embedding-based duplicate grouping in
``/api/v1/dedup/groups`` hits ANN instead of seq-scanning. The column is
nullable on purpose — rows indexed before this migration (or before the SSCD
pipeline step lands) carry NULL until ``backfill_dedup_embeddings()`` fills
them in.

Also adds a ``cos_distance REAL`` column to ``duplicate_members`` so the
near-dup distance can be surfaced on the API. The legacy ``hamming_dist``
column is left untouched — it remains available for any future pHash-based
near-dup fallback.

Safe to run on a DB where:
  * pgvector is not installed   → skip, no-op (logs a NOTICE).
  * column already exists       → IF NOT EXISTS.
  * index already exists        → IF NOT EXISTS.

The SSCD ResNet50 embedding space is a fixed 512-dimensional invariant.
Runtime configuration must not change the schema produced by this revision.

Revision ID: 007_dedup_embeddings
Revises: 006_hnsw_tune
Create Date: 2026-04-27
"""

from __future__ import annotations

from alembic import op

revision = "007_dedup_embeddings"
down_revision = "006_hnsw_tune"
branch_labels = None
depends_on = None

_DIM = 512


def upgrade() -> None:
    op.execute(
        f"""
        DO $$
        DECLARE
            has_pgvector BOOLEAN;
        BEGIN
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            ) INTO has_pgvector;

            IF NOT has_pgvector THEN
                RAISE NOTICE 'pgvector not installed - skipping dedup_embedding migration';
                RETURN;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_attribute
                 WHERE attrelid = 'media'::regclass
                   AND attname = 'dedup_embedding'
                   AND NOT attisdropped
            ) THEN
                EXECUTE 'ALTER TABLE media ADD COLUMN dedup_embedding vector({_DIM})';
            END IF;

            EXECUTE 'CREATE INDEX IF NOT EXISTS media_dedup_hnsw '
                 || 'ON media USING hnsw '
                 || '(dedup_embedding vector_cosine_ops)';
        END
        $$;
        """
    )

    op.execute(
        """
        ALTER TABLE duplicate_members
          ADD COLUMN IF NOT EXISTS cos_distance REAL;
        """
    )

    # Embedding-formed groups have no single representative phash. Allow NULL.
    op.execute(
        """
        ALTER TABLE duplicate_groups
          ALTER COLUMN phash DROP NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE duplicate_members DROP COLUMN IF EXISTS cos_distance;")
    op.execute("DROP INDEX IF EXISTS media_dedup_hnsw;")
    op.execute("ALTER TABLE media DROP COLUMN IF EXISTS dedup_embedding;")
    # Best-effort restore: only re-add NOT NULL if no NULL phashes survive,
    # otherwise leave it nullable so downgrade doesn't crash.
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM duplicate_groups WHERE phash IS NULL) THEN
                ALTER TABLE duplicate_groups ALTER COLUMN phash SET NOT NULL;
            END IF;
        END
        $$;
        """
    )
