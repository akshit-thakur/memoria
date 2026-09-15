"""media.scene_embedding (CLIP) + HNSW

What this migration does, in plain English:
  Adds the column that stores each photo's CLIP scene-embedding vector
  (exactly 768 numbers) and builds an HNSW index over it. This originally
  powered natural-language search ("beach sunset") and image-similar
  ("find more photos like this one"). Existing photos get NULL until the
  worker backfills them.

Adds a nullable ``scene_embedding`` column to the ``media`` table plus an HNSW
cosine index so ``/api/v1/search/text`` and ``/api/v1/search/similar/{id}``
hit ANN instead of seq-scanning. The column is nullable on purpose — rows
indexed before this migration (or before the CLIP pipeline step lands) carry
NULL until ``backfill_scene_embeddings()`` fills them in.

Safe to run on a DB where:
  * pgvector is not installed   → skip, no-op (logs a NOTICE).
  * column already exists       → IF NOT EXISTS.
  * index already exists        → IF NOT EXISTS.

This legacy compatibility column is permanently 768-dimensional for
OpenCLIP ViT-L/14. Phase 2 model-specific dimensions live in the untyped
``scene_embeddings.embedding`` column introduced by revision 011; runtime
model selection must never change the schema produced by this revision.

Revision ID: 005_scene_embeddings
Revises: 004_pgvector_hnsw
Create Date: 2026-04-26
"""

from __future__ import annotations

from alembic import op

revision = "005_scene_embeddings"
down_revision = "004_pgvector_hnsw"
branch_labels = None
depends_on = None

_DIM = 768


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
                RAISE NOTICE 'pgvector not installed - skipping scene_embedding migration';
                RETURN;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_attribute
                 WHERE attrelid = 'media'::regclass
                   AND attname = 'scene_embedding'
                   AND NOT attisdropped
            ) THEN
                EXECUTE 'ALTER TABLE media ADD COLUMN scene_embedding vector({_DIM})';
            END IF;

            EXECUTE 'CREATE INDEX IF NOT EXISTS media_scene_hnsw '
                 || 'ON media USING hnsw '
                 || '(scene_embedding vector_cosine_ops)';
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS media_scene_hnsw;")
    op.execute("ALTER TABLE media DROP COLUMN IF EXISTS scene_embedding;")
