"""HNSW tuning for the 159k+ scale (Phase E5)

What this migration does, in plain English:
  HNSW indexes have two main knobs: ``m`` (how many edges each node has
  in the graph) and ``ef_construction`` (how aggressively to explore
  when building the index). pgvector's defaults (m=16, ef_construction=64)
  work fine up to a hundred thousand or so vectors; past that, recall
  starts dropping. This migration rebuilds the two existing HNSW indexes
  with (m=24, ef_construction=128) so they keep returning the *right*
  neighbours as the library grows.

Migrations 004 and 005 created ``face_embeddings_hnsw`` and
``media_scene_hnsw`` with pgvector library defaults (``m = 16``,
``ef_construction = 64``). Recall holds at ~159k rows but has no headroom
as the corpus grows. This revision drops + recreates both indexes with
``WITH (m = 24, ef_construction = 128)`` for better recall-vs-speed trade.

Apply BEFORE the initial bulk crawl finishes so the index builds
incrementally on the better params, instead of doing a slow post-hoc
rebuild on a fully populated table.

Safe to run on a DB where:
  * pgvector is not installed       → skip, no-op (logs a NOTICE).
  * face_embeddings_hnsw absent     → IF EXISTS makes the drop a no-op.
  * media.scene_embedding absent    → guarded by a column check (the
                                       column is created by 005 only when
                                       pgvector is present).

Revision ID: 006_hnsw_tune
Revises: 005_scene_embeddings
Create Date: 2026-04-26
"""

from __future__ import annotations

from alembic import op

revision = "006_hnsw_tune"
down_revision = "005_scene_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        DECLARE
            has_pgvector  BOOLEAN;
            has_scene_col BOOLEAN;
        BEGIN
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            ) INTO has_pgvector;

            IF NOT has_pgvector THEN
                RAISE NOTICE 'pgvector not installed - skipping HNSW retune';
                RETURN;
            END IF;

            -- face_embeddings_hnsw: drop + recreate with tuned params.
            -- The table+column are guaranteed to exist (created in 001,
            -- retyped in 004), so no extra column probe needed here.
            EXECUTE 'DROP INDEX IF EXISTS face_embeddings_hnsw';
            EXECUTE 'CREATE INDEX face_embeddings_hnsw '
                 || 'ON face_embeddings USING hnsw '
                 || '(embedding vector_cosine_ops) '
                 || 'WITH (m = 24, ef_construction = 128)';

            -- media_scene_hnsw: only attempt if the column exists. Migration
            -- 005 may have no-op'd on a DB where pgvector was missing at
            -- the time it ran.
            SELECT EXISTS (
                SELECT 1 FROM pg_attribute
                 WHERE attrelid = 'media'::regclass
                   AND attname = 'scene_embedding'
                   AND NOT attisdropped
            ) INTO has_scene_col;

            IF has_scene_col THEN
                EXECUTE 'DROP INDEX IF EXISTS media_scene_hnsw';
                EXECUTE 'CREATE INDEX media_scene_hnsw '
                     || 'ON media USING hnsw '
                     || '(scene_embedding vector_cosine_ops) '
                     || 'WITH (m = 24, ef_construction = 128)';
            ELSE
                RAISE NOTICE 'media.scene_embedding column missing - skipping its HNSW retune';
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    """Revert to library default HNSW params (m=16, ef_construction=64)."""
    op.execute(
        """
        DO $$
        DECLARE
            has_pgvector  BOOLEAN;
            has_scene_col BOOLEAN;
        BEGIN
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            ) INTO has_pgvector;

            IF NOT has_pgvector THEN
                RAISE NOTICE 'pgvector not installed - nothing to downgrade';
                RETURN;
            END IF;

            EXECUTE 'DROP INDEX IF EXISTS face_embeddings_hnsw';
            EXECUTE 'CREATE INDEX face_embeddings_hnsw '
                 || 'ON face_embeddings USING hnsw '
                 || '(embedding vector_cosine_ops)';

            SELECT EXISTS (
                SELECT 1 FROM pg_attribute
                 WHERE attrelid = 'media'::regclass
                   AND attname = 'scene_embedding'
                   AND NOT attisdropped
            ) INTO has_scene_col;

            IF has_scene_col THEN
                EXECUTE 'DROP INDEX IF EXISTS media_scene_hnsw';
                EXECUTE 'CREATE INDEX media_scene_hnsw '
                     || 'ON media USING hnsw '
                     || '(scene_embedding vector_cosine_ops)';
            END IF;
        END
        $$;
        """
    )
