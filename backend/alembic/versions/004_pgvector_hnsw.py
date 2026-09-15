"""pgvector HNSW index on face_embeddings.embedding

What this migration does, in plain English:
  Upgrades the face-embedding storage from a plain Postgres float array
  into pgvector's typed ``vector(512)`` column, then builds an HNSW
  (Hierarchical Navigable Small World) index over it. This is what makes
  "find similar faces" run in milliseconds instead of seconds — the
  index lets pgvector answer "give me the N closest vectors to this one"
  without scanning every row.

Promotes face_embeddings.embedding from FLOAT[] to vector(512) and adds an
HNSW cosine index so near-duplicate-face lookups go from seq-scan to ANN.

Safe to run on a DB where:
  * pgvector is not installed   → skip, no-op.
  * column is already vector(*) → skip retype, still try to create index.
  * index already exists        → IF NOT EXISTS.

Revision ID: 004_pgvector_hnsw
Revises: 003_media_edits
Create Date: 2026-04-23
"""

from __future__ import annotations

from alembic import op

revision = "004_pgvector_hnsw"
down_revision = "003_media_edits"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Everything is wrapped in a DO block so the whole migration is a
    # pure no-op when pgvector isn't present. We check pg_extension rather
    # than CREATE EXTENSION — the install is an ops decision (needs the OS
    # package), and failing the migration would block every other boot.
    op.execute(
        """
        DO $$
        DECLARE
            has_pgvector  BOOLEAN;
            current_type  TEXT;
        BEGIN
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            ) INTO has_pgvector;

            IF NOT has_pgvector THEN
                RAISE NOTICE 'pgvector not installed - skipping HNSW migration';
                RETURN;
            END IF;

            -- Retype FLOAT[] → vector(512). Skip if already converted so the
            -- migration is safe to re-run and safe on fresh DBs where the
            -- column might already be vector(*).
            SELECT format_type(atttypid, atttypmod)
              INTO current_type
              FROM pg_attribute
             WHERE attrelid = 'face_embeddings'::regclass
               AND attname = 'embedding'
               AND NOT attisdropped;

            IF current_type IS DISTINCT FROM 'vector(512)' THEN
                EXECUTE 'ALTER TABLE face_embeddings '
                     || 'ALTER COLUMN embedding TYPE vector(512) '
                     || 'USING embedding::vector(512)';
            END IF;

            -- HNSW ANN index for cosine similarity.
            EXECUTE 'CREATE INDEX IF NOT EXISTS face_embeddings_hnsw '
                 || 'ON face_embeddings USING hnsw '
                 || '(embedding vector_cosine_ops)';
        END
        $$;
        """
    )


def downgrade() -> None:
    # Drop the index unconditionally; revert column to FLOAT[] only if
    # pgvector is present (otherwise the cast syntax errors).
    op.execute("DROP INDEX IF EXISTS face_embeddings_hnsw;")
    op.execute(
        """
        DO $$
        DECLARE
            has_pgvector BOOLEAN;
        BEGIN
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            ) INTO has_pgvector;

            IF has_pgvector THEN
                EXECUTE 'ALTER TABLE face_embeddings '
                     || 'ALTER COLUMN embedding TYPE FLOAT[] '
                     || 'USING embedding::real[]';
            END IF;
        END
        $$;
        """
    )
