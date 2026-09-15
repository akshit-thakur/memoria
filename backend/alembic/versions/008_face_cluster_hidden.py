"""face_clusters: add is_hidden + hidden_at columns for soft-delete

Adds is_hidden and hidden_at to face_clusters so clusters can be marked
hidden without being hard-deleted. The clustering worker's absorption
pre-pass uses is_hidden = TRUE to silently redirect new faces of hidden
persons back into the hidden cluster, preventing respawn.

Revision ID: 008_face_cluster_hidden
Revises: 007_dedup_embeddings
Create Date: 2026-05-24
"""

from __future__ import annotations

from alembic import op

revision = "008_face_cluster_hidden"
down_revision = "007_dedup_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_attribute
                 WHERE attrelid = 'face_clusters'::regclass
                   AND attname = 'is_hidden'
                   AND NOT attisdropped
            ) THEN
                EXECUTE 'ALTER TABLE face_clusters ADD COLUMN is_hidden BOOLEAN NOT NULL DEFAULT FALSE';
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_attribute
                 WHERE attrelid = 'face_clusters'::regclass
                   AND attname = 'hidden_at'
                   AND NOT attisdropped
            ) THEN
                EXECUTE 'ALTER TABLE face_clusters ADD COLUMN hidden_at TIMESTAMPTZ';
            END IF;

            EXECUTE 'CREATE INDEX IF NOT EXISTS face_clusters_hidden_idx
                     ON face_clusters(is_hidden) WHERE is_hidden = TRUE';
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS face_clusters_hidden_idx;")
    op.execute("ALTER TABLE face_clusters DROP COLUMN IF EXISTS hidden_at;")
    op.execute("ALTER TABLE face_clusters DROP COLUMN IF EXISTS is_hidden;")
