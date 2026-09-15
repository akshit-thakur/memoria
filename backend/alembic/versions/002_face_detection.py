"""face detection: extend face_embeddings + face_clusters; add face_cluster_merges

What this migration does, in plain English:
  Adds three columns to ``face_embeddings`` so we can record which model
  detected the face and how confident it was, then adds three rollup
  columns to ``face_clusters`` (representative photo, total face count,
  average confidence). Finally creates a ``face_cluster_merges`` audit
  table so the auto-merge logic in ``routers.py`` (rename a cluster →
  same-label clusters merge) has a paper trail.

Originally introduced as database/migrations/002_face_detection.sql. Adds
multi-model detection metadata, per-cluster rollups, and a merge audit log.

Revision ID: 002_face_detection
Revises: 001_initial_schema
Create Date: 2026-04-23
"""

from __future__ import annotations

from alembic import op

revision = "002_face_detection"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE face_embeddings
            ADD COLUMN IF NOT EXISTS detection_model TEXT DEFAULT 'ensemble';
        """
    )
    op.execute(
        """
        ALTER TABLE face_embeddings
            ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION DEFAULT 0.0;
        """
    )

    op.execute(
        """
        ALTER TABLE face_clusters
            ADD COLUMN IF NOT EXISTS representative_media_id BIGINT
            REFERENCES media(id) ON DELETE SET NULL;
        """
    )
    op.execute(
        """
        ALTER TABLE face_clusters
            ADD COLUMN IF NOT EXISTS face_count INT DEFAULT 0;
        """
    )
    op.execute(
        """
        ALTER TABLE face_clusters
            ADD COLUMN IF NOT EXISTS confidence_score DOUBLE PRECISION DEFAULT 0.0;
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS face_cluster_merges (
            id              BIGSERIAL    PRIMARY KEY,
            source_cluster  BIGINT       NOT NULL,
            target_cluster  BIGINT       NOT NULL,
            merged_count    INT          NOT NULL DEFAULT 0,
            merged_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        );
        """
    )

    op.execute(
        """
        CREATE OR REPLACE VIEW v_cluster_summary AS
        SELECT
            fc.id,
            fc.label,
            fc.face_count,
            fc.confidence_score,
            fc.representative_media_id,
            fc.created_at
        FROM face_clusters fc
        ORDER BY fc.face_count DESC;
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW  IF EXISTS v_cluster_summary;")
    op.execute("DROP TABLE IF EXISTS face_cluster_merges CASCADE;")
    op.execute("ALTER TABLE face_clusters   DROP COLUMN IF EXISTS confidence_score;")
    op.execute("ALTER TABLE face_clusters   DROP COLUMN IF EXISTS face_count;")
    op.execute("ALTER TABLE face_clusters   DROP COLUMN IF EXISTS representative_media_id;")
    op.execute("ALTER TABLE face_embeddings DROP COLUMN IF EXISTS confidence;")
    op.execute("ALTER TABLE face_embeddings DROP COLUMN IF EXISTS detection_model;")
