"""perf_indexes: add missing indexes for 300k-item scale

Three indexes that were absent and causing full-table scans:
  - idx_face_cluster_id: cluster_id on face_embeddings (~1M rows) used by
    auto-merge and label update queries in routers.py.
  - idx_media_blur_score: partial index on blur_score for blurry list and
    dashboard COUNT queries.
  - idx_proclog_status_updated: composite (status, updated_at) on
    processing_log for the /progress done-count query and stall detection.
"""

from alembic import op

revision = "010_perf_indexes"
down_revision = "009_app_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_face_cluster_id "
        "ON face_embeddings (cluster_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_media_blur_score "
        "ON media (blur_score DESC) WHERE blur_score > 0;"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_proclog_status_updated "
        "ON processing_log (status, updated_at);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_face_cluster_id;")
    op.execute("DROP INDEX IF EXISTS idx_media_blur_score;")
    op.execute("DROP INDEX IF EXISTS idx_proclog_status_updated;")
