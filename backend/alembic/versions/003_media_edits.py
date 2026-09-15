"""media_edits table for non-destructive image edits

What this migration does, in plain English:
  Adds a ``media_edits`` table that records every non-destructive edit
  the user makes through the UI (crop, rotate, filter, sharpen). Each
  row carries the original media id, the operation name, the parameters
  used as JSON, and the path of the new file under ``~/.memoria/edits/``.
  The original photo stays untouched on disk.

Originally introduced as database/migrations/003_media_edits.sql. Tracks
crop/rotate/filter outputs written to ~/.memoria/edits/ by the InfoPanel
toolbar and BlurryList "sharpen" action. The original file is never
modified; each edit produces a new JPEG and one row here.

Revision ID: 003_media_edits
Revises: 002_face_detection
Create Date: 2026-04-23
"""

from __future__ import annotations

from alembic import op

revision = "003_media_edits"
down_revision = "002_face_detection"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS media_edits (
            id              BIGSERIAL       PRIMARY KEY,
            media_id        BIGINT          NOT NULL REFERENCES media(id) ON DELETE CASCADE,
            op              TEXT            NOT NULL CHECK (op IN ('crop','rotate','filter')),
            params          JSONB           NOT NULL DEFAULT '{}',
            output_path     TEXT            NOT NULL,
            created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_media_edits_media   ON media_edits(media_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_media_edits_created ON media_edits(created_at DESC);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS media_edits CASCADE;")
