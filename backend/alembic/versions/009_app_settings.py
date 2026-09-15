"""app_settings: key/value store for runtime processing config

Adds a generic JSONB key/value table so the API and the (separate-process)
Huey worker can share runtime settings. The first consumer is the per-stage
processing feature toggles + preset that drive Dedup-first mode.

The seeded default has every feature enabled, which is exactly the pipeline's
behavior before this migration — so applying it changes nothing until a user
picks a preset from the UI.

Revision ID: 009_app_settings
Revises: 008_face_cluster_hidden
Create Date: 2026-05-28
"""

from __future__ import annotations

from alembic import op

revision = "009_app_settings"
down_revision = "008_face_cluster_hidden"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Plain idempotent DDL — CREATE TABLE IF NOT EXISTS + ON CONFLICT DO NOTHING.
    # The seed value is built with jsonb_build_object (rather than a JSON string
    # literal) so the ':' in JSON object syntax can't collide with SQLAlchemy's
    # ':name' bind-parameter parser inside op.execute.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key        TEXT PRIMARY KEY,
            value      JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute(
        """
        INSERT INTO app_settings (key, value)
        VALUES (
            'processing_features',
            jsonb_build_object(
                'preset', 'full',
                'features', jsonb_build_object(
                    'thumbnail', true,
                    'exif', true,
                    'phash', true,
                    'blur', true,
                    'faces', true,
                    'scene_clip', true,
                    'sscd_dedup', true
                )
            )
        )
        ON CONFLICT (key) DO NOTHING;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS app_settings;")
