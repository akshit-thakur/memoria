"""initial schema

What this migration does, in plain English:
  Creates the database from scratch. Adds two PostgreSQL extensions
  (pgvector for vector-similarity search, uuid-ossp for unique IDs) and
  the six tables that the rest of the app needs:

    • media               — one row per photo on disk.
    • face_embeddings     — one row per detected face.
    • face_clusters       — groups of faces (one per identified person).
    • duplicate_groups    — one row per group of duplicate photos.
    • duplicate_members   — many-to-one link from media into a dup group.
    • processing_log      — bookkeeping for sidecar recovery.

  Every schema object uses an `IF NOT EXISTS` guard or a duplicate-object
  exception block, so re-running the migration against a half-bootstrapped
  database is safe.

Creates the core Memoria tables: media, face_embeddings, face_clusters,
duplicate_groups, duplicate_members, processing_log — plus the pgvector
and uuid-ossp extensions. Every statement is idempotent so this migration
can run against a database that was previously bootstrapped via the
legacy database/schema.sql.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-04-23
"""

from __future__ import annotations

from alembic import op

revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Extensions. pgvector is optional; swallow the error so boxes without
    # the package can still run face detection with FLOAT[] embeddings.
    # The package/project is named "pgvector", but its PostgreSQL extension
    # name is "vector" (the value stored in pg_extension.extname).
    op.execute(
        """
        DO $$ BEGIN
            CREATE EXTENSION IF NOT EXISTS "vector";
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'pgvector not available - face embeddings stored as FLOAT[]';
        END $$;
        """
    )
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

    # ── media ──────────────────────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS media (
            id              BIGSERIAL        PRIMARY KEY,
            file_path       TEXT             NOT NULL,
            filename        TEXT             NOT NULL,
            directory       TEXT             NOT NULL,
            extension       VARCHAR(16)      NOT NULL,
            file_size       BIGINT,
            mime_type       VARCHAR(128),
            phash           VARCHAR(64),
            blur_score      DOUBLE PRECISION DEFAULT 0.0,
            exif_data       JSONB            DEFAULT '{}',
            file_created_at TIMESTAMPTZ,
            file_modified_at TIMESTAMPTZ,
            indexed_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_media_file_path UNIQUE (file_path)
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_media_directory  ON media (directory);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_media_phash      ON media (phash);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_media_extension  ON media (extension);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_media_indexed_at ON media (indexed_at DESC);")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_photos_keyset ON media (indexed_at DESC, id DESC);"
    )

    # ── face_embeddings ────────────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id          BIGSERIAL        PRIMARY KEY,
            media_id    BIGINT           NOT NULL REFERENCES media(id) ON DELETE CASCADE,
            bbox        JSONB            NOT NULL,
            embedding   FLOAT[]          NOT NULL,
            cluster_id  BIGINT,
            created_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_face_media_id ON face_embeddings (media_id);")

    # ── face_clusters ──────────────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS face_clusters (
            id          BIGSERIAL        PRIMARY KEY,
            label       TEXT,
            created_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
        );
        """
    )

    # FK from face_embeddings.cluster_id → face_clusters(id).
    op.execute(
        """
        DO $$ BEGIN
            ALTER TABLE face_embeddings
                ADD CONSTRAINT fk_face_cluster
                FOREIGN KEY (cluster_id) REFERENCES face_clusters(id) ON DELETE SET NULL;
        EXCEPTION WHEN duplicate_object THEN
            NULL;  -- constraint already present
        END $$;
        """
    )

    # ── duplicate_groups / duplicate_members ───────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS duplicate_groups (
            id          BIGSERIAL        PRIMARY KEY,
            phash       VARCHAR(64)      NOT NULL,
            created_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS duplicate_members (
            id              BIGSERIAL    PRIMARY KEY,
            group_id        BIGINT       NOT NULL REFERENCES duplicate_groups(id) ON DELETE CASCADE,
            media_id        BIGINT       NOT NULL REFERENCES media(id) ON DELETE CASCADE,
            hamming_dist    INT          DEFAULT 0,
            UNIQUE (group_id, media_id)
        );
        """
    )

    # ── processing_log ─────────────────────────────────────────────────────
    # Legacy DBs may already have a processing_log with a different shape
    # (pre-Alembic bootstrap via database/schema.sql). CREATE TABLE IF NOT
    # EXISTS won't alter the existing columns, so we explicitly ADD every
    # column we depend on. file_path is left NULLABLE so rows written under
    # the old shape survive; new sync code writes non-null values.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS processing_log (
            id          BIGSERIAL        PRIMARY KEY,
            file_path   TEXT,
            status      TEXT             NOT NULL DEFAULT 'pending',
            attempt     INT              NOT NULL DEFAULT 1,
            error_msg   TEXT,
            created_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
            updated_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute("ALTER TABLE processing_log ADD COLUMN IF NOT EXISTS file_path  TEXT;")
    op.execute("ALTER TABLE processing_log ADD COLUMN IF NOT EXISTS attempt    INT NOT NULL DEFAULT 1;")
    op.execute("ALTER TABLE processing_log ADD COLUMN IF NOT EXISTS error_msg  TEXT;")
    op.execute("ALTER TABLE processing_log ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
    op.execute("CREATE INDEX IF NOT EXISTS idx_proclog_path   ON processing_log(file_path);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_proclog_status ON processing_log(status);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS processing_log   CASCADE;")
    op.execute("DROP TABLE IF EXISTS duplicate_members CASCADE;")
    op.execute("DROP TABLE IF EXISTS duplicate_groups  CASCADE;")
    op.execute("DROP TABLE IF EXISTS face_embeddings   CASCADE;")
    op.execute("DROP TABLE IF EXISTS face_clusters     CASCADE;")
    op.execute("DROP TABLE IF EXISTS media             CASCADE;")
