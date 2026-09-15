"""Repair vector schema objects skipped when pgvector was unavailable.

Revisions 001, 004, 005, 007, and 011 historically allowed their
pgvector-dependent DDL to become a no-op.  Alembic still recorded those
revisions as applied, so installing pgvector later did not create the skipped
columns, tables, constraints, or indexes.  This repair revision makes the
extensions mandatory and restores the schema that revision 011 already
describes.

Revision ID: 012_vector_schema_repair
Revises: 011_pluggable_ml_models
Create Date: 2026-08-19
"""

from __future__ import annotations

from alembic import op


revision = "012_vector_schema_repair"
down_revision = "011_pluggable_ml_models"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Vector storage is a current runtime invariant.  Do not stamp this repair
    # as complete if the package or extension privileges are missing.
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector";')
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

    # Revision 011 can have added an array-oriented dimension check when 004
    # skipped the FLOAT[] -> vector(512) conversion.  Drop that stale check
    # before changing the type, then recreate the vector-aware form.
    op.execute(
        """
        DO $memoria$
        DECLARE
            embedding_type TEXT;
            dimension_constraint TEXT;
        BEGIN
            SELECT format_type(atttypid, atttypmod)
            INTO embedding_type
            FROM pg_attribute
            WHERE attrelid = 'face_embeddings'::regclass
              AND attname = 'embedding'
              AND NOT attisdropped;

            IF embedding_type IS NULL THEN
                RAISE EXCEPTION 'face_embeddings.embedding is missing';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM face_embeddings
                WHERE embedding_dimension IS DISTINCT FROM 512
            ) THEN
                RAISE EXCEPTION
                    'face_embeddings contains non-512 embedding_dimension metadata';
            END IF;

            IF embedding_type = 'vector(512)' THEN
                NULL;
            ELSIF embedding_type LIKE 'vector%' THEN
                IF EXISTS (
                    SELECT 1 FROM face_embeddings
                    WHERE vector_dims(embedding) <> 512
                ) THEN
                    RAISE EXCEPTION
                        'face_embeddings contains vectors that are not 512-dimensional';
                END IF;
                EXECUTE 'DROP INDEX IF EXISTS face_embeddings_hnsw';
                EXECUTE 'DROP INDEX IF EXISTS face_arcface_512_hnsw';
                EXECUTE 'DROP INDEX IF EXISTS face_facenet_512_hnsw';
                EXECUTE 'DROP INDEX IF EXISTS face_legacy_512_hnsw';
                ALTER TABLE face_embeddings
                    DROP CONSTRAINT IF EXISTS face_embedding_vector_dimension_ck;
                ALTER TABLE face_embeddings
                    ALTER COLUMN embedding TYPE vector(512)
                    USING embedding::vector(512);
            ELSIF embedding_type IN ('real[]', 'double precision[]') THEN
                IF EXISTS (
                    SELECT 1 FROM face_embeddings
                    WHERE cardinality(embedding) <> 512
                ) THEN
                    RAISE EXCEPTION
                        'face_embeddings contains arrays that are not 512-dimensional';
                END IF;
                ALTER TABLE face_embeddings
                    DROP CONSTRAINT IF EXISTS face_embedding_vector_dimension_ck;
                ALTER TABLE face_embeddings
                    ALTER COLUMN embedding TYPE vector(512)
                    USING embedding::vector(512);
            ELSE
                RAISE EXCEPTION USING MESSAGE =
                    'unsupported face_embeddings.embedding type: ' || embedding_type;
            END IF;

            SELECT pg_get_constraintdef(oid)
            INTO dimension_constraint
            FROM pg_constraint
            WHERE conrelid = 'face_embeddings'::regclass
              AND conname = 'face_embedding_vector_dimension_ck';

            IF dimension_constraint IS NULL
               OR position('vector_dims' IN dimension_constraint) = 0 THEN
                ALTER TABLE face_embeddings
                    DROP CONSTRAINT IF EXISTS face_embedding_vector_dimension_ck;
                ALTER TABLE face_embeddings
                    ADD CONSTRAINT face_embedding_vector_dimension_ck
                    CHECK (vector_dims(embedding) = embedding_dimension);
            END IF;
        END
        $memoria$;
        """
    )
    op.execute("DROP INDEX IF EXISTS face_embeddings_hnsw;")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS face_arcface_512_hnsw
        ON face_embeddings USING hnsw (embedding vector_cosine_ops)
        WITH (m = 24, ef_construction = 128)
        WHERE embedding_model_key = 'face/insightface-buffalo-l-arcface';
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS face_facenet_512_hnsw
        ON face_embeddings USING hnsw (embedding vector_cosine_ops)
        WITH (m = 24, ef_construction = 128)
        WHERE embedding_model_key = 'face/facenet-vggface2';
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS face_legacy_512_hnsw
        ON face_embeddings USING hnsw (embedding vector_cosine_ops)
        WITH (m = 24, ef_construction = 128)
        WHERE embedding_model_key = 'face/legacy-unknown-512';
        """
    )

    # Restore the two vector columns that revisions 005 and 007 could skip.
    # Existing array/untyped-vector columns are accepted only when every
    # non-null value has the fixed dimension required by the runtime.
    op.execute(
        """
        DO $memoria$
        DECLARE
            current_type TEXT;
        BEGIN
            SELECT format_type(atttypid, atttypmod)
            INTO current_type
            FROM pg_attribute
            WHERE attrelid = 'media'::regclass
              AND attname = 'scene_embedding'
              AND NOT attisdropped;

            IF current_type IS NULL THEN
                ALTER TABLE media ADD COLUMN scene_embedding vector(768);
            ELSIF current_type = 'vector(768)' THEN
                NULL;
            ELSIF current_type LIKE 'vector%' THEN
                IF EXISTS (
                    SELECT 1 FROM media
                    WHERE scene_embedding IS NOT NULL
                      AND vector_dims(scene_embedding) <> 768
                ) THEN
                    RAISE EXCEPTION
                        'media.scene_embedding contains vectors that are not 768-dimensional';
                END IF;
                EXECUTE 'DROP INDEX IF EXISTS media_scene_hnsw';
                ALTER TABLE media
                    ALTER COLUMN scene_embedding TYPE vector(768)
                    USING scene_embedding::vector(768);
            ELSIF current_type IN ('real[]', 'double precision[]') THEN
                IF EXISTS (
                    SELECT 1 FROM media
                    WHERE scene_embedding IS NOT NULL
                      AND cardinality(scene_embedding) <> 768
                ) THEN
                    RAISE EXCEPTION
                        'media.scene_embedding contains arrays that are not 768-dimensional';
                END IF;
                EXECUTE 'DROP INDEX IF EXISTS media_scene_hnsw';
                ALTER TABLE media
                    ALTER COLUMN scene_embedding TYPE vector(768)
                    USING scene_embedding::vector(768);
            ELSE
                RAISE EXCEPTION USING MESSAGE =
                    'unsupported media.scene_embedding type: ' || current_type;
            END IF;

            SELECT format_type(atttypid, atttypmod)
            INTO current_type
            FROM pg_attribute
            WHERE attrelid = 'media'::regclass
              AND attname = 'dedup_embedding'
              AND NOT attisdropped;

            IF current_type IS NULL THEN
                ALTER TABLE media ADD COLUMN dedup_embedding vector(512);
            ELSIF current_type = 'vector(512)' THEN
                NULL;
            ELSIF current_type LIKE 'vector%' THEN
                IF EXISTS (
                    SELECT 1 FROM media
                    WHERE dedup_embedding IS NOT NULL
                      AND vector_dims(dedup_embedding) <> 512
                ) THEN
                    RAISE EXCEPTION
                        'media.dedup_embedding contains vectors that are not 512-dimensional';
                END IF;
                EXECUTE 'DROP INDEX IF EXISTS media_dedup_hnsw';
                ALTER TABLE media
                    ALTER COLUMN dedup_embedding TYPE vector(512)
                    USING dedup_embedding::vector(512);
            ELSIF current_type IN ('real[]', 'double precision[]') THEN
                IF EXISTS (
                    SELECT 1 FROM media
                    WHERE dedup_embedding IS NOT NULL
                      AND cardinality(dedup_embedding) <> 512
                ) THEN
                    RAISE EXCEPTION
                        'media.dedup_embedding contains arrays that are not 512-dimensional';
                END IF;
                EXECUTE 'DROP INDEX IF EXISTS media_dedup_hnsw';
                ALTER TABLE media
                    ALTER COLUMN dedup_embedding TYPE vector(512)
                    USING dedup_embedding::vector(512);
            ELSE
                RAISE EXCEPTION USING MESSAGE =
                    'unsupported media.dedup_embedding type: ' || current_type;
            END IF;
        END
        $memoria$;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS media_scene_hnsw
        ON media USING hnsw (scene_embedding vector_cosine_ops)
        WITH (m = 24, ef_construction = 128);
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS media_dedup_hnsw
        ON media USING hnsw (dedup_embedding vector_cosine_ops);
        """
    )

    # Restore the model-keyed scene store skipped by revision 011.  Named
    # guards also repair a table created manually without its required keys.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS scene_embeddings (
            media_id          BIGINT NOT NULL,
            model_key         TEXT NOT NULL,
            embedding         vector NOT NULL,
            dimension         INTEGER NOT NULL,
            runtime_provider  TEXT NOT NULL DEFAULT 'unknown',
            runtime_precision TEXT NOT NULL DEFAULT 'unknown',
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute(
        """
        DO $memoria$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'scene_embeddings'::regclass
                  AND conname = 'scene_embeddings_pkey'
            ) THEN
                ALTER TABLE scene_embeddings
                    ADD CONSTRAINT scene_embeddings_pkey
                    PRIMARY KEY (media_id, model_key);
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'scene_embeddings'::regclass
                  AND conname = 'scene_embeddings_media_id_fkey'
            ) THEN
                ALTER TABLE scene_embeddings
                    ADD CONSTRAINT scene_embeddings_media_id_fkey
                    FOREIGN KEY (media_id) REFERENCES media(id) ON DELETE CASCADE;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'scene_embeddings'::regclass
                  AND conname = 'scene_embeddings_model_key_fkey'
            ) THEN
                ALTER TABLE scene_embeddings
                    ADD CONSTRAINT scene_embeddings_model_key_fkey
                    FOREIGN KEY (model_key)
                    REFERENCES ml_models(model_key) ON DELETE RESTRICT;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'scene_embeddings'::regclass
                  AND conname = 'fk_scene_embedding_model_dimension'
            ) THEN
                ALTER TABLE scene_embeddings
                    ADD CONSTRAINT fk_scene_embedding_model_dimension
                    FOREIGN KEY (model_key, dimension)
                    REFERENCES ml_models(model_key, embedding_dimension)
                    ON DELETE RESTRICT;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'scene_embeddings'::regclass
                  AND conname = 'scene_embeddings_dimension_ck'
            ) THEN
                ALTER TABLE scene_embeddings
                    ADD CONSTRAINT scene_embeddings_dimension_ck
                    CHECK (
                        dimension BETWEEN 1 AND 16000
                        AND vector_dims(embedding) = dimension
                    );
            END IF;
        END
        $memoria$;
        """
    )
    op.execute(
        """
        INSERT INTO scene_embeddings (
            media_id, model_key, embedding, dimension,
            runtime_provider, runtime_precision
        )
        SELECT
            id,
            'scene/open_clip/ViT-L-14/laion2b_s32b_b82k',
            scene_embedding::vector,
            768,
            'legacy',
            'unknown'
        FROM media
        WHERE scene_embedding IS NOT NULL
          AND vector_dims(scene_embedding) = 768
        ON CONFLICT (media_id, model_key) DO NOTHING;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scene_embeddings_model_media
        ON scene_embeddings (model_key, media_id);
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS scene_openclip_vitl14_hnsw
        ON scene_embeddings USING hnsw
        ((embedding::vector(768)) vector_cosine_ops)
        WITH (m = 24, ef_construction = 128)
        WHERE model_key = 'scene/open_clip/ViT-L-14/laion2b_s32b_b82k';
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS scene_openclip_siglip1152_hnsw
        ON scene_embeddings USING hnsw
        ((embedding::vector(1152)) vector_cosine_ops)
        WITH (m = 24, ef_construction = 128)
        WHERE model_key = 'scene/open_clip/ViT-SO400M-14-SigLIP-384/webli';
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS scene_hf_siglip2_1152_hnsw
        ON scene_embeddings USING hnsw
        ((embedding::vector(1152)) vector_cosine_ops)
        WITH (m = 24, ef_construction = 128)
        WHERE model_key = 'scene/hf/google/siglip2-so400m-patch14-384';
        """
    )


def downgrade() -> None:
    # This revision only restores invariants already owned by revisions
    # 004/005/007/011.  Removing them would make an 011 database invalid.
    pass
