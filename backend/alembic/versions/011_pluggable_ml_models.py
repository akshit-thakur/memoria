"""Pluggable ML model catalog, versioned vectors, jobs, and benchmarks.

The legacy ``media.scene_embedding`` and ``face_embeddings.embedding``
columns remain in place so this migration is additive.  Scene vectors move to
an untyped pgvector column keyed by an immutable model identity; typed partial
expression indexes provide HNSW acceleration for the built-in profiles while
custom dimensions retain a correct exact-search fallback.

Revision ID: 011_pluggable_ml_models
Revises: 010_perf_indexes
Create Date: 2026-08-03
"""

from __future__ import annotations

from alembic import op


revision = "011_pluggable_ml_models"
down_revision = "010_perf_indexes"
branch_labels = None
depends_on = None


DEFAULT_SCENE_MODEL_KEY = "scene/open_clip/ViT-L-14/laion2b_s32b_b82k"
OPENCLIP_SIGLIP_MODEL_KEY = (
    "scene/open_clip/ViT-SO400M-14-SigLIP-384/webli"
)
HF_SIGLIP2_MODEL_KEY = "scene/hf/google/siglip2-so400m-patch14-384"
LEGACY_FACE_MODEL_KEY = "face/legacy-unknown-512"
DEFAULT_FACE_MODEL_KEY = "face/insightface-buffalo-l-arcface"
FACENET_MODEL_KEY = "face/facenet-vggface2"


def upgrade() -> None:
    # Immutable, framework-free descriptions of every vector space.  A model
    # key may never be silently repointed at different weights or dimensions.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_models (
            model_key             TEXT PRIMARY KEY,
            kind                  TEXT NOT NULL CHECK (length(btrim(kind)) > 0),
            driver                TEXT NOT NULL CHECK (length(btrim(driver)) > 0),
            model_id              TEXT NOT NULL CHECK (length(btrim(model_id)) > 0),
            weights               TEXT,
            revision              TEXT,
            input_resolution      INTEGER[],
            embedding_dimension   INTEGER,
            default_precision     TEXT NOT NULL DEFAULT 'auto',
            provider_preferences  JSONB NOT NULL DEFAULT '[]'::jsonb,
            distance_metric       TEXT NOT NULL DEFAULT 'cosine',
            normalized            BOOLEAN NOT NULL DEFAULT TRUE,
            index_kind            TEXT NOT NULL DEFAULT 'exact',
            fingerprint           VARCHAR(64) NOT NULL,
            metadata              JSONB NOT NULL DEFAULT '{}'::jsonb,
            is_builtin            BOOLEAN NOT NULL DEFAULT FALSE,
            created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT ml_models_resolution_ck CHECK (
                input_resolution IS NULL OR (
                    cardinality(input_resolution) = 2
                    AND input_resolution[1] > 0
                    AND input_resolution[2] > 0
                )
            ),
            CONSTRAINT ml_models_dimension_ck CHECK (
                embedding_dimension IS NULL
                OR embedding_dimension BETWEEN 1 AND 16000
            ),
            CONSTRAINT ml_models_providers_ck CHECK (
                jsonb_typeof(provider_preferences) = 'array'
            ),
            CONSTRAINT ml_models_metadata_ck CHECK (jsonb_typeof(metadata) = 'object'),
            CONSTRAINT ml_models_distance_ck CHECK (
                distance_metric IN ('cosine', 'inner_product', 'l2')
            ),
            CONSTRAINT ml_models_index_kind_ck CHECK (
                index_kind IN ('hnsw', 'exact', 'none')
            ),
            CONSTRAINT ml_models_fingerprint_ck CHECK (
                fingerprint ~ '^[0-9a-f]{64}$'
            ),
            CONSTRAINT uq_ml_models_key_dimension UNIQUE (
                model_key, embedding_dimension
            )
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ml_models_kind ON ml_models (kind, model_key);"
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'uq_ml_models_key_dimension'
                  AND conrelid = 'ml_models'::regclass
            ) THEN
                ALTER TABLE ml_models
                    ADD CONSTRAINT uq_ml_models_key_dimension
                    UNIQUE (model_key, embedding_dimension);
            END IF;
        END
        $$;
        """
    )

    # Reject identity mutation even from an accidental manual UPDATE.  New
    # metadata requires a new model_key and therefore a new vector namespace.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION memoria_reject_ml_model_update()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW IS DISTINCT FROM OLD THEN
                RAISE EXCEPTION
                    'ml_models rows are immutable; register a new model_key instead';
            END IF;
            RETURN OLD;
        END
        $$;
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_trigger
                WHERE tgname = 'trg_ml_models_immutable'
                  AND tgrelid = 'ml_models'::regclass
            ) THEN
                CREATE TRIGGER trg_ml_models_immutable
                BEFORE UPDATE ON ml_models
                FOR EACH ROW EXECUTE FUNCTION memoria_reject_ml_model_update();
            END IF;
        END
        $$;
        """
    )

    # These fingerprints are produced by ModelSpec.fingerprint over the exact
    # built-in descriptors in backend/ml/scene.py.
    op.execute(
        """
        INSERT INTO ml_models (
            model_key, kind, driver, model_id, weights, revision,
            input_resolution, embedding_dimension, default_precision,
            provider_preferences, distance_metric, normalized, fingerprint,
            metadata, is_builtin, index_kind
        ) VALUES
        (
            'scene/open_clip/ViT-L-14/laion2b_s32b_b82k',
            'scene', 'open_clip', 'ViT-L-14', 'laion2b_s32b_b82k', NULL,
            ARRAY[224, 224], 768, 'auto',
            jsonb_build_array('cuda', 'mps', 'cpu'), 'cosine', TRUE,
            'b9de66a8ccbfdeb67904fc1087a6f9dc90908904f3410eaa8d7aa3631c0d0421',
            jsonb_build_object(
                'family', 'clip',
                'supports_text', TRUE,
                'profile', 'openclip-vit-l-14'
            ),
            TRUE, 'hnsw'
        ),
        (
            'scene/open_clip/ViT-SO400M-14-SigLIP-384/webli',
            'scene', 'open_clip', 'ViT-SO400M-14-SigLIP-384', 'webli', NULL,
            ARRAY[384, 384], 1152, 'auto',
            jsonb_build_array('cuda', 'mps', 'cpu'), 'cosine', TRUE,
            '931fbf11b946fc0f972ca203aad4de2dd839ec600b24de6ba534999f2df4ed3f',
            jsonb_build_object(
                'family', 'siglip',
                'supports_text', TRUE,
                'profile', 'openclip-siglip-so400m-384'
            ),
            TRUE, 'hnsw'
        ),
        (
            'scene/hf/google/siglip2-so400m-patch14-384',
            'scene', 'hf', 'google/siglip2-so400m-patch14-384', NULL, NULL,
            ARRAY[384, 384], 1152, 'auto',
            jsonb_build_array('cuda', 'mps', 'cpu'), 'cosine', TRUE,
            '28c1fe5664d075be28298b70cada2c5162d0cf29bbf0943c4c0ad82513556e53',
            jsonb_build_object(
                'family', 'siglip2',
                'supports_text', TRUE,
                'profile', 'hf-siglip2-so400m-384'
            ),
            TRUE, 'hnsw'
        ),
        (
            'face/legacy-unknown-512',
            'face-embedding', 'legacy', 'unknown', NULL, NULL,
            NULL, 512, 'unknown',
            jsonb_build_array('unknown'), 'cosine', TRUE,
            'd9632d20cf83207736da4580dc9cf7ce51b9f32af9f3bd6b3695d6c924d5c677',
            jsonb_build_object(
                'legacy', TRUE,
                'note', 'Pre-Phase-2 rows may contain ArcFace or FaceNet vectors'
            ),
            TRUE, 'exact'
        ),
        (
            'face/insightface-buffalo-l-arcface',
            'face-embedding', 'insightface', 'buffalo_l/arcface', NULL, NULL,
            ARRAY[112, 112], 512, 'fp32',
            jsonb_build_array('tensorrt', 'onnx-cuda', 'cpu'), 'cosine', TRUE,
            '5657d44f1b2ba0d7dcad1ee9304f1165265ebb298a11ac547f426cba8be11944',
            jsonb_build_object(),
            TRUE, 'hnsw'
        ),
        (
            'face/facenet-vggface2',
            'face-embedding', 'facenet-pytorch',
            'inception-resnet-v1/vggface2', NULL, NULL,
            ARRAY[160, 160], 512, 'fp32',
            jsonb_build_array('cuda', 'cpu'), 'cosine', TRUE,
            '859f4a6b2d63c1515f41779d414352b20d8496927428ebedeac3cd61fc447a09',
            jsonb_build_object(),
            TRUE, 'hnsw'
        )
        ON CONFLICT (model_key) DO NOTHING;
        """
    )

    # Desired is the environment-selected write target; active is the fully
    # indexed model currently serving search.  They deliberately diverge while
    # a background reindex is running.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_model_state (
            kind               TEXT PRIMARY KEY,
            desired_model_key  TEXT REFERENCES ml_models(model_key) ON DELETE RESTRICT,
            active_model_key   TEXT REFERENCES ml_models(model_key) ON DELETE RESTRICT,
            status             TEXT NOT NULL DEFAULT 'ready',
            generation         BIGINT NOT NULL DEFAULT 1,
            last_error         TEXT,
            updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT ml_model_state_status_ck CHECK (
                status IN ('ready', 'building', 'indexing', 'failed', 'disabled')
            ),
            CONSTRAINT ml_model_state_generation_ck CHECK (generation >= 0)
        );
        """
    )
    op.execute(
        """
        INSERT INTO ml_model_state (
            kind, desired_model_key, active_model_key, status, generation
        ) VALUES
            (
                'scene',
                'scene/open_clip/ViT-L-14/laion2b_s32b_b82k',
                'scene/open_clip/ViT-L-14/laion2b_s32b_b82k',
                'ready', 1
            ),
            (
                'face-embedding',
                'face/insightface-buffalo-l-arcface',
                CASE
                    WHEN EXISTS (SELECT 1 FROM face_embeddings)
                    THEN 'face/legacy-unknown-512'
                    ELSE 'face/insightface-buffalo-l-arcface'
                END,
                CASE
                    WHEN EXISTS (SELECT 1 FROM face_embeddings)
                    THEN 'building'
                    ELSE 'ready'
                END,
                1
            )
        ON CONFLICT (kind) DO NOTHING;
        """
    )

    # The untyped vector column accepts different dimensions in different
    # model namespaces. vector_dims enforces truthful per-row metadata.
    op.execute(
        """
        DO $memoria$
        DECLARE
            has_pgvector BOOLEAN;
        BEGIN
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            ) INTO has_pgvector;

            IF NOT has_pgvector THEN
                RAISE NOTICE
                    'pgvector not installed - skipping scene_embeddings storage';
                RETURN;
            END IF;

            EXECUTE $ddl$
                CREATE TABLE IF NOT EXISTS scene_embeddings (
                    media_id          BIGINT NOT NULL
                                      REFERENCES media(id) ON DELETE CASCADE,
                    model_key         TEXT NOT NULL
                                      REFERENCES ml_models(model_key) ON DELETE RESTRICT,
                    embedding         vector NOT NULL,
                    dimension         INTEGER NOT NULL,
                    runtime_provider  TEXT NOT NULL DEFAULT 'unknown',
                    runtime_precision TEXT NOT NULL DEFAULT 'unknown',
                    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (media_id, model_key),
                    CONSTRAINT fk_scene_embedding_model_dimension
                        FOREIGN KEY (model_key, dimension)
                        REFERENCES ml_models(model_key, embedding_dimension)
                        ON DELETE RESTRICT,
                    CONSTRAINT scene_embeddings_dimension_ck CHECK (
                        dimension BETWEEN 1 AND 16000
                        AND vector_dims(embedding) = dimension
                    )
                )
            $ddl$;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'fk_scene_embedding_model_dimension'
                  AND conrelid = 'scene_embeddings'::regclass
            ) THEN
                EXECUTE 'ALTER TABLE scene_embeddings '
                     || 'ADD CONSTRAINT fk_scene_embedding_model_dimension '
                     || 'FOREIGN KEY (model_key, dimension) '
                     || 'REFERENCES ml_models(model_key, embedding_dimension) '
                     || 'ON DELETE RESTRICT';
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'scene_embeddings_dimension_ck'
                  AND conrelid = 'scene_embeddings'::regclass
            ) THEN
                EXECUTE 'ALTER TABLE scene_embeddings '
                     || 'ADD CONSTRAINT scene_embeddings_dimension_ck '
                     || 'CHECK (dimension BETWEEN 1 AND 16000 '
                     || 'AND vector_dims(embedding) = dimension)';
            END IF;

            EXECUTE 'CREATE INDEX IF NOT EXISTS idx_scene_embeddings_model_media '
                 || 'ON scene_embeddings (model_key, media_id)';

            -- Migration 005 may have been skipped on installations that did
            -- not yet have pgvector. Copy only truthful 768-d legacy rows.
            IF EXISTS (
                SELECT 1 FROM pg_attribute
                WHERE attrelid = 'media'::regclass
                  AND attname = 'scene_embedding'
                  AND NOT attisdropped
            ) THEN
                EXECUTE $copy$
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
                    ON CONFLICT (media_id, model_key) DO NOTHING
                $copy$;
            END IF;

            -- Typed expression + literal partial predicate must match the
            -- query expression exactly for PostgreSQL to select these indexes.
            -- Build after the legacy bulk copy so upgrades do not maintain an
            -- HNSW graph one migrated row at a time.
            EXECUTE 'CREATE INDEX IF NOT EXISTS scene_openclip_vitl14_hnsw '
                 || 'ON scene_embeddings USING hnsw '
                 || '((embedding::vector(768)) vector_cosine_ops) '
                 || 'WITH (m = 24, ef_construction = 128) '
                 || 'WHERE model_key = '
                 || quote_literal('scene/open_clip/ViT-L-14/laion2b_s32b_b82k');
            EXECUTE 'CREATE INDEX IF NOT EXISTS scene_openclip_siglip1152_hnsw '
                 || 'ON scene_embeddings USING hnsw '
                 || '((embedding::vector(1152)) vector_cosine_ops) '
                 || 'WITH (m = 24, ef_construction = 128) '
                 || 'WHERE model_key = '
                 || quote_literal('scene/open_clip/ViT-SO400M-14-SigLIP-384/webli');
            EXECUTE 'CREATE INDEX IF NOT EXISTS scene_hf_siglip2_1152_hnsw '
                 || 'ON scene_embeddings USING hnsw '
                 || '((embedding::vector(1152)) vector_cosine_ops) '
                 || 'WITH (m = 24, ef_construction = 128) '
                 || 'WHERE model_key = '
                 || quote_literal('scene/hf/google/siglip2-so400m-patch14-384');
        END
        $memoria$;
        """
    )

    # Persistent, resumable state for chunked background migrations.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_reindex_jobs (
            id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            kind              TEXT NOT NULL,
            model_key         TEXT NOT NULL
                              REFERENCES ml_models(model_key) ON DELETE RESTRICT,
            status            TEXT NOT NULL DEFAULT 'queued',
            cursor_media_id   BIGINT NOT NULL DEFAULT 0,
            total_items       BIGINT NOT NULL DEFAULT 0,
            processed_items   BIGINT NOT NULL DEFAULT 0,
            failed_items      BIGINT NOT NULL DEFAULT 0,
            skipped_items     BIGINT NOT NULL DEFAULT 0,
            force_reindex     BOOLEAN NOT NULL DEFAULT FALSE,
            batch_size        INTEGER,
            metrics           JSONB NOT NULL DEFAULT '{}'::jsonb,
            error             TEXT,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            requested_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            started_at        TIMESTAMPTZ,
            finished_at       TIMESTAMPTZ,
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT ml_reindex_jobs_status_ck CHECK (
                status IN (
                    'queued', 'running', 'indexing', 'cancel_requested',
                    'cancelled', 'completed', 'failed'
                )
            ),
            CONSTRAINT ml_reindex_jobs_counts_ck CHECK (
                cursor_media_id >= 0
                AND total_items >= 0
                AND processed_items >= 0
                AND failed_items >= 0
                AND skipped_items >= 0
                AND (batch_size IS NULL OR batch_size > 0)
            ),
            CONSTRAINT ml_reindex_jobs_metrics_ck CHECK (jsonb_typeof(metrics) = 'object')
        );
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_reindex_jobs_active
        ON ml_reindex_jobs (kind, model_key)
        WHERE status IN ('queued', 'running', 'indexing', 'cancel_requested');
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ml_reindex_jobs_recent
        ON ml_reindex_jobs (kind, requested_at DESC);
        """
    )

    # Per-media completion is separate from embeddings.  In particular, a
    # successful face pass that found zero faces must remain completed.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_processing_state (
            media_id    BIGINT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
            kind        TEXT NOT NULL,
            model_key   TEXT NOT NULL REFERENCES ml_models(model_key) ON DELETE RESTRICT,
            status      TEXT NOT NULL,
            item_count  INTEGER NOT NULL DEFAULT 0,
            duration_ms DOUBLE PRECISION,
            error       TEXT,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (media_id, kind, model_key),
            CONSTRAINT ml_processing_state_status_ck CHECK (
                status IN ('completed', 'failed')
            ),
            CONSTRAINT ml_processing_state_values_ck CHECK (
                item_count >= 0 AND (duration_ms IS NULL OR duration_ms >= 0)
            )
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ml_processing_state_lookup
        ON ml_processing_state (kind, model_key, status, media_id);
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_model_benchmarks (
            id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            model_key           TEXT NOT NULL
                                REFERENCES ml_models(model_key) ON DELETE RESTRICT,
            reindex_job_id      UUID REFERENCES ml_reindex_jobs(id) ON DELETE SET NULL,
            benchmark_type      TEXT NOT NULL DEFAULT 'scene',
            sample_count        INTEGER NOT NULL DEFAULT 0,
            batch_size          INTEGER,
            duration_ms         DOUBLE PRECISION NOT NULL DEFAULT 0,
            items_per_second    DOUBLE PRECISION NOT NULL DEFAULT 0,
            query_count         INTEGER NOT NULL DEFAULT 0,
            query_latency_ms    JSONB NOT NULL DEFAULT '{}'::jsonb,
            peak_vram_mb        DOUBLE PRECISION,
            runtime_provider    TEXT NOT NULL DEFAULT 'unknown',
            runtime_precision   TEXT NOT NULL DEFAULT 'unknown',
            metrics             JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT ml_model_benchmarks_values_ck CHECK (
                sample_count >= 0
                AND query_count >= 0
                AND duration_ms >= 0
                AND items_per_second >= 0
                AND (batch_size IS NULL OR batch_size > 0)
                AND (peak_vram_mb IS NULL OR peak_vram_mb >= 0)
            ),
            CONSTRAINT ml_model_benchmarks_query_ck CHECK (
                jsonb_typeof(query_latency_ms) = 'object'
            ),
            CONSTRAINT ml_model_benchmarks_metrics_ck CHECK (
                jsonb_typeof(metrics) = 'object'
            )
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ml_model_benchmarks_recent
        ON ml_model_benchmarks (model_key, created_at DESC);
        """
    )

    # Additive face metadata. Existing rows predate explicit identity fields
    # and may contain either ArcFace or the FaceNet fallback, so keep them in a
    # separate legacy namespace rather than claiming a provenance we cannot know.
    op.execute(
        """
        ALTER TABLE face_embeddings
            ADD COLUMN IF NOT EXISTS embedding_model_key TEXT,
            ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER,
            ADD COLUMN IF NOT EXISTS detection_pipeline_key TEXT,
            ADD COLUMN IF NOT EXISTS detector_models JSONB NOT NULL DEFAULT '[]'::jsonb,
            ADD COLUMN IF NOT EXISTS runtime_provider TEXT,
            ADD COLUMN IF NOT EXISTS runtime_precision TEXT;
        """
    )
    op.execute(
        """
        UPDATE face_embeddings
        SET embedding_model_key = 'face/legacy-unknown-512'
        WHERE embedding_model_key IS NULL;
        """
    )
    op.execute(
        """
        DO $$
        DECLARE
            embedding_type TEXT;
        BEGIN
            SELECT format_type(atttypid, atttypmod)
            INTO embedding_type
            FROM pg_attribute
            WHERE attrelid = 'face_embeddings'::regclass
              AND attname = 'embedding'
              AND NOT attisdropped;

            IF embedding_type LIKE 'vector(%' THEN
                EXECUTE 'UPDATE face_embeddings '
                     || 'SET embedding_dimension = vector_dims(embedding) '
                     || 'WHERE embedding_dimension IS NULL';
            ELSE
                UPDATE face_embeddings
                SET embedding_dimension = cardinality(embedding)
                WHERE embedding_dimension IS NULL;
            END IF;
        END
        $$;
        """
    )
    op.execute(
        """
        UPDATE face_embeddings
        SET detector_models = CASE
                WHEN detector_models = '[]'::jsonb
                THEN jsonb_build_array(COALESCE(detection_model, 'unknown'))
                ELSE detector_models
            END,
            detection_pipeline_key = COALESCE(
                detection_pipeline_key,
                'face/ensemble-v1' || chr(58) || 'mtcnn+yunet+insightface'
            ),
            runtime_provider = COALESCE(runtime_provider, 'unknown'),
            runtime_precision = COALESCE(runtime_precision, 'unknown');
        """
    )
    op.execute(
        """
        ALTER TABLE face_embeddings
            ALTER COLUMN embedding_model_key SET NOT NULL,
            ALTER COLUMN embedding_dimension SET NOT NULL,
            ALTER COLUMN detection_pipeline_key SET NOT NULL,
            ALTER COLUMN runtime_provider SET NOT NULL,
            ALTER COLUMN runtime_precision SET NOT NULL;
        """
    )
    op.execute(
        """
        DO $$
        DECLARE
            embedding_type TEXT;
        BEGIN
            BEGIN
                ALTER TABLE face_embeddings
                    ADD CONSTRAINT fk_face_embedding_model_dimension
                    FOREIGN KEY (embedding_model_key, embedding_dimension)
                    REFERENCES ml_models(model_key, embedding_dimension)
                    ON DELETE RESTRICT;
            EXCEPTION WHEN duplicate_object THEN
                NULL;
            END;

            BEGIN
                ALTER TABLE face_embeddings
                    ADD CONSTRAINT face_embedding_dimension_ck
                    CHECK (
                        embedding_dimension IS NULL
                        OR embedding_dimension BETWEEN 1 AND 16000
                    );
            EXCEPTION WHEN duplicate_object THEN
                NULL;
            END;

            BEGIN
                ALTER TABLE face_embeddings
                    ADD CONSTRAINT face_detector_models_ck
                    CHECK (jsonb_typeof(detector_models) = 'array');
            EXCEPTION WHEN duplicate_object THEN
                NULL;
            END;

            SELECT format_type(atttypid, atttypmod)
            INTO embedding_type
            FROM pg_attribute
            WHERE attrelid = 'face_embeddings'::regclass
              AND attname = 'embedding'
              AND NOT attisdropped;

            BEGIN
                IF embedding_type LIKE 'vector(%' THEN
                    EXECUTE 'ALTER TABLE face_embeddings '
                         || 'ADD CONSTRAINT face_embedding_vector_dimension_ck '
                         || 'CHECK (vector_dims(embedding) = embedding_dimension)';
                ELSE
                    ALTER TABLE face_embeddings
                        ADD CONSTRAINT face_embedding_vector_dimension_ck
                        CHECK (cardinality(embedding) = embedding_dimension);
                END IF;
            EXCEPTION WHEN duplicate_object THEN
                NULL;
            END;
        END
        $$;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_face_embedding_model_key
        ON face_embeddings (embedding_model_key, media_id);
        """
    )
    # Existing rows prove that face processing completed for their media/model
    # namespace. Seed model-scoped completion so restart coverage checks can
    # distinguish them from legacy zero-face media, which must be reprocessed
    # because no row exists from which to infer completion.
    op.execute(
        """
        INSERT INTO ml_processing_state (
            media_id, kind, model_key, status, item_count,
            duration_ms, error, updated_at
        )
        SELECT
            media_id,
            'face-embedding',
            embedding_model_key,
            'completed',
            COUNT(*)::INTEGER,
            NULL,
            NULL,
            NOW()
        FROM face_embeddings
        GROUP BY media_id, embedding_model_key
        ON CONFLICT (media_id, kind, model_key) DO NOTHING;
        """
    )
    op.execute(
        """
        DO $$
        DECLARE
            embedding_type TEXT;
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            ) THEN
                RETURN;
            END IF;

            SELECT format_type(atttypid, atttypmod)
            INTO embedding_type
            FROM pg_attribute
            WHERE attrelid = 'face_embeddings'::regclass
              AND attname = 'embedding'
              AND NOT attisdropped;

            IF embedding_type LIKE 'vector(%' THEN
                EXECUTE 'CREATE INDEX IF NOT EXISTS face_arcface_512_hnsw '
                     || 'ON face_embeddings USING hnsw '
                     || '(embedding vector_cosine_ops) '
                     || 'WITH (m = 24, ef_construction = 128) '
                     || 'WHERE embedding_model_key = '
                     || quote_literal('face/insightface-buffalo-l-arcface');
                EXECUTE 'CREATE INDEX IF NOT EXISTS face_facenet_512_hnsw '
                     || 'ON face_embeddings USING hnsw '
                     || '(embedding vector_cosine_ops) '
                     || 'WITH (m = 24, ef_construction = 128) '
                     || 'WHERE embedding_model_key = '
                     || quote_literal('face/facenet-vggface2');
                EXECUTE 'CREATE INDEX IF NOT EXISTS face_legacy_512_hnsw '
                     || 'ON face_embeddings USING hnsw '
                     || '(embedding vector_cosine_ops) '
                     || 'WITH (m = 24, ef_construction = 128) '
                     || 'WHERE embedding_model_key = '
                     || quote_literal('face/legacy-unknown-512');
            END IF;
        END
        $$;
        """
    )
    # The pre-Phase-2 global ANN index mixes mutually incompatible face
    # spaces. Supported models now use the literal partial indexes above;
    # unknown future identities deliberately fall back to exact search.
    op.execute("DROP INDEX IF EXISTS face_embeddings_hnsw;")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS face_legacy_512_hnsw;")
    op.execute("DROP INDEX IF EXISTS face_facenet_512_hnsw;")
    op.execute("DROP INDEX IF EXISTS face_arcface_512_hnsw;")
    op.execute(
        """
        DO $$
        DECLARE
            embedding_type TEXT;
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            ) THEN
                RETURN;
            END IF;
            SELECT format_type(atttypid, atttypmod)
            INTO embedding_type
            FROM pg_attribute
            WHERE attrelid = 'face_embeddings'::regclass
              AND attname = 'embedding'
              AND NOT attisdropped;
            IF embedding_type LIKE 'vector(%' THEN
                EXECUTE 'CREATE INDEX IF NOT EXISTS face_embeddings_hnsw '
                     || 'ON face_embeddings USING hnsw '
                     || '(embedding vector_cosine_ops) '
                     || 'WITH (m = 24, ef_construction = 128)';
            END IF;
        END
        $$;
        """
    )
    op.execute("DROP INDEX IF EXISTS idx_face_embedding_model_key;")
    op.execute(
        "ALTER TABLE face_embeddings DROP CONSTRAINT IF EXISTS face_detector_models_ck;"
    )
    op.execute(
        "ALTER TABLE face_embeddings DROP CONSTRAINT IF EXISTS face_embedding_dimension_ck;"
    )
    op.execute(
        "ALTER TABLE face_embeddings DROP CONSTRAINT IF EXISTS face_embedding_vector_dimension_ck;"
    )
    op.execute(
        "ALTER TABLE face_embeddings DROP CONSTRAINT IF EXISTS fk_face_embedding_model_dimension;"
    )
    op.execute("ALTER TABLE face_embeddings DROP COLUMN IF EXISTS runtime_precision;")
    op.execute("ALTER TABLE face_embeddings DROP COLUMN IF EXISTS runtime_provider;")
    op.execute("ALTER TABLE face_embeddings DROP COLUMN IF EXISTS detector_models;")
    op.execute("ALTER TABLE face_embeddings DROP COLUMN IF EXISTS detection_pipeline_key;")
    op.execute("ALTER TABLE face_embeddings DROP COLUMN IF EXISTS embedding_dimension;")
    op.execute("ALTER TABLE face_embeddings DROP COLUMN IF EXISTS embedding_model_key;")

    op.execute("DROP TABLE IF EXISTS ml_model_benchmarks;")
    op.execute("DROP TABLE IF EXISTS ml_processing_state;")
    op.execute("DROP TABLE IF EXISTS ml_reindex_jobs;")
    op.execute("DROP INDEX IF EXISTS scene_hf_siglip2_1152_hnsw;")
    op.execute("DROP INDEX IF EXISTS scene_openclip_siglip1152_hnsw;")
    op.execute("DROP INDEX IF EXISTS scene_openclip_vitl14_hnsw;")
    op.execute("DROP TABLE IF EXISTS scene_embeddings;")
    op.execute("DROP TABLE IF EXISTS ml_model_state;")
    op.execute("DROP TRIGGER IF EXISTS trg_ml_models_immutable ON ml_models;")
    op.execute("DROP FUNCTION IF EXISTS memoria_reject_ml_model_update();")
    op.execute("DROP TABLE IF EXISTS ml_models;")
