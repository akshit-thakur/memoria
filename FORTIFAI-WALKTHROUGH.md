# Memoria: technical evidence walkthrough for Fortifai

Repository inspection date: 20 October 2026.

This is an implementation evidence report, not a claim of production deployment, personal authorship, or measured business impact. Paths are repository-relative; named symbols identify the supporting implementation. **Fact** means supported by inspected code or configuration, **inference** means a consequence or interpretation of that design, and **unknown** means the repository does not establish it. Library-provided models and algorithms are distinguished from project integration code.

Scope: backend source, ML drivers, migrations, tests, frontend integration, dependency manifests, deployment files, recovery utilities, README, and benchmark harness. No dependencies were installed, models downloaded, application started, pipelines executed, or tests run for this report. Test descriptions below describe checked-in assertions, not newly observed passing results. No labeled image evaluation corpus, saved benchmark result, or production run artifact was found in the inspected repository. Runtime databases and user model caches were not inspected. The private `.env` was not used as evidence of deployment.

## 1. Project Overview

Memoria is a local-first photo-library application. It discovers files, extracts metadata and visual features, stores those features, and exposes photo browsing, semantic text/image search, near-duplicate groups, face groups, blur review, and EXIF-derived locations. It also offers image edits and user-triggered cleanup. This solves photo organization and retrieval, not enterprise risk scoring. Evidence: `backend/watcher.py::incremental_scan`, `backend/workers/tasks.py::process_media`, `backend/routers.py` route handlers, and `frontend/src/components/{SearchView,DuplicateGroups,FaceGrid,BlurryList,LocationMap}.tsx`.

The implementation is substantially beyond a single notebook: it includes a Python HTTP API, persistent task queue, SQL migrations, multiple pretrained model adapters, a web frontend, recovery scripts, regression tests, and native/container startup paths. The defensible maturity description is **an implemented local application with regression-test source and deployment configuration; end-to-end operation and production deployment are unverified in this review**. Having tests or Docker files is not evidence that this checkout passes a full release gate.

Implemented capabilities and boundaries:

| Capability | Evidence and qualification |
|---|---|
| Image ingestion and configurable stages | `workers/tasks.py::process_media`, `process_media_batch`; `processing_config.py::normalize`. Paths in this table are under `backend/` unless stated otherwise. |
| Text-to-image and image-to-image retrieval | `routers.py::search_text`, `search_similar`; image/text embedding similarity, not generated answers. |
| Near-duplicate grouping | `dedup_engine.py` produces SSCD vectors; `routers.py::_refresh_duplicate_groups` constructs connected components. pHash is separately calculated, but is not the current grouping algorithm. |
| Face detection, embeddings, clustering and manual labels | `face_detection/face_pipeline.py::FaceDetector`; `workers/tasks.py::cluster_faces`; `routers.py::faces_update_label`, `faces_merge`, `faces_hide_cluster`. No named-person classifier is trained. |
| Blur review and basic edits | `scripts/blur_detection/blur_detect.py::score_image_detailed`; `routers.py::blurry_sharpen`, `editor_crop`, `editor_rotate`, `editor_filter`. Sharpening/filtering is image processing, not a learned restoration model. |
| Video discovery | `constants.py::VIDEO_EXTENSIONS` and `workers/tasks.py::process_media`: videos are marked skipped/synced without image inference. Keyframe extraction is explicitly future work. Do not claim implemented video understanding. |
| Extensible feature contracts | `ml/base.py::FeatureExtractorEngine` is an interface; its references to future object descriptors do not establish an object-detection pipeline. |

No generative LLM, prompt chain, autonomous tool-using agent, financial/risk anomaly detector, time-series model, supervised training loop, or fine-tuning pipeline was found.

## 2. Architecture and Data Flow

### Components

| Component | Responsibility and evidence |
|---|---|
| Discovery | Watchdog events plus filesystem crawling; SQLite scan index, exclusions, debounce/TTL claims, enqueue retries and stale-file reconciliation. `backend/watcher.py::MediaEventHandler`, `incremental_scan`, `_dispatch_scan_batch`, `_reconcile_stale_index_paths`. |
| Queue and orchestration | `SqliteHuey` persists work; task lifecycle signals record outcomes and interrupted work can be requeued. `backend/workers/__init__.py`; task definitions in `backend/workers/tasks.py`. |
| Image decoding | Shared Pillow/OpenCV paths, HEIC support and lazy RAW decoding via rawpy. `backend/image_utils.py::decode_pil`, `decode_cv2_bgr`, `make_thumbnail`, `make_preview`. |
| ML adapters | Framework-light specifications/registry; lazy scene and face engines; separate SSCD singleton. `backend/ml/{base,registry,providers,scene,face}.py`, `backend/face_engine.py`, `backend/dedup_engine.py`. |
| Persistence | PostgreSQL/pgvector for media, embeddings, groups, processing state, model descriptors, reindex jobs and benchmark records; JSON sidecars for recovery; disk image caches. `backend/model_store.py`, `backend/sidecar.py`, `backend/alembic/versions/001_initial_schema.py` through `012_vector_schema_repair.py`. |
| API | FastAPI/asyncpg for reads and mutations; blocking inference offloaded where appropriate. `backend/main.py::lifespan`, `backend/routers.py::search_text`. Workers use synchronous psycopg. |
| UI | Next.js/React/TypeScript, React Query, virtualized grids and MapLibre dependencies. `frontend/package.json`, `frontend/src/components/PhotoGrid.tsx`, `frontend/src/lib/api.ts`, `frontend/src/hooks/useMedia.ts`. |
| Diagnostics | Per-process timing/counter snapshots, persistent throughput records, heartbeat and API status. `backend/metrics.py`, `backend/throughput.py`, `backend/workers/heartbeat.py`, `backend/main.py::get_metrics`, `get_progress`. |

### Input-to-output flow

1. A crawl, filesystem event, or folder-scan request discovers an eligible path. Exclusions and scan bookkeeping determine dispatch; accepted paths enter Huey's SQLite queue (`backend/watcher.py::incremental_scan`, `_dispatch_paths`; `backend/routers.py::library_scan_folder`). The browser folder picker sends a bounded relative-path sample and total count to resolve a server-visible directory; it is not uploading all image bytes (`frontend/src/lib/pickDirectory.ts`, `backend/library_paths.py::validate_picked_scan_request`).
2. The worker checks sidecar state and processing configuration. New images proceed through enabled thumbnail, EXIF/GPS, pHash, blur, face, scene and SSCD stages; stat and metadata synchronization are always on (`backend/workers/tasks.py::process_media`; `backend/processing_config.py::FEATURE_KEYS`, `PRESETS`).
3. Single-image stages pass a dictionary payload. Batch processing uses `_BatchEntry`, CPU work, batched model calls and metadata writes. `_run_batch_pipelined` overlaps GPU work on half A with CPU work on half B, then synchronization of A with GPU work on B. This is deterministic task orchestration, not an AI agent.
4. `sync_metadata` writes a pending sidecar, transactionally upserts database content, then marks the sidecar synced. Scene vectors go to a model-keyed table; dedup vectors remain on `media`; faces retain embedding-model and detector-pipeline provenance. See `backend/workers/tasks.py::_persist_scene_embedding`, `sync_metadata`, `_sync_metadata_batch_inner`.
5. Retrieval uses the relevant vector space and hydrates SQL media records. `search_text` loads the active scene model, encodes the query in a background thread, then queries pgvector; `search_similar` starts from an active stored image vector. `_hydrate_scene_neighbors` preserves vector-ranking order. Text encoding defaults to CPU in the API process via `MEMORIA_API_ML_DEVICE`, independently of the ingestion worker.
6. Duplicate and person grouping are separate downstream operations, not outputs of the scene encoder (`backend/routers.py::_refresh_duplicate_groups`; `backend/workers/tasks.py::cluster_faces`).

### Model-versioned storage

`backend/model_store.py` separates desired and active models. Ingestion can populate a desired model namespace while search continues using the active namespace. `scene_embeddings` supports model-specific dimensions; built-in partial HNSW indexes cover 768- and 1152-dimensional scene spaces. Custom profiles use exact queries unless a supported index plan exists. Face spaces remain 512-dimensional but ArcFace and FaceNet have distinct keys: equal dimension does not imply compatible geometry.

`ModelSpec.fingerprint` hashes a canonical model/configuration descriptor. `normalize_model_descriptor` recomputes that identity and rejects mismatched fingerprints. This is configuration identity, **not a cryptographic checksum of downloaded weight bytes**. Evidence: `backend/ml/base.py::ModelSpec`, `backend/model_store.py::normalize_model_descriptor`, `scene_knn_plan`; migration `011_pluggable_ml_models.py`.

## 3. Hands-on AI/ML Implementation

### 3.1 Scene and text embeddings

Problem: map images and natural-language queries into a shared vector space for semantic retrieval.

| Built-in profile | Weights/model source | Input and vector dimension |
|---|---|---|
| Default OpenCLIP | `ViT-L-14`, `laion2b_s32b_b82k` | 224×224; 768 dimensions |
| OpenCLIP SigLIP | `ViT-SO400M-14-SigLIP-384`, `webli` | 384×384; 1152 dimensions |
| Hugging Face SigLIP2 | `google/siglip2-so400m-patch14-384` | 384×384; 1152 dimensions |

Evidence: `backend/ml/scene.py::BUILTIN_SCENE_SPECS`, `DEFAULT_SCENE_MODEL_KEY`, `OpenClipSceneEngine`, `HuggingFaceSiglipSceneEngine`.

These are **pretrained models used for inference only**. Driver code loads weights through OpenCLIP or Transformers, selects evaluation mode and performs inference without training gradients. Image preprocessing/tokenization is delegated to the selected model's library transforms/processors. Worker input commonly reuses a cached thumbnail, otherwise decodes the original with `max_dim=1024`; the driver performs final model preprocessing (`backend/workers/tasks.py::compute_scene_embedding`; scene driver `_load_state`, `encode_images`, `encode_text`). This bounds intermediate work but can discard image detail; no comparison against full-resolution retrieval quality is supplied.

Features are normalized and validated for expected dimension, batch count and finite values (`backend/ml/scene.py::_normalise_features`, `_LazyTorchSceneEngine._vectors`; `backend/ml/base.py::validate_embedding`, `validate_embeddings`). Provider/precision settings select CPU, CUDA or MPS as supported by the driver. Accelerator OOM handling includes smaller batch retries and CPU fallback paths (`backend/ml/scene.py::_switch_to_cpu`, `encode_images`; `backend/ml/providers.py::resolve_execution`). Requested acceleration is not proof of actual acceleration; runtime provider/precision is recorded.

Queries use cosine distance for built-ins. `scene_knn_plan` supports metric-specific operators and exact custom-model fallback; limits are bounded, and custom keys remain bound SQL parameters (`backend/model_store.py::query_scene_neighbors_async`). No learned reranker, query expansion, answer generation, retrieval-grounded LLM, or relevance calibration is implemented.

### 3.2 Face detection consensus and embeddings

Problem: find plausible faces and generate comparable vectors for person grouping. The default pipeline combines pretrained MTCNN, OpenCV YuNet and InsightFace `buffalo_l`, with InsightFace ArcFace embeddings. FaceNet's pretrained VGGFace2 InceptionResnetV1 is an alternative embedder; `none` disables embeddings. Evidence: `backend/ml/face.py::FaceEngineConfig`; `backend/face_detection/face_pipeline.py::_MTCNNDetector`, `_YuNetDetector`, `_InsightFaceDetector`, `_FaceNetEmbedder`, `FaceDetector`.

Concrete rules:

- Per-detector confidence thresholds are MTCNN 0.93, YuNet 0.80 and InsightFace 0.50. MTCNN additionally uses cascade thresholds `[0.85, 0.90, 0.95]` and `min_face_size=60`.
- Shared geometry checks require width/height at least 30 pixels and aspect ratio between 0.5 and 1.5.
- `_merge_detections` greedily groups overlapping detections from distinct models at IoU ≥0.3. At least two distinct detectors must agree; boxes are averaged and confidence is the maximum detector score. It is not a calibrated ensemble probability or a learned fusion layer.
- `FaceDetector.detect` raises when fewer than the required number of detectors successfully execute. Successful detector execution and agreement on a particular face are separate conditions.
- ArcFace embeddings come from the InsightFace adapter. FaceNet instead embeds cropped/resized RGB faces at 160×160 and replaces InsightFace vectors. Returned embeddings are checked for finite values, 512 dimensions and positive norm, then normalized. A consensus detection can lack an embedding and consequently not become a persisted face-vector row.
- Per-driver locks protect shared detectors; configured face batch concurrency defaults to 2. Hardware support is driver-specific, not uniform. `resolve_face_execution` and `reconcile_face_onnx_execution` distinguish requested plans from loaded ONNX session provider chains; non-TensorRT face execution reports fp32.

All recognition/detection networks are imported pretrained. The custom work is adapter integration, consensus, validation, lifecycle management and provenance. No face-network training or fine-tuning is evidenced. Strict consensus plausibly reduces false positives while losing some true faces; this is a **tradeoff inference**, not a measured precision/recall improvement.

### 3.3 Person clustering

`backend/workers/tasks.py::cluster_faces` selects unclustered vectors from the active face namespace, normalizes them, and builds a pgvector neighbor graph with **k=20**, cosine distance **≤0.5**. It symmetrizes a SciPy CSR matrix and calls scikit-learn DBSCAN with `metric='precomputed'`, **eps=0.5**, **min_samples=2**. A second HDBSCAN pass uses Euclidean distance on normalized vectors with **min_cluster_size=3**. Implementations are in `backend/face_detection/face_cluster.py::cluster_dbscan_precomputed`, `cluster_hdbscan`, `cross_verify`.

This is **unsupervised clustering fitted to existing pretrained embeddings**. It is legitimate applied ML, but not training the embedding network. DBSCAN noise (`-1`) remains unclustered; it is not a risk anomaly score. Hidden-cluster absorption routes nearby new faces into hidden groups. Ordinary clustering operates on unclustered rows, so it does not globally refit all existing visible identities or guarantee identity continuity across ingestion batches.

`cross_verify` compares pairwise same-cluster relationships between the two algorithms. The worker converts agreement labels into 1.0 or 0.5 and averages them per group. This heuristic confidence is neither ground-truth accuracy nor a probability of correct identity. Approximate top-k graph construction can omit radius neighbors, changing DBSCAN results versus an exhaustive distance graph. HDBSCAN still uses the full selected embedding array, and pairwise cross-verification uses nested loops.

### 3.4 SSCD near-duplicate retrieval and grouping

`backend/dedup_engine.py` loads Meta's pretrained SSCD TorchScript checkpoint, described in the repository as a ResNet50 copy-detection model. It uses evaluation/no-gradient inference, resize of the short edge to 288, a 288×288 center crop, RGB tensor conversion and ImageNet mean/std normalization. Outputs are L2-normalized; dimension is probed during loading and storage expects 512 dimensions. `_ensure_loaded` caches the model and remembers load failure. The worker shares a decoded thumbnail-derived image with the scene stage where possible (`compute_dedup_embedding`, `_bstep_dedup_embed_batch`).

`backend/routers.py::_refresh_duplicate_groups` retrieves up to **8** nearest neighbors per image by default (`MEMORIA_DEDUP_KNN`), retains cosine distance **≤0.10** (`MEMORIA_DEDUP_THRESHOLD`), and uses union-find connected components. It streams edges and replaces the group snapshot transactionally. Per-member displayed distance is its nearest retained edge distance, not distance to a learned centroid.

Connected components are transitive: A can group with C through B without A–C satisfying the threshold. Top-k/ANN candidates can also miss links. Thresholds are configurable; no labeled threshold sweep or tuning result is present. Selecting a retained original uses filename/metadata/file-size/date heuristics in `_MEMBER_ORDER_SQL`, not learned perceptual quality (`dedup_delete_duplicates`). SSCD checkpoint benchmark claims in comments describe the upstream model, not evaluation performed by Memoria. The source notes non-commercial weight licensing; enterprise suitability requires separate licensing review.

### 3.5 Deterministic perceptual hash

`backend/workers/tasks.py::compute_phash` converts an image to grayscale 32×32, computes a DCT-II through matrix multiplication, selects the top-left 8×8 coefficients and thresholds them against the median excluding DC from the median calculation. It emits a 64-bit, 16-character hex hash. The bit loop still includes the DC coefficient. Failures return an empty hash and increment a counter.

This is hand-coded signal processing, without pretrained parameters or training. Current duplicate grouping uses SSCD, so do not describe the grouping endpoint as a pHash Hamming-distance search. A speed comparison in the pHash comment is not supported by a committed measurement artifact.

### 3.6 Blur scoring

`backend/scripts/blur_detection/blur_detect.py::score_image_detailed` combines six deterministic sharpness features, then reports `blur_score = 1 - weighted_sharpness`:

| Feature | Normalization/selection | Weight |
|---|---|---:|
| Laplacian variance | variance /1000, capped at 1 | 0.30 |
| Tenengrad | mean squared Sobel gradient /5000, capped | 0.20 |
| FFT | high-frequency energy outside radius 0.10×diagonal; ratio /0.10, capped; max edge 2048 | 0.15 |
| CPBD-inspired edge-width score | Canny 100/200; fraction of measured edges narrower than 5 pixels | 0.15 |
| Brenner | two-pixel horizontal difference energy /2000, capped | 0.10 |
| Local standard deviation | 15×15 window; mean local std /30, capped | 0.10 |

Default classification is blurry above 0.75 (`MEMORIA_BLUR_THRESHOLD`), soft above 0.50, otherwise sharp. The CPBD-named routine is a simplified edge-width heuristic; do not claim a validated reference CPBD implementation. Although comments call these “models,” these are **fixed image-processing rules**, not six trained ML models. FFT may use Torch acceleration; that does not make it deep learning. Worker failures or critical memory pressure can set blur to 0.0, making missing analysis indistinguishable from a sharp score at that field alone (`backend/workers/tasks.py::detect_blur`).

### Training, tuning and optimization boundaries

- **Training/fine-tuning:** no neural-network optimization loop, labeled training split, optimizer, loss computation or fine-tuned artifact found.
- **Unsupervised fitting:** DBSCAN/HDBSCAN fit clusters to pretrained vectors; accurately call this clustering.
- **Hyperparameters:** detector confidence, IoU, clustering radius, neighbor count and blur weights are explicit settings. Configurability is not evidence of systematic tuning, cross-validation or calibrated selection.
- **Pipeline optimization:** batched inference, decoded-image reuse, CPU/GPU overlap, adaptive batch sizing, precision/provider selection, HNSW retrieval and bounded caches are implemented. No measured speedup should be inferred from their presence.
- **Inference:** all pretrained vision encoders/detectors above operate in inference paths. No traditional supervised estimator, time-series forecasting, statistical hypothesis-testing pipeline or risk anomaly-detection system is demonstrated.

## 4. Evaluation and Validation

### Checked-in regression mechanisms

| Test file under `backend/tests/` | Actual validation and limits |
|---|---|
| `test_ml_models.py` | Built-in profile dimensions, aliases, legacy configuration, finite/dimension validation, mocked provider fallback and descriptor round trip. Explicitly avoids model loading/downloads. |
| `test_face_engine_config.py` | Detector/embedder aliases, invalid quorum and driver rejection, fixed 512-dimensional validation, lazy specification inspection. Does not evaluate faces on labeled photos. |
| `test_model_store.py` | Fingerprint tampering rejection, built-in versus custom index plans, malicious custom key remains a bound parameter through a fake async connection. Not a live PostgreSQL integration suite. |
| `test_api_contracts.py` | Route registration, response aliases, hydration order and active namespace usage. Several checks inspect function source strings rather than exercising HTTP and database behavior. |
| `test_database_bootstrap.py`, `test_database_config.py` | Schema-postcondition test doubles/head expectations, database URL normalization/encoding. Not proof of a real clean install or upgrade. |
| `test_watcher.py` | Temporary-file/mocked-dispatch cases: failed enqueues, unreadable subtrees, partial scans, cancellation, recreated files, TTL claims and shutdown lifecycle. |
| `test_sidecar.py` | External-store containment, collision-safe deterministic paths, corruption preservation, read-only guard. |
| `test_huey_shutdown.py` | Interrupted-task reenqueue behavior and Supervisor ordering assertions. |
| `test_library_paths.py` | Root containment, folder matching and bounded/unsafe picker payload cases. |
| `test_processing_config.py`, `test_throughput.py` | Sparse feature updates, SSCD→thumbnail dependency, preset errors, rate calculations, zero-duration behavior, JSONL round trip and malformed-line tolerance. |

These provide deterministic reference expectations for engineering behavior. They are not golden image outputs. No reviewed test provides retrieval recall@k, face precision/recall, cluster purity/ARI, duplicate detection F1, blur classification accuracy, or a cross-model quality comparison. No coverage percentage or current test pass count is asserted.

### Metrics the implementation actually computes

- `backend/metrics.py::_Histogram.snapshot`: sample count, mean, min/max, p50/p95/p99 over a bounded sliding sample window, with inclusive quantiles. Counters and series limits bound diagnostic state. These are operational statistics, not prediction-quality metrics.
- `backend/throughput.py::build_record`: files, bytes/decimal GB, elapsed seconds, files/hour, GB/hour, enabled stages and stage timings. `StageClock` explicitly allows overlapping stage totals to exceed wall time. Single-file callers sum stage timings; batch callers measure their execution path. Rates are not interchangeable with whole-library elapsed throughput.
- `backend/workers/tasks.py::reindex_scene_embeddings`, `_face_reindex_benchmark`: processed/failed counts, encoding/detection time, runtime provider/precision, peak observed VRAM and reindex throughput. Face benchmark batch size is correctly recorded as 1 despite chunked database reads.
- `backend/workers/tasks.py::cluster_faces`: cluster/noise counts and noise percentage; these measure clustering output structure, not correct identities.
- API, database-neighbor, decoding and model-stage timings, cache counters and task failure/retry counters are instrumented in `backend/main.py`, `backend/routers.py`, `backend/workers/__init__.py` and task functions.

`backend/bench.py::main` samples paths, optionally resets metrics, enqueues **single-image tasks**, waits for queue drain, warms representative endpoints and generates Markdown with `render_markdown`. Default count 100 and example count 500 are harness inputs, not demonstrated data volumes. Existing synced sidecars can cause skips; concurrent workload and stale snapshots can contaminate results. Queue drain does not establish successful processing, and this harness does not directly benchmark the watcher's batched path. No saved performance result was found; README also explicitly states none are published.

Reindex benchmark records capture runtime behavior. `finalize_reindex_and_activate` gates on coverage/state/failures and can atomically store a benchmark, but there is **no accuracy or minimum-throughput acceptance threshold** before promotion.

### Diagnostics and manual validation

Error counters distinguish stage errors, memory skips and retries; `processing_log`, `ml_processing_state` and reindex jobs retain statuses/errors. Snapshots and `processing_runs.jsonl` support later inspection, while `frontend/src/components/DiagnosticsDrawer.tsx` and `SettingsView.tsx` expose diagnostics/model operations. UI review supports duplicate inspection, face labeling/merging/hiding and blurry-image review. Those corrections do not feed an implemented training loop. The blur CLI exposes feature breakdowns; `backend/reconcile_sidecars.py::reconcile` supports `dry_run` for recovery preview. These are manual tools, not a documented completed validation campaign.

## 5. Reliability and Failure Handling

### Implemented safeguards

**Input and schema validation.** `backend/library_paths.py` validates canonical filesystem paths and bounded picker requests; `backend/main.py::_safe_media_source` guards media access. Face crop dimensions use FastAPI query bounds. `ModelSpec`, `FaceEngineConfig` and vector validators reject malformed identities, incompatible dimensions and non-finite outputs. Migration `011_pluggable_ml_models.py` adds model/vector dimension constraints, state/count checks and a uniqueness constraint for active reindex work. `docker/bootstrap.py::validate_schema` checks schema postconditions after migration.

**Request boundary.** `backend/main.py::_request_boundary_middleware` checks allowed hosts/origins and requires JSON for API POST/PUT/PATCH. CORS has explicit origins. These are local request safeguards, not authentication or tenant authorization; absent Origin is allowed. Some request bodies are manually parsed, and feature values are coerced with `bool(...)` (`processing_config.py::normalize`), so schema strictness is not uniform.

**Retry/recovery.** Huey task decorators configure finite retries; watcher enqueue helpers retry and preserve retryable discovery state on failure. Interrupted-task handling is tested in `test_huey_shutdown.py`. Scene OOM handling reduces work or falls back to CPU; reindex batch errors can isolate individual images (`reindex_scene_embeddings`). Memory pressure changes batch sizes or skips/pauses work (`backend/memory_mgmt.py::MemoryMonitor`; task stage guards).

**Database atomicity.** Media/vector writes use transactions; per-path upserts make replay practical. Duplicate group replacement is transactional. `backend/model_store.py::finalize_reindex_and_activate` locks the job and model state, checks cancellation and desired identity, briefly SHARE-locks media for coverage checks, rejects failed/incomplete promotion, and commits activation/job completion/optional benchmark together. Non-forced jobs can rewind for newly missing rows. This is an implemented cutover design; race correctness has not been demonstrated here by live concurrency testing.

**Sidecar integrity.** `backend/sidecar.py::save_sidecar` writes a temporary file then `os.replace`; `_SidecarLock` combines thread locks with POSIX advisory locks. Corrupt JSON is preserved with a unique suffix. External storage hashes canonical source-directory names to avoid collisions and source-directory writes. This avoids partial JSON replacement, but there is no cross-filesystem/database atomic transaction or explicit fsync of sidecar file/directory in that helper.

**Human control.** Cleanup/edit routes are explicit API/UI operations, not autonomous deletion based solely on similarity. Read-only mode rejects media mutations. Duplicate deletion moves files to trash and retains rows for failed moves (`backend/routers.py::_read_only_media_guard`, `dedup_delete_duplicates`). Face labeling/merge/hide operations support correction. There is no separate enterprise approval policy or role-based authorization system.

### Material limits and incomplete guarantees

1. **Sync is recoverable dual-write logic, not two-phase commit.** Sidecar and PostgreSQL can disagree after a crash. `sync_metadata` catches database failures, marks the sidecar failed and returns a payload instead of rethrowing; therefore Huey automatic retries do not cover every recorded stage/database failure. The outer task can log completion despite incomplete analysis.
2. **Standalone recovery is narrower than ingestion.** `backend/reconcile_sidecars.py::_upsert_to_postgres` restores path/name/directory/extension, pHash, blur and EXIF only, then marks synced. It does not replay scene vectors, face vectors/provenance, dedup embeddings or full processing-state records. Do not claim complete ML-state restoration from this utility.
3. **Idempotency is state-based.** `process_media` skips ML when a sidecar says synced, and replay/upsert keys use the path. This is not content-hash-based exactly-once processing. A synced marker alone does not prove features match current file bytes or newly enabled stages. Watcher modification detection does not itself establish that every downstream synced entry is invalidated correctly.
4. **Partial model failure may look like absence.** Blur can become 0.0; scene/dedup vectors can be absent; face detections without valid embeddings are not persisted as vectors. Completion must be interpreted with stage/provenance diagnostics. No blanket “every completed photo has all ML outputs” claim is safe.
5. **Filesystem mutation is not rolled back with SQL.** Trash moves precede row deletion; a subsequent database failure does not automatically move files back (`dedup_delete_duplicates`). Editor output and edit records similarly span filesystem and database operations (`_record_edit`, editor handlers).
6. **Auditability is operational.** Model keys, provider/precision, job state, processing logs and edit records provide useful traceability. There is no immutable compliance audit trail, actor identity, signed artifacts or retention policy demonstrated. Metrics reset, bounded windows and best-effort writes can discard diagnostic history.
7. **Model artifacts are not fully attested.** SSCD download handling uses temporary files, size/content-length checks, fsync and ZIP plausibility checks; these guard incomplete downloads, not publisher checksum authenticity (`dedup_engine.py::ensure_model_present`). Some model loading can require network access at first use.

## 6. Engineering Decisions and Tradeoffs

| Design choice | Supported reasoning and tradeoff |
|---|---|
| Separate scene, face and copy-detection spaces | Code assigns different tasks to different encoders; `dedup_engine.py` explicitly motivates copy detection separately from scene semantics. Prevents equating “same subject” with “same photo,” at additional model memory/compute cost. No local quality comparison establishes the best encoder. |
| Strict multi-detector consensus | Face-pipeline comments explicitly motivate false-positive suppression. Two-detector quorum trades availability/recall and compute for stricter acceptance. Claimed false-positive reduction is unmeasured. |
| Desired versus active model state | `model_store.py` keeps search in a coherent embedding namespace during reindex. Inference: safer migration costs extra storage and may require blocking locks at cutover; coverage does not imply quality. |
| Partial HNSW indexes plus exact custom fallback | `scene_knn_plan` uses known model/dimension pairs; custom model keys stay bound. Inference: flexible model integration preserves space isolation, but custom search may require full scans and explicit index work to scale. |
| SQLite Huey and one worker | `workers/__init__.py` documents replacing Redis-backed orchestration to reduce dependencies/RAM; Compose explicitly treats one worker as a model-memory invariant. Simpler local operations constrain horizontal throughput. |
| Batch overlap and decode reuse | `_run_batch_pipelined` and shared image payloads overlap different resources and reduce repeated reads. Inference: extra lifecycle/concurrency complexity; there is no demonstrated acceleration factor. |
| Thumbnail-derived inference | `compute_scene_embedding` and `compute_dedup_embedding` reuse cached data. Inference: lower decoding cost and memory versus possible loss of fine details or altered copy-detection behavior. |
| Heuristics alongside pretrained networks | Blur, pHash, duplicate connected components and selection rules are deterministic and inspectable. They need domain validation and can fail on images unlike the assumptions; no LLM is necessary for these implemented decisions. |
| Local media with external sidecars | `sidecar.py` documents preserving native adjacent sidecars while allowing read-only container libraries. Inference: operational flexibility introduces path mapping and reconciliation complexity. |

Do not infer a documented historical model-selection experiment from the availability of several profiles. Configuration support and engineering rationale are not comparative evaluation.

## 7. Scale, Performance, and Deployment

**Deployment configuration:** `start.sh`, `stop.sh` and `scheduled.sh` supply local lifecycle paths. `Dockerfile` builds Python dependencies and a standalone Next.js frontend. `compose.yaml` pairs the app with PostgreSQL 16/pgvector 0.8.2; Supervisor runs API, worker, watcher and frontend in the app container (`docker/supervisord.conf`). `docker/bootstrap.py::prepare_database` migrates and checks the database. `compose.gpu.yaml` supplies an optional NVIDIA deployment profile; CPU/GPU requirements and `backend/dependency_smoke.py` check dependency/provider boundaries. These are deployment mechanisms, not evidence of a running service.

Compose defaults to loopback-published API/web ports, read-only media and image filesystem, a writable persistent state volume, dropped capabilities, no-new-privileges, health checks and bounded container log rotation. Default ingestion batch size is 16; one Huey thread-worker is enforced in Compose. Native mode and environment overrides can differ. ML singleton instances are per process, so API text inference and worker image inference can each hold model state; the one-worker setting does not mean one model instance across the entire application.

**Demonstrated scale:** repository source and synthetic test fixtures establish behavior, not a measured production library size. No sustained ingestion rate, peak concurrent-user load, labeled sample count, service uptime, cloud deployment record or enterprise workload is evidenced. Benchmark command examples must not become resume numbers.

**Limits inferred from code:**

- SQLite queue/scan index, filesystem locks and local cache paths target one host; scaling replicas requires shared-state and ownership design.
- HNSW is approximate; candidate truncation affects clustering/grouping as well as recall. Custom scene models can use exact scans.
- Face clustering loads all selected unclustered embeddings, HDBSCAN processes them together, and `cross_verify` is O(n²) in pair comparisons even though DBSCAN's neighbor graph is sparse.
- Duplicate graph construction performs per-image neighbor queries and retains union-find state in memory; groups can chain over broad sets.
- Sidecar updates load and rewrite per-directory JSON. Very large directories increase write cost; locks serialize concurrent updates.
- `throughput.py::read_recent` bounds returned records with a deque but scans the entire JSONL file; its memory bound is not a time bound. Logging has no demonstrated application-level retention service.
- A large model may load on the first search request, producing cold-start latency. API inference is offloaded to a thread but the engine serializes inference with locks.

**Enterprise changes proposed, not implemented:** add authenticated/authorized tenants, protected network access and actor audit records; migrate queue/state coordination for distributed workers; define content/version-aware completion and explicit per-stage failure states; make recovery cover all derived state; build representative labeled retrieval/face/dedup/blur evaluation sets and model-change gates; validate ANN recall against exact search; benchmark actual target hardware/data; introduce centralized telemetry/alerts, retention, backups and restore drills; pin/verify model artifacts and review licenses. These follow from the local trust boundary, recovery gaps and unmeasured quality, not from a claim that the application currently supports enterprise operations.

## 8. Fortifai Relevance

Classification concerns repository capability, not proof that a particular applicant personally implemented it.

| Requirement | Classification | Verified relevance and boundary |
|---|---|---|
| Applied ML/deep learning | **Directly demonstrated** | Pretrained multimodal/face/copy-detection inference, batching, normalization, device handling, and density clustering. `ml/scene.py`, `face_detection/face_pipeline.py`, `dedup_engine.py`, `workers/tasks.py::cluster_faces`. No neural training/fine-tuning evidence. |
| Statistical or anomaly-detection techniques | **Partially demonstrated** | Deterministic image statistics and density-clustering noise labels. `blur_detect.py`, `face_cluster.py`. No temporal, financial or risk anomaly detector; noise is not validated anomaly detection. |
| LLM and agentic systems | **Not demonstrated** | CLIP/SigLIP text encoders and Huey workflows are not generative LLMs or agents. No prompt/tool loop or LLM evaluation found. |
| Structured-data retrieval and reasoning | **Partially demonstrated** | Direct SQL/JSON metadata joins, model-scoped vector retrieval and rule-based grouping/selection. `model_store.py`, `routers.py`. No LLM reasoning over structured records or risk-domain inference. |
| Model evaluation and regression testing | **Partially demonstrated** | Engineering regressions, runtime benchmarks, failure counts and migration coverage checks. `backend/tests/`, `bench.py`. No labeled predictive-quality evaluation or golden image regression set. |
| Production Python engineering | **Directly demonstrated** | FastAPI, asyncpg/psycopg, Huey, migrations, typed model contracts, tests and container configuration. Evidence of engineering practices, not proven production operations/SLA. |
| Reliability, observability and auditability | **Partially demonstrated** | Transactions, recovery markers, locks, retries, provenance, metrics and heartbeat. Incomplete sidecar replay, soft-failure semantics and lack of identity/audit immutability limit stronger claims. |

For Fortifai, the strongest transfer is reliable inference/retrieval infrastructure and explicit model lifecycle handling. Risk-domain modeling, generative agents and quantitative model-quality evaluation remain gaps to discuss candidly.

## 9. Technical Interview Deep Dive

**1. How do you prevent incompatible embeddings from being compared during a model change?**

Model keys identify distinct spaces; scene rows are namespaced and dimension-checked, while search reads the active model. Desired vectors can be built separately, with coverage/cancellation-aware atomic promotion. ArcFace and FaceNet are separate keys despite both being 512-dimensional (`backend/model_store.py::scene_knn_plan`, `finalize_reindex_and_activate`; `backend/ml/face.py`). Descriptor hashes do not verify weight bytes, and completed coverage is not a relevance-quality gate.

**2. What did the project train, and what would you call hyperparameter tuning?**

It integrates pretrained encoders/detectors for inference and fits unsupervised DBSCAN/HDBSCAN clusters. It does not train/fine-tune a neural model. Thresholds and profiles are configurable; no search over a labeled validation set is shown. Batching, precision and decode reuse are pipeline optimization, not model training (`backend/ml/scene.py`, `backend/dedup_engine.py`, `backend/face_detection/face_cluster.py`).

**3. Is sparse ANN-based DBSCAN equivalent to full DBSCAN, and is agreement confidence accuracy?**

No. k=20 and approximate neighbors can omit valid epsilon edges and alter connectivity/density. The worker runs HDBSCAN on normalized vectors and compares pair relationships, but agreement can be jointly wrong. Cluster confidence averages heuristic 1.0/0.5 values, not labeled correctness (`backend/workers/tasks.py::cluster_faces`; `backend/face_detection/face_cluster.py::cross_verify`). Validate against exact neighborhoods and labeled identities before making accuracy claims.

**4. What happens if the process dies between the sidecar write and SQL commit?**

A pending sidecar supports replay; SQL transactions protect database writes, and file replacement protects JSON visibility. However, the stores are not jointly atomic. `sync_metadata` swallows some database errors, and standalone reconciliation only restores basic media fields before marking synced. Full recovery needs explicit derived-state verification/backfill (`backend/workers/tasks.py::sync_metadata`; `backend/sidecar.py::save_sidecar`; `backend/reconcile_sidecars.py::_upsert_to_postgres`).

**5. Why use SSCD rather than the scene model or pHash for duplicate groups?**

The source explicitly separates pretrained copy-detection features from broad scene semantics. pHash remains a deterministic coarse descriptor; grouping uses SSCD cosine neighbors and union-find. The code establishes this task separation, not superiority on this user's library. Connected components can chain non-identical endpoints and thresholds are uncalibrated locally (`backend/dedup_engine.py`; `backend/routers.py::_refresh_duplicate_groups`).

**6. How would you substantiate a throughput claim?**

Use a fresh/disposable, described library, fixed hardware/models/provider/precision/preset/batch size, record cold/warm conditions and successful versus failed/skipped work, and measure wall time. Existing metrics/JSONL and `bench.py` provide tooling, but the harness enqueues single-image tasks and queue drain alone is insufficient. No measured improvement is currently available (`backend/bench.py::main`; `backend/throughput.py::build_record`).

**7. What happens under GPU memory pressure or a failed detector?**

Scene engines can split batches and switch to CPU; memory monitoring adapts work. Face inference requires successful detector quorum and preserves distinct embedding identities. Outer stages may soft-skip rather than fail the job, so operational availability can come at feature completeness cost. Consult actual runtime fields and stage counters, not only job completion (`backend/ml/scene.py`; `backend/face_detection/face_pipeline.py::FaceDetector.detect`; `backend/workers/tasks.py::detect_blur`, `detect_faces`, `compute_scene_embedding`).

**8. What would block deploying this as a multi-user risk intelligence service?**

The domain and outputs differ, and there is no risk model or LLM agent. Engineering blockers include absent authentication/tenant isolation, single-host queue/state assumptions, incomplete full-state recovery and missing labeled model evaluation. The existing model registry, transactional cutover, SQL retrieval and diagnostic infrastructure are reusable design experience, not proof of enterprise readiness (`compose.yaml`, `backend/main.py::_request_boundary_middleware`, `backend/model_store.py`, `backend/reconcile_sidecars.py`).

## 10. Evidence Index

Strength: **implementation** = direct executable code; **regression** = checked-in assertion, not a test result from this review; **configuration** = declared deployment/dependency behavior; **inference** = reasoned limitation. All paths below are repository-relative.

| Technical claim | Supporting file path | Relevant function/class/test | Evidence strength | Qualification |
|---|---|---|---|---|
| Incremental discovery and safe retry bookkeeping | `backend/watcher.py` | `incremental_scan`, `_dispatch_scan_batch` | Implementation | Filesystem/runtime behavior not executed here. |
| Failed dispatch and unreadable subtrees are tested | `backend/tests/test_watcher.py` | `test_failed_scan_dispatch_removes_index_row_for_retry`, `test_unreadable_subtree_is_not_pruned_as_deleted` | Regression | Mocked/temp-file coverage, not full-stack proof. |
| Durable local task queue | `backend/workers/__init__.py` | `huey = SqliteHuey(...)` | Implementation | SQLite, not distributed orchestration. |
| Interrupted work reenqueue regression | `backend/tests/test_huey_shutdown.py` | `test_interrupted_task_is_requeued_exactly_once` | Regression | Test name does not establish global exactly-once processing. |
| Batched CPU/GPU overlap | `backend/workers/tasks.py` | `_run_batch_pipelined`, `process_media_batch` | Implementation | No measured speedup. |
| Processing presets preserve sparse updates | `backend/processing_config.py`; `backend/tests/test_processing_config.py` | `normalize`, `test_sparse_update_preserves_current_features` | Implementation + regression | Boolean coercion is permissive. |
| Shared decoders and thumbnails | `backend/image_utils.py` | `decode_pil`, `decode_cv2_bgr`, `make_thumbnail` | Implementation | Codec support depends on installed dependencies. |
| Pretrained scene profiles | `backend/ml/scene.py` | `BUILTIN_SCENE_SPECS`, `OpenClipSceneEngine`, `HuggingFaceSiglipSceneEngine` | Implementation | Library models; no local training. |
| Provider fallback and precision reporting | `backend/ml/providers.py`; `backend/ml/scene.py` | `resolve_execution`, `_switch_to_cpu` | Implementation | Hardware/provider availability unverified. |
| Vector shape and finite checks | `backend/ml/base.py`; `backend/tests/test_ml_models.py` | `validate_embedding`, `test_embedding_validation_rejects_non_finite_and_wrong_length` | Implementation + regression | Does not validate semantic quality. |
| Model descriptor identity | `backend/model_store.py`; `backend/tests/test_model_store.py` | `normalize_model_descriptor`, `test_descriptor_fingerprint_is_recomputed_and_tampering_is_rejected` | Implementation + regression | Configuration digest, not weights checksum. |
| Model/dimension-specific ANN and exact fallback | `backend/model_store.py` | `scene_knn_plan`, `query_scene_neighbors_async` | Implementation | Exact custom queries may be costly. |
| Safe custom SQL model binding | `backend/tests/test_model_store.py` | `test_custom_model_key_remains_bound_and_uses_exact_query_shape` | Regression | Fake connection; no real planner test. |
| Active-space text/image search | `backend/routers.py` | `search_text`, `search_similar`, `_hydrate_scene_neighbors` | Implementation | No generative LLM or reranker. |
| Face quorum and geometry filtering | `backend/face_detection/face_pipeline.py` | `_merge_detections`, `_is_valid_face_bbox`, `FaceDetector.detect` | Implementation | No labeled precision/recall results. |
| Face vector spaces remain distinct | `backend/ml/face.py` | `FaceEngineConfig`, `INSIGHTFACE_ARCFACE_SPEC`, `FACENET_VGGFACE2_SPEC` | Implementation | Same dimension does not mean interoperability. |
| DBSCAN/HDBSCAN clustering and agreement | `backend/workers/tasks.py`; `backend/face_detection/face_cluster.py` | `cluster_faces`, `cluster_dbscan_precomputed`, `cross_verify` | Implementation | ANN approximation; agreement is not accuracy. |
| SSCD inference | `backend/dedup_engine.py` | `_build_preprocess`, `_load`, `encode_image`, `encode_images_batch` | Implementation | Upstream pretrained weights; license note in source. |
| Duplicate grouping and trash behavior | `backend/routers.py` | `_refresh_duplicate_groups`, `dedup_delete_duplicates` | Implementation | Transitive groups; file moves and SQL not atomic together. |
| Hand-coded perceptual hash | `backend/workers/tasks.py` | `compute_phash` | Implementation | Not current duplicate-group distance engine. |
| Fixed blur ensemble | `backend/scripts/blur_detection/blur_detect.py` | `score_image_detailed`, `_score_cpbd`, `_classify` | Implementation | Heuristics, not trained models or calibrated probabilities. |
| Namespaced schema constraints/indexes | `backend/alembic/versions/011_pluggable_ml_models.py`; `backend/alembic/versions/012_vector_schema_repair.py` | `upgrade` | Implementation | Migration availability does not prove deployed schema. |
| Guarded model cutover | `backend/model_store.py` | `finalize_reindex_and_activate`, async counterpart | Implementation | Coverage/runtime benchmark, no quality gate. |
| Pending/synced dual-write workflow | `backend/workers/tasks.py` | `sync_metadata` | Implementation | Caught failures may not trigger task retries. |
| Atomic sidecars and corruption preservation | `backend/sidecar.py`; `backend/tests/test_sidecar.py` | `save_sidecar`, `load_sidecar`, `_SidecarLock` | Implementation + regression | Not distributed commit or full durability proof. |
| Recovery restores basic fields only | `backend/reconcile_sidecars.py` | `_upsert_to_postgres`, `reconcile` | Implementation | Missing complete derived-state replay. |
| Percentile instrumentation | `backend/metrics.py` | `_Histogram.snapshot`, `PeriodicSnapshotter` | Implementation | Bounded windows; no retained performance results found. |
| Persistent throughput records | `backend/throughput.py`; `backend/tests/test_throughput.py` | `build_record`, `log_run`, `test_build_record_rates` | Implementation + regression | Rate calculation tests are not benchmark measurements. |
| Benchmark harness | `backend/bench.py` | `main`, `render_markdown` | Implementation | Single-file dispatch; no committed run result. |
| Local request/media boundaries | `backend/main.py`; `backend/library_paths.py`; `backend/routers.py` | `_request_boundary_middleware`, `validate_picked_scan_request`, `_read_only_media_guard` | Implementation | No authenticated user/tenant authorization. |
| Container/bootstrap support | `Dockerfile`; `compose.yaml`; `compose.gpu.yaml`; `docker/bootstrap.py` | build stages, services, `prepare_database`, `validate_schema` | Configuration + implementation | No build/deployment verification performed. |
| Video inference absent | `backend/constants.py`; `backend/workers/tasks.py` | `VIDEO_EXTENSIONS`, `process_media` | Implementation | Discovery/skip is not video understanding. |
| Enterprise scaling limitations | `backend/face_detection/face_cluster.py`; `backend/sidecar.py`; `backend/throughput.py` | `cross_verify`, `save_sidecar`, `read_recent` | Inference from implementation | No measured upper-capacity bound asserted. |

## Claims Safe to Use on a Resume

Candidate project-level achievements for human review. Use personal-action wording only after confirming contribution/ownership; no claim below implies measured accuracy, production deployment or model training.

- Integrated pretrained OpenCLIP/SigLIP, face-analysis and SSCD inference into a configurable Python photo-processing pipeline with batched execution and runtime device handling.
- Implemented model-keyed pgvector retrieval and transactional reindex promotion with dimension validation, coverage checks and cancellation handling.
- Combined detector consensus with pretrained face embeddings and DBSCAN/HDBSCAN clustering for person grouping, retaining model provenance.
- Added filesystem discovery/retry handling, atomic JSON sidecars and PostgreSQL synchronization, with regression tests covering recovery-related edge cases.
- Built operational instrumentation for stage latency, processing throughput, memory pressure and model reindex runs, plus a reproducible benchmark harness.
