**Status:** Early public release. The system is actively evolving, and some architectural choices, edge cases, and failure modes are still being refined.

# Memoria

Local-first, single-user photo indexing. Source media stays on the deployment; PostgreSQL/pgvector stores the index; ML runs in the worker process.

## Features

Browse a filesystem photo library, search by scene or similar image, organize faces, review blurry photos and near-duplicates, and explore GPS locations on a map. Model selection, re-indexing, and diagnostics are available in the web UI. Inference runs locally on the deployment; a GPU is optional.

## Runtime

- Python 3.10-3.12, FastAPI/Uvicorn, PostgreSQL 16+ with `pgvector` and `uuid-ossp`.
- One Huey SQLite queue and one thread worker. Do not increase worker count.
- Next.js standalone server on port `5004`; API on `8000`.
- CPU is supported. CUDA/ONNX CUDA/TensorRT are optional.
- Native writable state defaults to `~/.memoria`; Compose overrides it under `/var/lib/memoria`.

## Docker Compose (recommended)

Install Docker Engine or Docker Desktop with Compose 2.30+ before starting.
Run the following commands from the repository root.

The base Compose build is CPU-only and uses the Docker engine's native
architecture. On an Apple Silicon Mac it builds Linux/arm64 layers and never
resolves an NVIDIA/CUDA base image. Docker Desktop does not expose Apple MPS to
Linux containers, so inference in Docker is CPU-based; use native mode only if
you specifically want PyTorch MPS.

```bash
cp .env.example .env
# Edit .env: set DB_PASSWORD to a unique password.
# Set MEMORIA_MEDIA_PATH=/absolute/path/to/your/photos.
# If unset, the repo-local ./media directory is mounted.
./start.sh --docker
```

Open Memoria at `http://localhost:5004` once startup completes. Initial indexing
and model downloads can take time; the sidebar status pill shows progress.

The folder picker uses a separate read-only `/host` mount, which defaults to
`./media`. Set `MEMORIA_HOST_MOUNT` to your photo directory or the narrow common
parent of several libraries to make them available in the picker. The default
automatic scan uses `MEMORIA_MEDIA_PATH`, mounted at `/media/library`.

`start.sh --docker` creates and waits for PostgreSQL, creates the database/user
through the official image entrypoint, enables `vector` and `uuid-ossp`, applies
every Alembic migration, and waits for the API and web server health checks. No
database GUI, manual `CREATE DATABASE`, or manual SQL is required.

On a Linux x86_64 host with the NVIDIA Container Toolkit, auto-detection selects
the CUDA overlay. Selection can also be made explicit:

```bash
./start.sh --docker --cpu  # portable CPU image, including Apple Silicon
./start.sh --docker --gpu  # Linux x86_64 + visible NVIDIA GPU only
```

The equivalent direct commands are:

```bash
docker compose up -d --build --wait --wait-timeout 300
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build --wait --wait-timeout 300
```

## Data flow

```text
configured roots
  -> incremental scan + watchdog events
  -> Huey/SQLite
  -> one batched worker
       thumbnail, stat, EXIF/GPS, pHash, blur
       face detection/embedding, scene embedding, SSCD embedding
       sidecar pending -> PostgreSQL upsert -> sidecar synced
  -> FastAPI + PostgreSQL/pgvector
  -> Next.js UI
```

The worker writes scene vectors to the desired model namespace and search reads only the active namespace. A model is promoted only after its re-index and benchmark transaction completes. Face and dedup vectors remain 512-dimensional; scene dimensions are model-keyed.

## Storage contract

| Store | Contents |
| --- | --- |
| PostgreSQL | media metadata, model catalog/state, scene/face/dedup vectors, jobs, benchmarks |
| SQLite | Huey queue and incremental-scan `(path, mtime, size)` index |
| Sidecar JSON | recoverable per-directory metadata journal; `pending`, `synced`, or `failed` |
| Runtime state | thumbnails, previews, face crops, model weights, metrics, edits, trash |

Compose mounts source media read-only and puts sidecars in the configured writable state volume. Native mode keeps adjacent sidecars unless `MEMORIA_SIDECAR_ROOT` is set.

## Native execution (optional)

Install PostgreSQL client/server 16+, `pgvector`, `uuid-ossp`, Python 3.10-3.12, Node.js 20+, and the native libraries required by Pillow/OpenCV/rawpy. On macOS, native setup uses the CPU `onnxruntime` wheel because `onnxruntime-gpu` is Linux/NVIDIA-only; PyTorch may still use Apple MPS. Create a database or provide a full PostgreSQL URL.

```bash
cp .env.example .env
# Set CRAWL_ROOTS and DB_* or DATABASE_URL.
./start.sh
```

`start.sh` creates `backend/venv`, selects the CPU or NVIDIA dependency profile,
waits for PostgreSQL, enables the two extensions, runs `alembic upgrade head`,
installs frontend dependencies with `npm ci`, builds the standalone frontend,
then starts Uvicorn, the Huey worker, the watcher, and Next.js. Native mode
requires PostgreSQL client tools on the host; use Compose when PostgreSQL should
be entirely container-managed.

```bash
./stop.sh
backend/venv/bin/pip install -r backend/requirements-dev.txt
backend/venv/bin/pip install --no-deps -r backend/requirements-managed-no-deps.txt
backend/venv/bin/python backend/dependency_smoke.py cpu
PYTHONPATH=backend backend/venv/bin/pytest -q backend/tests
```

Use `MEMORIA_FRONTEND_MODE=development` only for frontend work. Logs are under `logs/`; PID files are under `.pids/`.

## Configuration

The operational defaults are in `.env.example`. The settings that affect identity or throughput are:

- `CRAWL_ROOTS`, `MEMORIA_MEDIA_READ_ONLY`, `MEMORIA_SIDECAR_ROOT`.
- `DATABASE_URL` or `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`.
- `MEMORIA_BATCH_SIZE` and `MEMORIA_HUEY_WORKERS=1`.
- `MEMORIA_SCENE_PROFILE` plus optional scene descriptor fields.
- `MEMORIA_FACE_*` and `MEMORIA_INSIGHTFACE_USE_TRT`.
- `MEMORIA_DEDUP_THRESHOLD`, `MEMORIA_DEDUP_KNN`, `MEMORIA_HNSW_EF_SEARCH`.

Do not reuse a model key for different weights, dimensions, input shapes, or semantics. Do not compare vectors from different model keys.

## Evaluation

Run `backend/bench.py` against a declared sample root and count. Report deployment, CPU/GPU, model/provider/precision, batch size, and dataset. The relevant metrics are:

- end-to-end files/sec and GB/hour;
- worker stage mean/p50/p95/p99/max;
- API route p50/p95/p99/max;
- face and semantic vector-search p95;
- scene re-index images/sec and peak VRAM;
- cached thumbnail p95;
- queue depth, failed work, and memory pressure events.

The benchmarking guide explains how to run `backend/bench.py` against a
disposable sample library in either Compose or native mode. It measures
ingestion throughput, worker-stage timings, API latency, vector-search timing,
cache behavior, and recorded model benchmarks. Results are meaningful only
when they include the dataset, hardware, storage, model key, provider,
precision, batch size, and whether warm-up is included. Queue drain is not a
guarantee that every image succeeded, and the single-file harness does not
represent the watcher's batched-ingestion throughput. No measured performance
results are published in this repository. The commands and measurement limits
are described above.

## Trust boundary

Memoria has no built-in authentication or multi-user isolation. Keep the default
loopback bindings; remote access requires an external authenticated access layer.

Memoria is single-user and local-first. Configure narrow `CRAWL_ROOTS` or
`MEMORIA_LIBRARY_MOUNTS`; requests are restricted to canonical in-root paths
and excluded directories. Compose source media is read-only and writable state
is kept in its persistent volume. Set a unique `DB_PASSWORD` in `.env` before
Compose startup and keep it unchanged for an existing database volume.

Before publishing or deploying, run the manual release gate against a
disposable sample library. It covers publication hygiene (including secrets,
private paths, runtime state, model assets, and generated benchmark reports),
automated checks, native/Compose smoke tests, filesystem boundaries, recovery,
model-namespace safety, and documentation/evidence review. The release gate
described above should be completed before publishing or deploying.

Local inference does not imply zero network traffic: model weights may be
downloaded once when absent, and the Map view requests OpenStreetMap raster
tiles. Next.js telemetry is disabled by default for native, Compose, and direct
package-script execution.

## Safety invariants

- Alembic is the only schema mechanism and migrations are repeat-safe.
- Source files are never overwritten by indexing. Editor outputs are non-destructive; delete operations move files to runtime trash.
- Sidecar writes are atomic and replayable. Database-only embeddings require a full re-index after database loss.
- Excluded paths are centralized in `backend/constants.py`.
- Every frontend API URL is built in `frontend/src/lib/endpoints.ts`.

## Operations and recovery

Stop Compose with `./stop.sh --docker` or native services with `./stop.sh`.
Normal shutdown preserves indexed data. For container diagnostics, use
`docker compose ps` and `docker compose logs --tail=100 app db`.

Back up PostgreSQL, the persistent application state volume, and your source
library. For native deployments, include configured writable state and adjacent
sidecars. Stop ingestion while taking a coordinated backup. Thumbnails and
previews can be regenerated; model vectors exist only in PostgreSQL.

The sidecar reconciler in `backend/reconcile_sidecars.py` can replay pending or
failed metadata syncs. Run it with the same Python environment, database
configuration, and filesystem paths as the worker; use `--dry-run` first.
It skips synced entries and cannot restore embeddings after database loss.
A lost database requires full media/model re-indexing.

`./clean_slate.sh` (or `./clean_slate.sh --docker`) is a confirmed reset, not a
routine shutdown. Docker reset removes persistent volumes without an automatic
backup. Preserve required data before using it.

## Included reference material

The architecture, benchmarking, frontend-development, release-validation, and
environment details are summarized in this README so it can be used on its own.
