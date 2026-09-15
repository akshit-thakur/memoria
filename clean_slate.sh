#!/usr/bin/env bash
set -euo pipefail

# clean_slate.sh
#
# In plain English: the native/Docker "re-index from scratch" script. Native
# mode clears the Huey queue, scan index, cache, Postgres media rows, and
# configured sidecars while preserving source photos, models, edits, and trash.
# Docker mode removes the Compose state volumes after a separate confirmation.
#
# Before native-mode deletion it makes timestamped backups in /tmp:
#   • /tmp/memoria-queue-backup-<timestamp>.db
#   • /tmp/memoria-scan-backup-<timestamp>.db
#   • /tmp/memoria-db-backup-<timestamp>.dump
#   • /tmp/memoria_sidecars_backup-<timestamp>.tar.gz
# so a misclick is recoverable.
#
# Always run with `--dry-run` first to see exactly what will happen.
# `--yes` skips the interactive confirmation; useful for automated tests.
#
# Safely back up and then wipe queued tasks, DB media rows, and per-directory
# sidecar JSON (.memoria_metadata.json). Use --dry-run to print actions without
# executing. Use --yes to skip confirmation prompts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS=$(date +%Y%m%d%H%M%S)
DRY_RUN=false
YES=false
DOCKER_REQUESTED=false

usage() {
  cat <<EOF
Usage: $0 [--docker|-d] [--dry-run] [--yes]

Native mode will:
  - stop watcher/huey_consumer/uvicorn processes (if running)
  - back up the Huey SQLite queue (queue.db) to /tmp
  - delete queue.db and scan_index.db
  - clear the configured Memoria cache directory
  - back up Postgres DB to /tmp (pg_dump)
  - truncate media plus model re-index/benchmark state in Postgres
  - archive and delete sidecars in MEMORIA_SIDECAR_ROOT, or legacy adjacent
    sidecars below explicit CRAWL_ROOTS (skipped when CRAWL_ROOTS is unset)

Docker mode will run 'docker compose down -v --remove-orphans'. This removes
the Memoria Compose containers, network, and named volumes. It does not create
automatic volume backups or touch source media and unrelated repository paths.

Options:
  --docker, -d  Force Docker cleanup. Without this flag, an existing Memoria
                Compose deployment is detected automatically.
  --dry-run    Print actions but do not execute them.
  --yes, -y    Skip confirmation prompt and perform destructive actions.
  --help, -h   Show this message.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --docker|-d) DOCKER_REQUESTED=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --yes|-y) YES=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" = false ]; then
    "$@"
  fi
}

COMPOSE=(docker compose --project-directory "$ROOT_DIR" -f "$ROOT_DIR/compose.yaml")
DOCKER_AVAILABLE=false
DOCKER_DEPLOYMENT=false
if command -v docker >/dev/null 2>&1 \
  && docker compose version >/dev/null 2>&1 \
  && docker info >/dev/null 2>&1; then
  DOCKER_AVAILABLE=true
  COMPOSE_IDS=$("${COMPOSE[@]}" ps -a -q 2>/dev/null || true)
  [ -n "$COMPOSE_IDS" ] && DOCKER_DEPLOYMENT=true
fi

if [ "$DOCKER_REQUESTED" = true ] && [ "$DOCKER_AVAILABLE" = false ]; then
  echo "--docker was requested, but Docker Engine or Docker Compose v2 is unavailable." >&2
  exit 1
fi

if [ "$DOCKER_REQUESTED" = true ] || [ "$DOCKER_DEPLOYMENT" = true ]; then
  echo "clean_slate: Docker mode will stop the Memoria Compose project and remove its named volumes."
  echo "Source media and unrelated repository directories are not touched."
  echo "This removes database, queue, cache, sidecar, model, edit, and trash state stored in Compose volumes."

  if [ "$YES" = false ] && [ "$DRY_RUN" = false ]; then
    read -r -p "Permanently reset the Docker deployment and its volumes? [y/N] " ans
    case "$ans" in
      [Yy]*) ;;
      *) echo "Aborted."; exit 1 ;;
    esac
  fi

  run "${COMPOSE[@]}" down -v --remove-orphans

  echo "clean_slate Docker reset completed. Restart with: ./start.sh --docker"
  exit 0
fi

[ -f "$ROOT_DIR/.env" ] && set -a && source "$ROOT_DIR/.env" && set +a
DB_NAME=${DB_NAME:-memoria}
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_USER=${DB_USER:-$(whoami)}
DATABASE_URL_SUPPLIED=false
[ -n "${DATABASE_URL:-}" ] && DATABASE_URL_SUPPLIED=true

# ``realpath -m`` is a GNU extension and fails with the BSD ``realpath``
# shipped by macOS. Native Memoria already requires Python, whose realpath
# implementation safely resolves existing symlinks while also normalising a
# not-yet-created final path (the behaviour we need for reset validation).
PATH_PYTHON="$(command -v python3 || true)"
if [ -z "$PATH_PYTHON" ] && [ -x "$ROOT_DIR/backend/venv/bin/python" ]; then
  PATH_PYTHON="$ROOT_DIR/backend/venv/bin/python"
fi
[ -n "$PATH_PYTHON" ] \
  || { echo "python3 is required to validate reset paths" >&2; exit 1; }

resolve_path() {
  "$PATH_PYTHON" -c \
    'import os, sys; print(os.path.realpath(os.path.abspath(sys.argv[1])))' \
    "$1"
}

[ -n "${DATABASE_URL:-}" ] && {
  DB_URL_PARTS=$("$PATH_PYTHON" -c '
import sys
from urllib.parse import unquote, urlparse
u = urlparse(sys.argv[1])
if u.scheme not in ("postgres", "postgresql") or not u.hostname or not u.path.lstrip("/"):
    raise SystemExit(2)
print("\x1f".join((u.hostname, str(u.port or 5432), unquote(u.username or ""), unquote(u.password or ""), unquote(u.path.lstrip("/")))))
' "$DATABASE_URL") || { echo "DATABASE_URL is not a valid PostgreSQL URL" >&2; exit 1; }
  IFS=$'\x1f' read -r DB_HOST DB_PORT DB_USER DB_PASSWORD DB_NAME <<< "$DB_URL_PARTS"
}
[ -n "${DB_PASSWORD:-}" ] && export PGPASSWORD="$DB_PASSWORD"

database_psql() {
  if [ "$DATABASE_URL_SUPPLIED" = true ]; then
    psql --dbname="$DATABASE_URL" "$@"
  else
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" "$@"
  fi
}

database_dump() {
  if [ "$DATABASE_URL_SUPPLIED" = true ]; then
    pg_dump --dbname="$DATABASE_URL" "$@"
  else
    pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$@" "$DB_NAME"
  fi
}

for path_var in MEMORIA_STATE_DIR MEMORIA_DATA_DIR MEMORIA_CACHE_DIR HUEY_DB MEMORIA_SCAN_INDEX_DB; do
  path_value="${!path_var:-}"
  case "$path_value" in
    "~") path_value="$HOME" ;;
    "~/"*) path_value="$HOME/${path_value:2}" ;;
  esac
  if [ -n "$path_value" ] && [[ "$path_value" != /* ]]; then
    echo "$path_var must be an absolute path for a destructive reset: $path_value" >&2
    exit 1
  fi
  [ -z "$path_value" ] || printf -v "$path_var" '%s' "$path_value"
done
STATE_ROOT_DIR="${MEMORIA_STATE_DIR:-${HOME%/}/.memoria}"
DEFAULT_DATA_DIR="${STATE_ROOT_DIR%/}/data"
STATE_DATA_DIR="${MEMORIA_DATA_DIR:-$DEFAULT_DATA_DIR}"
QUEUE_DB="${HUEY_DB:-${STATE_DATA_DIR%/}/queue.db}"
SCAN_DB="${MEMORIA_SCAN_INDEX_DB:-${STATE_DATA_DIR%/}/scan_index.db}"
CACHE_DIR="${MEMORIA_CACHE_DIR:-${STATE_ROOT_DIR%/}/cache}"

STATE_ROOT_RESOLVED=$(resolve_path "$STATE_ROOT_DIR")
CACHE_DIR_RESOLVED=$(resolve_path "$CACHE_DIR")
HOME_RESOLVED=$(resolve_path "$HOME")
REPO_RESOLVED=$(resolve_path "$ROOT_DIR")
case "$CACHE_DIR_RESOLVED" in
  /|"$HOME_RESOLVED"|"$REPO_RESOLVED"|"$STATE_ROOT_RESOLVED")
    echo "Refusing unsafe cache reset target: $CACHE_DIR_RESOLVED" >&2
    exit 1
    ;;
esac
if [[ "$CACHE_DIR_RESOLVED" != "$STATE_ROOT_RESOLVED/"* ]]; then
  echo "MEMORIA_CACHE_DIR must be a child of MEMORIA_STATE_DIR for clean_slate: $CACHE_DIR_RESOLVED" >&2
  exit 1
fi

SIDECAR_ROOT_RESOLVED=""
if [ -n "${MEMORIA_SIDECAR_ROOT:-}" ]; then
  [[ "$MEMORIA_SIDECAR_ROOT" = /* ]] \
    || { echo "MEMORIA_SIDECAR_ROOT must be absolute for a destructive reset." >&2; exit 1; }
  SIDECAR_ROOT_RESOLVED=$(resolve_path "$MEMORIA_SIDECAR_ROOT")
  HOME_RESOLVED=$(resolve_path "$HOME")
  REPO_RESOLVED=$(resolve_path "$ROOT_DIR")
  case "$SIDECAR_ROOT_RESOLVED" in
    /|"$HOME_RESOLVED"|"$REPO_RESOLVED")
      echo "Refusing unsafe MEMORIA_SIDECAR_ROOT: $SIDECAR_ROOT_RESOLVED" >&2
      exit 1
      ;;
  esac
  if [[ "$HOME_RESOLVED" == "$SIDECAR_ROOT_RESOLVED/"* || "$REPO_RESOLVED" == "$SIDECAR_ROOT_RESOLVED/"* ]]; then
    echo "Refusing MEMORIA_SIDECAR_ROOT that contains HOME or the repository: $SIDECAR_ROOT_RESOLVED" >&2
    exit 1
  fi

  if [ -n "${MEMORIA_STATE_DIR:-}" ]; then
    STATE_RESOLVED=$(resolve_path "$MEMORIA_STATE_DIR")
    if [[ "$STATE_RESOLVED" == "$SIDECAR_ROOT_RESOLVED" || "$STATE_RESOLVED" == "$SIDECAR_ROOT_RESOLVED/"* ]]; then
      echo "MEMORIA_SIDECAR_ROOT must be a dedicated child, not MEMORIA_STATE_DIR or its parent." >&2
      exit 1
    fi
  fi

  if [ -n "${CRAWL_ROOTS:-}" ]; then
    IFS=':' read -r -a _crawl_roots_for_validation <<< "$CRAWL_ROOTS"
    for media_root in "${_crawl_roots_for_validation[@]}"; do
      [ -n "$media_root" ] || continue
      [[ "$media_root" = /* ]] \
        || { echo "Every CRAWL_ROOTS entry must be absolute for a destructive reset: $media_root" >&2; exit 1; }
      MEDIA_ROOT_RESOLVED=$(resolve_path "$media_root")
      if [[ "$SIDECAR_ROOT_RESOLVED" == "$MEDIA_ROOT_RESOLVED" ||
            "$SIDECAR_ROOT_RESOLVED" == "$MEDIA_ROOT_RESOLVED/"* ||
            "$MEDIA_ROOT_RESOLVED" == "$SIDECAR_ROOT_RESOLVED/"* ]]; then
        echo "MEMORIA_SIDECAR_ROOT must not overlap media root: $MEDIA_ROOT_RESOLVED" >&2
        exit 1
      fi
    done
  fi
elif [ -n "${CRAWL_ROOTS:-}" ]; then
  IFS=':' read -r -a _crawl_roots_for_validation <<< "$CRAWL_ROOTS"
  for media_root in "${_crawl_roots_for_validation[@]}"; do
    [ -n "$media_root" ] || continue
    [[ "$media_root" = /* ]] \
      || { echo "Every CRAWL_ROOTS entry must be absolute for a destructive reset: $media_root" >&2; exit 1; }
    MEDIA_ROOT_RESOLVED=$(resolve_path "$media_root")
    [ "$MEDIA_ROOT_RESOLVED" != "/" ] \
      || { echo "Refusing to scan / for sidecars; configure narrower CRAWL_ROOTS." >&2; exit 1; }
  done
fi

for required_bin in psql pg_dump; do
  command -v "$required_bin" >/dev/null 2>&1 \
    || { echo "$required_bin is required; refusing a reset without a database backup/preflight." >&2; exit 1; }
done
database_psql -v ON_ERROR_STOP=1 -tAc "SELECT 1" >/dev/null \
  || { echo "PostgreSQL reset target is not reachable; no state has been changed." >&2; exit 1; }

echo "clean_slate: will operate on DB='$DB_NAME' (host=$DB_HOST port=$DB_PORT user=$DB_USER) and Huey queue at $QUEUE_DB.
Backup files will be written to /tmp with timestamp $TS."
echo "Cache reset target: $CACHE_DIR_RESOLVED"
[ -n "$SIDECAR_ROOT_RESOLVED" ] && echo "Dedicated sidecar reset target: $SIDECAR_ROOT_RESOLVED"

if [ "$YES" = false ] && [ "$DRY_RUN" = false ]; then
  read -r -p "This will irreversibly delete queued tasks, cache files, DB rows, and sidecar JSONs. Continue? [y/N] " ans
  case "$ans" in
    [Yy]*) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

# 1) Stop only processes owned by start.sh's PID files. stop.sh performs the
# full SIGTERM -> five-second grace period -> SIGKILL sequence and never uses
# broad process-name matching that could terminate unrelated applications.
echo "Stopping PID-managed native services (if running)..."
run "$ROOT_DIR/stop.sh"

# 2) Backup and delete the Huey SQLite queue + scan index
if [ -f "$QUEUE_DB" ]; then
  run cp -- "$QUEUE_DB" "/tmp/memoria-queue-backup-$TS.db"
  run rm -f -- "$QUEUE_DB"
  echo "Huey queue backed up to /tmp/memoria-queue-backup-$TS.db and removed."
else
  echo "No Huey queue at $QUEUE_DB; skipping queue wipe."
fi

if [ -f "$SCAN_DB" ]; then
  run cp -- "$SCAN_DB" "/tmp/memoria-scan-backup-$TS.db"
  run rm -f -- "$SCAN_DB"
  echo "Scan index backed up to /tmp/memoria-scan-backup-$TS.db and removed."
else
  echo "No scan index at $SCAN_DB; skipping."
fi

# 3) Clear only the validated cache subtree. Cache contents are derived state
# and are intentionally not backed up; models, edits, trash, and source media
# are outside this exact target and remain untouched.
if [ -d "$CACHE_DIR_RESOLVED" ]; then
  echo "Clearing Memoria cache at $CACHE_DIR_RESOLVED"
  if [ "$DRY_RUN" = true ]; then
    printf '+ find %q -mindepth 1 -depth -delete\n' "$CACHE_DIR_RESOLVED"
  else
    find "$CACHE_DIR_RESOLVED" -mindepth 1 -depth -delete
  fi
else
  echo "No cache directory at $CACHE_DIR_RESOLVED; skipping."
fi

# 4) Backup Postgres DB (pg_dump)
echo "Backing up Postgres DB to /tmp/memoria-db-backup-$TS.dump (custom format)"
run database_dump -Fc -f "/tmp/memoria-db-backup-$TS.dump"

# 5) Clear every derived/indexing table. Truncating media cascades to most
# per-media rows, but it does not remove standalone face/duplicate clusters or
# processing_log rows. Leaving those behind makes a "clean" re-index report
# stale failures, people, and duplicate groups until each feature is rebuilt.
echo "Clearing indexed media and all derived processing/model state (CASCADE)."
RESET_SQL="
TRUNCATE TABLE
    processing_log,
    face_cluster_merges,
    face_embeddings,
    face_clusters,
    duplicate_members,
    duplicate_groups,
    media_edits,
    scene_embeddings,
    ml_processing_state,
    ml_model_benchmarks,
    ml_reindex_jobs,
    media
RESTART IDENTITY CASCADE;

UPDATE ml_model_state
SET active_model_key = desired_model_key,
    status = CASE
        WHEN desired_model_key IS NULL THEN 'disabled'
        ELSE 'ready'
    END,
    generation = generation + 1,
    last_error = NULL,
    updated_at = NOW();
"
run database_psql -v ON_ERROR_STOP=1 -c "$RESET_SQL"

# 6) Archive and delete sidecars. A dedicated external store is safe to clear
# recursively after validation. Legacy adjacent sidecars are limited strictly
# to explicit CRAWL_ROOTS; an unset value no longer implies a destructive HOME walk.
SIDECAR_BACKUP="/tmp/memoria_sidecars_backup-$TS.tar.gz"
if [ -n "$SIDECAR_ROOT_RESOLVED" ]; then
  if [ -d "$SIDECAR_ROOT_RESOLVED" ] && find "$SIDECAR_ROOT_RESOLVED" -mindepth 1 -print -quit | grep -q .; then
    echo "Archiving dedicated sidecar store to $SIDECAR_BACKUP"
    if [ "$DRY_RUN" = true ]; then
      printf '+ tar -czf %q -C %q .\n' "$SIDECAR_BACKUP" "$SIDECAR_ROOT_RESOLVED"
      printf '+ find %q -mindepth 1 -delete\n' "$SIDECAR_ROOT_RESOLVED"
    else
      tar -czf "$SIDECAR_BACKUP" -C "$SIDECAR_ROOT_RESOLVED" .
      find "$SIDECAR_ROOT_RESOLVED" -mindepth 1 -delete
    fi
  else
    echo "No sidecar state found under $SIDECAR_ROOT_RESOLVED"
  fi
elif [ -z "${CRAWL_ROOTS:-}" ]; then
  echo "CRAWL_ROOTS is unset; skipping legacy sidecar deletion rather than scanning all of HOME."
else
  SIDE_LIST="/tmp/memoria_sidecars_files_$TS.bin"
  : > "$SIDE_LIST"
  IFS=':' read -r -a CRAWL_ROOT_LIST <<< "$CRAWL_ROOTS"
  for crawl_root in "${CRAWL_ROOT_LIST[@]}"; do
    [ -n "$crawl_root" ] || continue
    CRAWL_ROOT_RESOLVED=$(resolve_path "$crawl_root")
    [ "$CRAWL_ROOT_RESOLVED" != "/" ] \
      || { echo "Refusing to scan / for sidecars; configure narrower CRAWL_ROOTS." >&2; exit 1; }
    if [ -d "$CRAWL_ROOT_RESOLVED" ]; then
      find "$CRAWL_ROOT_RESOLVED" -type f -name ".memoria_metadata.json" -print0 >> "$SIDE_LIST"
    else
      echo "Crawl root does not exist; skipping: $CRAWL_ROOT_RESOLVED"
    fi
  done

  if [ -s "$SIDE_LIST" ]; then
    echo "Archiving legacy sidecars from configured crawl roots to $SIDECAR_BACKUP"
    if [ "$DRY_RUN" = true ]; then
      printf '+ tar --null -czf %q -T %q\n' "$SIDECAR_BACKUP" "$SIDE_LIST"
      echo "+ delete only the .memoria_metadata.json files listed in $SIDE_LIST"
    else
      tar --null -czf "$SIDECAR_BACKUP" -T "$SIDE_LIST"
      while IFS= read -r -d '' sidecar_file; do
        rm -f -- "$sidecar_file"
      done < "$SIDE_LIST"
    fi
  else
    echo "No legacy sidecars found below configured CRAWL_ROOTS."
  fi
fi

# 7) Final status
echo "=== Final quick checks ==="
if command -v psql >/dev/null 2>&1; then
  echo "Postgres media count:"; database_psql -c "SELECT COUNT(*) FROM media;" || true
fi
if [ -f "$QUEUE_DB" ]; then
  echo "Huey queue still present (unexpected): $QUEUE_DB"
else
  echo "Huey queue removed."
fi

echo "clean_slate completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "Notes:"
echo " - Backups (if created) are in /tmp with prefixes 'memoria-queue-backup-', 'memoria-scan-backup-', 'memoria-db-backup-'."
echo " - Sidecars were archived to /tmp/memoria_sidecars_backup-${TS}.tar.gz (if any existed)."
echo " - Cache contents were removed from $CACHE_DIR_RESOLVED; models, edits, trash, and source media were preserved."
echo " - To restart services: run ./start.sh"
