#!/usr/bin/env bash
# start.sh – Start Memoria natively or through Docker Compose
# Usage: ./start.sh [--docker|-d] [--cpu|--gpu]
#
# In plain English: the single command that boots Memoria locally. It
# does four things in order:
#   1. Stops processes tracked by the previous run and refuses unsafe port
#      collisions with untracked processes.
#   2. Verifies PostgreSQL is reachable, runs `alembic upgrade head` to
#      apply any pending migrations, and checks the pgvector + uuid-ossp
#      extensions are enabled.
#   3. Launches the four service processes — uvicorn (API), huey_consumer
#      (worker), watcher, Next.js (frontend) — into the background,
#      tracking each one's PID in `.pids/<name>.pid` so stop.sh can
#      gracefully shut them down later.
#   4. Hits /health on the API and / on the frontend to confirm they
#      came up healthy.
# All four service logs land in `logs/<name>.log`.
#
# Env overrides (optional):
#   DB_NAME      – PostgreSQL database name  (default: memoria)
#   DB_HOST      – PostgreSQL host           (default: localhost)
#   DB_PORT      – PostgreSQL port           (default: 5432)
#   DB_USER      – PostgreSQL user           (default: current OS user)
#   DATABASE_URL – Full PostgreSQL URL. When set, the database must exist.
#   MEMORIA_FRONTEND_MODE – production (default) or development.
#   MEMORIA_API_PORT / MEMORIA_WEB_PORT – listening ports (8000 / 5004).
#   CRAWL_ROOTS  – Colon-separated list of absolute paths to index.
#                  When set, ONLY these directories are crawled/watched.
#                  Example: export CRAWL_ROOTS="/Users/you/Pictures:/Volumes/MyDrive"
#                  When unset, the entire home directory is crawled (slow on first run).

set -euo pipefail

# Keep native builds/dev runs aligned with the container's telemetry opt-out.
export NEXT_TELEMETRY_DISABLED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
PIDS_DIR="$ROOT_DIR/.pids"

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $0 [--docker|-d] [--cpu|--gpu]

Without flags, start the native API, single Huey worker, watcher, and
production Next.js server. With --docker/-d, start the Compose stack and
use a native-architecture CPU image by default. On a compatible Linux/NVIDIA
host, auto-detection enables compose.gpu.yaml. --cpu forces the portable image;
--gpu explicitly requests the NVIDIA image and pass-through.
EOF
}

DOCKER_MODE=false
DOCKER_ACCELERATOR=auto
while [ $# -gt 0 ]; do
    case "$1" in
        --docker|-d) DOCKER_MODE=true ;;
        --cpu)
            [ "$DOCKER_ACCELERATOR" != "nvidia" ] \
                || error "--cpu and --gpu are mutually exclusive."
            DOCKER_ACCELERATOR=cpu
            ;;
        --gpu|--nvidia)
            [ "$DOCKER_ACCELERATOR" != "cpu" ] \
                || error "--cpu and --gpu are mutually exclusive."
            DOCKER_ACCELERATOR=nvidia
            ;;
        --help|-h) usage; exit 0 ;;
        *) error "Unknown argument: $1 (use --help for usage)." ;;
    esac
    shift
done

if [ "$DOCKER_MODE" = false ] && [ "$DOCKER_ACCELERATOR" != "auto" ]; then
    error "--cpu/--gpu select a Docker image and require --docker."
fi

if [ "$DOCKER_MODE" = true ]; then
    command -v docker >/dev/null 2>&1 \
        || error "docker is not in PATH - install Docker Engine and Compose, then retry."
    docker compose version >/dev/null 2>&1 \
        || error "Docker Compose v2 is unavailable - install the docker compose plugin."
    docker info >/dev/null 2>&1 \
        || error "Docker Engine is installed but its daemon is not reachable."

    COMPOSE=(docker compose --project-directory "$ROOT_DIR" -f "$ROOT_DIR/compose.yaml")
    GPU_ENABLED=false
    DOCKER_SERVER_OS=$(docker info --format '{{.OSType}}' 2>/dev/null || true)
    DOCKER_SERVER_ARCH=$(docker info --format '{{.Architecture}}' 2>/dev/null || true)

    case "$DOCKER_ACCELERATOR" in
        cpu)
            info "CPU image explicitly selected."
            ;;
        nvidia)
            [ "$DOCKER_SERVER_OS" = "linux" ] \
                || error "--gpu requires a Linux Docker engine (reported: ${DOCKER_SERVER_OS:-unknown})."
            case "$DOCKER_SERVER_ARCH" in
                x86_64|amd64) ;;
                *) error "--gpu currently requires an x86_64 Docker engine (reported: ${DOCKER_SERVER_ARCH:-unknown})." ;;
            esac
            command -v nvidia-smi >/dev/null 2>&1 \
                && nvidia-smi -L >/dev/null 2>&1 \
                || error "--gpu was requested, but nvidia-smi cannot see an NVIDIA GPU."
            GPU_ENABLED=true
            ;;
        auto)
            if [ "$DOCKER_SERVER_OS" = "linux" ] \
                && { [ "$DOCKER_SERVER_ARCH" = "x86_64" ] || [ "$DOCKER_SERVER_ARCH" = "amd64" ]; } \
                && command -v nvidia-smi >/dev/null 2>&1 \
                && nvidia-smi -L >/dev/null 2>&1; then
                GPU_ENABLED=true
                info "NVIDIA GPU detected; enabling the opt-in CUDA image."
            elif [ "$(uname -s)" = "Darwin" ]; then
                info "macOS Docker detected; using the native-architecture CPU image (no CUDA layers)."
            else
                info "No compatible NVIDIA runtime detected; using the CPU image."
            fi
            ;;
    esac

    if [ "$GPU_ENABLED" = true ]; then
        COMPOSE+=(-f "$ROOT_DIR/compose.gpu.yaml")
    fi

    # The default bind must exist because long-form bind mounts do not create
    # missing host paths consistently across Docker Engine/Desktop versions.
    mkdir -p "$ROOT_DIR/media" \
        || error "Could not create the default media directory at $ROOT_DIR/media."

    "${COMPOSE[@]}" config --quiet \
        || error "Docker Compose configuration is invalid; check .env paths and values."

    # Compose parses .env itself in Docker mode, so paths containing spaces do
    # not have to obey Bash's source syntax. Ask Compose for the few wrapper
    # settings we need rather than evaluating the whole dotenv file as shell.
    # Validate host ports before a potentially long image build.
    _compose_environment_value() {
        "${COMPOSE[@]}" config --environment 2>/dev/null \
            | sed -n "s/^$1=//p" \
            | tail -1
    }
    CONFIGURED_COMPOSE_WAIT=$(_compose_environment_value MEMORIA_COMPOSE_WAIT_SECONDS)
    CONFIGURED_API_PORT=$(_compose_environment_value MEMORIA_API_PORT)
    CONFIGURED_WEB_PORT=$(_compose_environment_value MEMORIA_WEB_PORT)
    COMPOSE_WAIT_SECONDS="${MEMORIA_COMPOSE_WAIT_SECONDS:-${CONFIGURED_COMPOSE_WAIT:-300}}"
    [[ "$COMPOSE_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
        || error "MEMORIA_COMPOSE_WAIT_SECONDS must be a positive integer number of seconds."

    DOCKER_API_PORT="${MEMORIA_API_PORT:-${CONFIGURED_API_PORT:-8000}}"
    DOCKER_WEB_PORT="${MEMORIA_WEB_PORT:-${CONFIGURED_WEB_PORT:-5004}}"
    RUNNING_APP_ID=$("${COMPOSE[@]}" ps --status running -q app 2>/dev/null || true)
    for port_spec in "API:$DOCKER_API_PORT" "web:$DOCKER_WEB_PORT"; do
        port_label=${port_spec%%:*}
        host_port=${port_spec#*:}
        case "$port_label" in
            API) port_variable=MEMORIA_API_PORT ;;
            web) port_variable=MEMORIA_WEB_PORT ;;
        esac
        [[ "$host_port" =~ ^[1-9][0-9]*$ ]] && [ "$host_port" -le 65535 ] \
            || error "The Compose $port_label host port must be an integer from 1 through 65535 (got '$host_port')."

        # A running app container legitimately owns its existing published
        # ports. Otherwise report host listeners before Compose spends minutes
        # building an image only to fail while creating the app container.
        if [ -z "$RUNNING_APP_ID" ] && command -v lsof >/dev/null 2>&1; then
            PORT_OWNERS=$(lsof -nP -tiTCP:"$host_port" -sTCP:LISTEN 2>/dev/null \
                | paste -sd, - || true)
            [ -z "$PORT_OWNERS" ] \
                || error "Compose $port_label host port $host_port is already in use by PID(s) $PORT_OWNERS. Stop that process or set $port_variable to a free port in .env, then retry."
        fi
    done

    # Start PostgreSQL alone first. The application entrypoint also performs
    # its own connection retry, so Alembic is protected both outside and
    # inside the container from a database that is still accepting startup.
    info "Starting the Compose PostgreSQL service..."
    "${COMPOSE[@]}" up -d db \
        || error "Could not start the Compose PostgreSQL service."

    info "Waiting for Compose PostgreSQL to accept authenticated connections..."
    DB_READY=false
    _attempt=0
    while [ "$_attempt" -lt 60 ]; do
        if "${COMPOSE[@]}" exec -T db sh -ec \
            'PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -qAt \
                -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1"' \
            >/dev/null 2>&1; then
            DB_READY=true
            break
        fi
        _attempt=$((_attempt + 1))
        sleep 2
    done
    [ "$DB_READY" = true ] \
        || error "Compose PostgreSQL authentication did not succeed within 120 seconds. Inspect 'docker compose logs db'; if DB credentials changed after the volume was created, restore the original values or run './clean_slate.sh --docker' after backing up."

    info "Building and starting Memoria; waiting for migrations and both HTTP services..."
    if ! "${COMPOSE[@]}" up -d --build --wait \
        --wait-timeout "$COMPOSE_WAIT_SECONDS"; then
        warn "Compose did not become healthy. Current service state:"
        "${COMPOSE[@]}" ps || true
        error "Docker Compose failed; inspect './start.sh --docker' output or run 'docker compose logs app db'."
    fi

    WEB_ENDPOINT=$("${COMPOSE[@]}" port app 5004 2>/dev/null | head -1 || true)
    [ -n "$WEB_ENDPOINT" ] || WEB_ENDPOINT="127.0.0.1:5004"
    if [ "$GPU_ENABLED" = true ]; then
        info "Memoria is healthy with NVIDIA pass-through at http://${WEB_ENDPOINT}."
    else
        info "Memoria is healthy with CPU inference at http://${WEB_ENDPOINT}."
    fi
    info "PostgreSQL, extensions, and Alembic schema were initialized automatically."
    info "Run './stop.sh --docker' to stop the stack."
    exit 0
fi

[ ! -f "$ROOT_DIR/.env" ] \
    || ! grep -Eq '^MEMORIA_HOST_MOUNT=[A-Za-z]:\\[[:space:]]*$' "$ROOT_DIR/.env" \
    || error ".env contains a trailing Windows backslash. Use forward slashes (for example MEMORIA_HOST_MOUNT=C:/) so the next line is not joined accidentally."
[ -f "$ROOT_DIR/.env" ] && set -a && source "$ROOT_DIR/.env" && set +a

DB_NAME="${DB_NAME:-memoria}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-$(whoami)}"
DATABASE_URL="${DATABASE_URL:-}"
DATABASE_URL_SUPPLIED=false
[ -n "$DATABASE_URL" ] && DATABASE_URL_SUPPLIED=true

MEMORIA_BIND_ADDRESS="${MEMORIA_BIND_ADDRESS:-127.0.0.1}"
MEMORIA_API_HOST="${MEMORIA_API_HOST:-$MEMORIA_BIND_ADDRESS}"
MEMORIA_API_PORT="${MEMORIA_API_PORT:-8000}"
MEMORIA_WEB_HOST="${MEMORIA_WEB_HOST:-$MEMORIA_BIND_ADDRESS}"
MEMORIA_WEB_PORT="${MEMORIA_WEB_PORT:-5004}"
MEMORIA_FRONTEND_MODE="${MEMORIA_FRONTEND_MODE:-production}"
MEMORIA_FRONTEND_REBUILD="${MEMORIA_FRONTEND_REBUILD:-auto}"

# HTTP clients cannot connect to wildcard bind addresses, and IPv6 literals
# need brackets when embedded in URLs. Concrete hostnames/addresses remain
# unchanged so custom native binds are probed through the interface that is
# actually listening.
_url_probe_host() {
    case "$1" in
        0.0.0.0) printf '%s' '127.0.0.1' ;;
        ::|\[::\]) printf '%s' '[::1]' ;;
        \[*\]) printf '%s' "$1" ;;
        *:*) printf '[%s]' "$1" ;;
        *) printf '%s' "$1" ;;
    esac
}

MEMORIA_API_PROBE_HOST="$(_url_probe_host "$MEMORIA_API_HOST")"
MEMORIA_WEB_PROBE_HOST="$(_url_probe_host "$MEMORIA_WEB_HOST")"
MEMORIA_API_PROBE_URL="http://${MEMORIA_API_PROBE_HOST}:${MEMORIA_API_PORT}"
MEMORIA_WEB_PROBE_URL="http://${MEMORIA_WEB_PROBE_HOST}:${MEMORIA_WEB_PORT}"

case "$MEMORIA_FRONTEND_MODE" in
    production|development) ;;
    *) echo "[ERROR] MEMORIA_FRONTEND_MODE must be 'production' or 'development'." >&2; exit 1 ;;
esac
case "$MEMORIA_FRONTEND_REBUILD" in
    auto|always|never) ;;
    *) echo "[ERROR] MEMORIA_FRONTEND_REBUILD must be 'auto', 'always', or 'never'." >&2; exit 1 ;;
esac

# -- 0. Resolve configuration and preserve process ownership ----------------
# Managed PID files below are the authority. Never kill unrelated processes by
# a broad command-line match; an untracked port occupant is reported instead.
LOG_DIR="$ROOT_DIR/logs"

# If DB_PASSWORD is set, export it as PGPASSWORD so psql/createdb/pg_isready pick it up
[ -n "${DB_PASSWORD:-}" ] && export PGPASSWORD="$DB_PASSWORD"
# Ensure CRAWL_ROOTS is exported so watcher/worker subprocesses inherit it
export CRAWL_ROOTS="${CRAWL_ROOTS:-}"
# Used by the Next.js same-origin proxy. Container builds supply their own
# supervised loopback address; native mode must use a host the API bind can
# actually receive traffic on.
export MEMORIA_INTERNAL_API_URL="${MEMORIA_INTERNAL_API_URL:-$MEMORIA_API_PROBE_URL}"

VENV_DIR="$BACKEND_DIR/venv"

mkdir -p "$PIDS_DIR" "$LOG_DIR"

# ── 0. Stop any previously tracked processes ─────────────────────────────────
# Running start.sh a second time without stop.sh would spawn duplicate services.
# Each PID file also stores the process birth time. A numeric PID alone is not
# safe because operating systems reuse it after a process exits.
_process_identity() {
    ps -p "$1" -o lstart= 2>/dev/null \
        | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

_write_pid_file() {
    local name="$1"
    local pid="$2"
    local identity
    identity=$(_process_identity "$pid")
    [ -n "$identity" ] || identity="unavailable"
    printf '%s\n%s\n' "$pid" "$identity" > "$PIDS_DIR/${name}.pid"
}

_kill_pid_file() {
    local name="$1"
    local pid_file="$PIDS_DIR/${name}.pid"
    [ -f "$pid_file" ] || return 0
    local pid stored_identity actual_identity
    IFS= read -r pid < "$pid_file" || pid=""
    stored_identity=$(sed -n '2p' "$pid_file")
    if ! [[ "$pid" =~ ^[0-9]+$ ]] || [ "$pid" -le 1 ]; then
        warn "Ignoring invalid PID file: $pid_file"
        rm -f -- "$pid_file"
        return 0
    fi
    if kill -0 "$pid" 2>/dev/null; then
        actual_identity=$(_process_identity "$pid")
        if [ -z "$stored_identity" ] || [ "$stored_identity" = "unavailable" ] \
            || [ "$actual_identity" != "$stored_identity" ]; then
            warn "Refusing to signal PID $pid from stale or legacy $pid_file; process identity does not match."
            rm -f -- "$pid_file"
            return 0
        fi
        info "Stopping previous $name (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 5 ]; do
            sleep 1; waited=$((waited + 1))
        done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
}
for svc in watcher huey uvicorn frontend; do
    _kill_pid_file "$svc"
done

# Refuse to take over a port owned by an untracked process (for example an
# orphaned service or a frontend started outside this script).
for port in "$MEMORIA_API_PORT" "$MEMORIA_WEB_PORT"; do
    squatter=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    if [ -n "$squatter" ]; then
        error "Port $port is occupied by untracked PID(s) $squatter. Stop that process or choose a different MEMORIA_*_PORT."
    fi
done

# ── PostgreSQL binary discovery ───────────────────────────────────────────────
# pg_isready / psql / createdb may live outside $PATH depending on how Postgres
# was installed.  Check the two common non-standard locations:
#   • macOS Postgres.app: /Applications/Postgres.app/Contents/Versions/<ver>/bin
#   • Versioned Linux PostgreSQL installs: /usr/pgsql-<ver>/bin
if ! command -v pg_isready &>/dev/null; then
    # macOS Postgres.app
    PGAPP_BASE="/Applications/Postgres.app/Contents/Versions"
    if [ -d "$PGAPP_BASE" ]; then
        PGAPP_VER=$(ls -1 "$PGAPP_BASE" | grep -E '^[0-9]+' | sort -rn | head -1)
        if [ -n "$PGAPP_VER" ] && [ -d "$PGAPP_BASE/$PGAPP_VER/bin" ]; then
            export PATH="$PGAPP_BASE/$PGAPP_VER/bin:$PATH"
            info "Detected Postgres.app v$PGAPP_VER – added its bin dir to PATH."
        fi
    fi
    # Linux RPM installs (/usr/pgsql-<ver>/bin)
    if ! command -v pg_isready &>/dev/null; then
        PG_RPM_BIN=$(ls -1d /usr/pgsql-*/bin 2>/dev/null | sort -t- -k2 -rn | head -1)
        if [ -n "$PG_RPM_BIN" ]; then
            export PATH="$PG_RPM_BIN:$PATH"
            info "Detected Linux PostgreSQL at $PG_RPM_BIN – added to PATH."
        fi
    fi
fi

# ── 1. Preflight checks ───────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 \
    || error "python3 is not in PATH - install Python 3.10 through 3.12 and retry."
python3 -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' \
    || error "Python 3.10 through 3.12 is required."

NATIVE_ML_PROFILE=cpu
if [ "$(uname -s)" = "Linux" ] \
    && { [ "$(uname -m)" = "x86_64" ] || [ "$(uname -m)" = "amd64" ]; } \
    && command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi -L >/dev/null 2>&1; then
    NATIVE_ML_PROFILE=gpu
fi
NATIVE_REQUIREMENTS="$BACKEND_DIR/requirements-${NATIVE_ML_PROFILE}.txt"
MANAGED_REQUIREMENTS="$BACKEND_DIR/requirements-managed-no-deps.txt"
DEPENDENCY_SMOKE="$BACKEND_DIR/dependency_smoke.py"
[ -f "$NATIVE_REQUIREMENTS" ] \
    || error "Missing dependency profile: $NATIVE_REQUIREMENTS"
[ -f "$MANAGED_REQUIREMENTS" ] \
    || error "Missing managed dependency list: $MANAGED_REQUIREMENTS"
[ -f "$DEPENDENCY_SMOKE" ] \
    || error "Missing dependency smoke check: $DEPENDENCY_SMOKE"

info "Checking Python venv at $VENV_DIR..."
DEPS_PROFILE_FILE="$VENV_DIR/.memoria-requirements-profile"
NEEDS_BACKEND_INSTALL=false
REBUILD_BACKEND_VENV=false

# A venv is not portable across Python interpreter replacements, and changing
# dependency profiles can leave orphaned nvidia-* distributions behind even
# after torch itself is uninstalled. Recreate those environments from the
# already-validated host python instead of trying to repair package metadata in
# place. Requirements-only changes below remain an incremental reinstall.
if [ ! -d "$VENV_DIR" ]; then
    NEEDS_BACKEND_INSTALL=true
    REBUILD_BACKEND_VENV=true
elif [ ! -x "$VENV_DIR/bin/python" ] || [ ! -x "$VENV_DIR/bin/pip" ]; then
    NEEDS_BACKEND_INSTALL=true
    REBUILD_BACKEND_VENV=true
elif ! "$VENV_DIR/bin/python" -c \
    'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' \
    >/dev/null 2>&1 \
    || ! "$VENV_DIR/bin/pip" --version >/dev/null 2>&1; then
    NEEDS_BACKEND_INSTALL=true
    REBUILD_BACKEND_VENV=true
elif [ ! -f "$DEPS_PROFILE_FILE" ] \
    || [ "$(cat "$DEPS_PROFILE_FILE" 2>/dev/null || true)" != "$NATIVE_ML_PROFILE" ]; then
    NEEDS_BACKEND_INSTALL=true
    REBUILD_BACKEND_VENV=true
elif ! "$VENV_DIR/bin/python" "$DEPENDENCY_SMOKE" "$NATIVE_ML_PROFILE" \
    >/dev/null 2>&1; then
    NEEDS_BACKEND_INSTALL=true
    REBUILD_BACKEND_VENV=true
elif [ ! -x "$VENV_DIR/bin/uvicorn" ]; then
    NEEDS_BACKEND_INSTALL=true
elif find "$BACKEND_DIR" -maxdepth 1 -type f -name 'requirements*.txt' \
    -newer "$DEPS_PROFILE_FILE" -print -quit | grep -q .; then
    NEEDS_BACKEND_INSTALL=true
fi

if [ "$NEEDS_BACKEND_INSTALL" = true ]; then
    info "Installing backend dependencies (${NATIVE_ML_PROFILE} profile)..."
    if [ "$NATIVE_ML_PROFILE" = "cpu" ] && [ "$(uname -s)" = "Darwin" ]; then
        info "macOS uses CPU ONNX Runtime; native PyTorch may still use Apple MPS."
    fi
    if [ "$REBUILD_BACKEND_VENV" = true ]; then
        info "Rebuilding the backend venv from the supported host Python..."
        python3 -m venv --clear "$VENV_DIR" \
            || error "Could not rebuild Python venv at $VENV_DIR."
    fi
    "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel \
        || error "Could not update pip/setuptools in the backend venv."
    # Remove distributions that share the cv2/onnxruntime namespaces. The two
    # face packages are reinstalled without dependencies below because their
    # published dependency metadata conflicts with the tested image stack.
    "$VENV_DIR/bin/pip" uninstall -y \
        facenet-pytorch insightface \
        albumentations albucore \
        opencv-python opencv-python-headless opencv-contrib-python \
        opencv-contrib-python-headless \
        onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
    "$VENV_DIR/bin/pip" install -r "$NATIVE_REQUIREMENTS" \
        || error "pip install failed for $NATIVE_REQUIREMENTS"
    "$VENV_DIR/bin/pip" install --no-deps -r "$MANAGED_REQUIREMENTS" \
        || error "pip install failed for $MANAGED_REQUIREMENTS"
    "$VENV_DIR/bin/python" "$DEPENDENCY_SMOKE" "$NATIVE_ML_PROFILE" \
        || error "The ${NATIVE_ML_PROFILE} dependency smoke check failed."
    printf '%s\n' "$NATIVE_ML_PROFILE" > "$DEPS_PROFILE_FILE"
    info "Venv ready."
fi

PYTHON="$VENV_DIR/bin/python"
UVICORN="$VENV_DIR/bin/uvicorn"
HUEY_CONSUMER="$VENV_DIR/bin/huey_consumer"
ALEMBIC="$VENV_DIR/bin/alembic"

for bin in "$PYTHON" "$UVICORN" "$HUEY_CONSUMER" "$ALEMBIC"; do
    [ -x "$bin" ] || error "Missing executable: $bin – re-run pip install -r backend/requirements.txt"
done

# Keep PostgreSQL CLI preflight and migrations on the same connection. A full
# URL takes precedence over DB_*; component variables remain the native-mode
# default and are also what Compose maps into its internal URL.
if [ -n "$DATABASE_URL" ]; then
    _DB_URL_PARTS=$("$PYTHON" -c '
import sys
from urllib.parse import unquote, urlparse
u = urlparse(sys.argv[1])
if u.scheme not in ("postgres", "postgresql") or not u.hostname or not u.path.lstrip("/"):
    raise SystemExit(2)
parts = (u.hostname, str(u.port or 5432), unquote(u.username or ""), unquote(u.password or ""), unquote(u.path.lstrip("/")))
print("\x1f".join(parts))
' "$DATABASE_URL") || error "DATABASE_URL must be a PostgreSQL URL containing a host and database name."
    IFS=$'\x1f' read -r DB_HOST DB_PORT DB_USER DB_PASSWORD DB_NAME <<< "$_DB_URL_PARTS"
    [ -n "$DB_USER" ] || error "DATABASE_URL must include a PostgreSQL user."
    if [ -n "$DB_PASSWORD" ]; then
        export PGPASSWORD="$DB_PASSWORD"
    else
        unset PGPASSWORD
    fi
fi

# ── 2. PostgreSQL reachability ────────────────────────────────────────────────
command -v pg_isready &>/dev/null || error "pg_isready is not in PATH - install PostgreSQL client tools and re-run."
command -v psql &>/dev/null || error "psql is not in PATH - install PostgreSQL client tools and re-run."
if [ "$DATABASE_URL_SUPPLIED" = false ]; then
    command -v createdb &>/dev/null || error "createdb is not in PATH - install PostgreSQL client tools and re-run."
fi

DB_WAIT_SECONDS="${MEMORIA_DB_WAIT_SECONDS:-30}"
[[ "$DB_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
    || error "MEMORIA_DB_WAIT_SECONDS must be a positive integer."

_database_is_ready() {
    if [ "$DATABASE_URL_SUPPLIED" = true ]; then
        pg_isready --dbname="$DATABASE_URL" -q
    else
        pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -q
    fi
}

info "Waiting up to ${DB_WAIT_SECONDS}s for PostgreSQL at $DB_HOST:$DB_PORT..."
DB_READY=false
DB_DEADLINE=$((SECONDS + DB_WAIT_SECONDS))
while [ "$SECONDS" -lt "$DB_DEADLINE" ]; do
    if _database_is_ready; then
        DB_READY=true
        break
    fi
    sleep 1
done
[ "$DB_READY" = true ] \
    || error "PostgreSQL is not reachable at $DB_HOST:$DB_PORT. Start the configured PostgreSQL service and retry."

# ── 3. Database + migrations ─────────────────────────────────────────────────
info "Checking database '$DB_NAME'..."
if [ "$DATABASE_URL_SUPPLIED" = true ]; then
    psql --dbname="$DATABASE_URL" -v ON_ERROR_STOP=1 -tAc "SELECT 1" >/dev/null \
        || error "DATABASE_URL is reachable but its database/login is invalid. Create the database first or fix the URL."
else
    DB_EXISTS=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt \
        | cut -d'|' -f1 | tr -d ' ' | grep -xF "$DB_NAME" || true)

    if [ -z "$DB_EXISTS" ]; then
        info "Database '$DB_NAME' not found – creating..."
        createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
    fi
fi

_psql_target() {
    if [ "$DATABASE_URL_SUPPLIED" = true ]; then
        psql --dbname="$DATABASE_URL" "$@"
    else
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" "$@"
    fi
}

# Vector-dependent migrations require both extensions on a clean database.
# Fail before Alembic with an actionable message when the package or database
# privileges are missing.
info "Ensuring PostgreSQL extensions are enabled..."
for ext in "uuid-ossp" "vector"; do
    _psql_target -v ON_ERROR_STOP=1 \
        -c "CREATE EXTENSION IF NOT EXISTS \"$ext\";" >/dev/null \
        || error "Cannot enable PostgreSQL extension '$ext'. Install its OS package and grant $DB_USER permission to CREATE EXTENSION."
done

# Alembic migrations are idempotent and always run on startup. The first
# run on a freshly-created DB applies everything; subsequent runs are fast
# no-ops. All migrations live in backend/alembic/versions/.
info "Running Alembic migrations (alembic upgrade head)..."
if [ -z "$DATABASE_URL" ]; then
    _DB_URL_USER=$("$PYTHON" -c 'import sys; from urllib.parse import quote; print(quote(sys.argv[1], safe=""))' "$DB_USER")
    _DB_URL_NAME=$("$PYTHON" -c 'import sys; from urllib.parse import quote; print(quote(sys.argv[1], safe=""))' "$DB_NAME")
    _DB_URL_AUTH="$_DB_URL_USER"
    if [ -n "${DB_PASSWORD:-}" ]; then
        _DB_URL_PASSWORD=$("$PYTHON" -c 'import sys; from urllib.parse import quote; print(quote(sys.argv[1], safe=""))' "$DB_PASSWORD")
        _DB_URL_AUTH="${_DB_URL_AUTH}:${_DB_URL_PASSWORD}"
    fi
    _DB_URL_HOST="$DB_HOST"
    [[ "$_DB_URL_HOST" == *:* ]] && _DB_URL_HOST="[$_DB_URL_HOST]"
    DATABASE_URL="postgresql://${_DB_URL_AUTH}@${_DB_URL_HOST}:${DB_PORT}/${_DB_URL_NAME}"
fi
export DATABASE_URL
(cd "$BACKEND_DIR" && "$ALEMBIC" upgrade head) \
    || error "Alembic migration failed – check DATABASE_URL, extensions, or backend/alembic/versions/."
MEMORIA_SCHEMA_VALIDATE_ONLY=1 "$PYTHON" "$ROOT_DIR/docker/bootstrap.py" \
    || error "Alembic reached head, but required tables, vector types, constraints, or indexes are incomplete."
info "Schema is at head and passed catalog postcondition checks."

# ── 4. PostgreSQL extension checks ────────────────────────────────────────────
info "Checking PostgreSQL extensions..."
for ext in "uuid-ossp" "vector"; do
    INSTALLED=$(_psql_target -tAc \
        "SELECT COUNT(*) FROM pg_extension WHERE extname = '$ext';" 2>/dev/null || echo "0")
    if [ "$INSTALLED" -eq "0" ]; then
        warn "Extension '$ext' is NOT enabled in database '$DB_NAME'."
        warn "  Install the OS package and run: psql -d $DB_NAME -c 'CREATE EXTENSION IF NOT EXISTS \"$ext\";'"
    else
        info "Extension '$ext' is enabled."
    fi
done

# ── 5. Node / npm checks + frontend node_modules ─────────────────────────────
command -v curl &>/dev/null || error "curl is not in PATH – install it and re-run."
command -v node &>/dev/null || error "node is not in PATH – install Node.js 20+ and re-run."
command -v npm  &>/dev/null || error "npm is not in PATH – install Node.js 20+ and re-run."

NODE_MAJOR=$(node -e "console.log(parseInt(process.version.slice(1).split('.')[0]))" 2>/dev/null)
if [ -n "$NODE_MAJOR" ] && [ "$NODE_MAJOR" -lt 20 ]; then
    error "Node.js $NODE_MAJOR detected – this project requires Node 20 or later."
fi
info "Node.js $(node --version) / npm $(npm --version) detected."

if [ ! -f "$FRONTEND_DIR/node_modules/.package-lock.json" ] || \
   [ "$FRONTEND_DIR/package-lock.json" -nt "$FRONTEND_DIR/node_modules/.package-lock.json" ]; then
    info "Frontend dependencies are missing or stale – running reproducible npm install..."
    (cd "$FRONTEND_DIR" && npm ci --include=dev) || error "npm ci failed – check frontend/package-lock.json."
fi

NEXT_BIN="$FRONTEND_DIR/node_modules/.bin/next"
[ -x "$NEXT_BIN" ] || error "Next.js executable is missing – run npm ci inside frontend/."

# Production is the normal host runtime. Rebuild only when source/config or a
# build-time public setting changed. Override with MEMORIA_FRONTEND_REBUILD.
if [ "$MEMORIA_FRONTEND_MODE" = "production" ]; then
    BUILD_ID="$FRONTEND_DIR/.next/BUILD_ID"
    BUILD_ENV_FILE="$FRONTEND_DIR/.next/.memoria-build-env"
    BUILD_ENV_VALUE="NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL:-}|MEMORIA_INTERNAL_API_URL=${MEMORIA_INTERNAL_API_URL}"
    NEEDS_BUILD=false

    if [ ! -f "$BUILD_ID" ]; then
        NEEDS_BUILD=true
    elif [ "$MEMORIA_FRONTEND_REBUILD" = "always" ]; then
        NEEDS_BUILD=true
    elif [ "$MEMORIA_FRONTEND_REBUILD" = "auto" ]; then
        if find "$FRONTEND_DIR/src" "$FRONTEND_DIR/public" -type f -newer "$BUILD_ID" -print -quit | grep -q .; then
            NEEDS_BUILD=true
        fi
        for build_input in package.json package-lock.json next.config.ts tsconfig.json postcss.config.mjs; do
            if [ -f "$FRONTEND_DIR/$build_input" ] && [ "$FRONTEND_DIR/$build_input" -nt "$BUILD_ID" ]; then
                NEEDS_BUILD=true
            fi
        done
        if [ ! -f "$BUILD_ENV_FILE" ] || [ "$(cat "$BUILD_ENV_FILE")" != "$BUILD_ENV_VALUE" ]; then
            NEEDS_BUILD=true
        fi
    fi

    if [ "$NEEDS_BUILD" = true ]; then
        [ "$MEMORIA_FRONTEND_REBUILD" != "never" ] \
            || error "No production frontend build exists. Remove MEMORIA_FRONTEND_REBUILD=never or run 'npm run build' in frontend/."
        info "Building optimized Next.js production frontend..."
        (cd "$FRONTEND_DIR" && npm run build) \
            || error "Next.js production build failed – check the output above."
        printf '%s\n' "$BUILD_ENV_VALUE" > "$BUILD_ENV_FILE"
    else
        info "Using existing Next.js production build."
    fi

    # `output: standalone` is also used by the container image. Next.js omits
    # public/static assets from that server bundle, so stage them beside
    # server.js for the native production runtime as well.
    STANDALONE_DIR="$FRONTEND_DIR/.next/standalone"
    [ -f "$STANDALONE_DIR/server.js" ] \
        || error "Standalone Next.js server is missing – force a frontend rebuild."
    mkdir -p "$STANDALONE_DIR/public" "$STANDALONE_DIR/.next/static"
    cp -a "$FRONTEND_DIR/public/." "$STANDALONE_DIR/public/"
    cp -a "$FRONTEND_DIR/.next/static/." "$STANDALONE_DIR/.next/static/"
fi

# Validate the worker topology before launching any process. Previously this
# check happened after Uvicorn and the watcher were already running, leaving a
# partial deployment behind when the value was invalid.
_HUEY_WORKERS="${MEMORIA_HUEY_WORKERS:-1}"
[ "$_HUEY_WORKERS" = "1" ] \
    || error "MEMORIA_HUEY_WORKERS must remain 1 (single-worker memory guarantee)."

echo ""
info "Starting all Memoria services..."
echo "----------------------------------------------"

# ── 6. Uvicorn (FastAPI) ──────────────────────────────────────────────────────
info "Starting uvicorn → ${MEMORIA_API_PROBE_URL} (bind: ${MEMORIA_API_HOST})"
(cd "$BACKEND_DIR" && \
    "$UVICORN" main:app --host "$MEMORIA_API_HOST" --port "$MEMORIA_API_PORT" \
    >> "$LOG_DIR/uvicorn.log" 2>&1) &
UVICORN_PID=$!
_write_pid_file uvicorn "$UVICORN_PID"
info "  uvicorn PID: $UVICORN_PID  (log: logs/uvicorn.log)"

# ── 7. Huey consumer ──────────────────────────────────────────────────────────
# The single worker is a memory-safety invariant: ML model singletons must not
# be double-loaded, regardless of deployment resources or accelerator.
# Cap BLAS/OpenMP/torch thread pools so the worker doesn't spawn 200+ threads.
# Uncapped, PyTorch + NumPy + ONNX Runtime create O(N_CPU) threads each, which
# inflates RSS by several GB on a many-core machine.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# glibc malloc: use mmap for allocations >=32 MB so they get returned to the
# OS immediately when freed, instead of staying in the Python heap forever.
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-33554432}"
# Trim the brk heap back to the OS after freeing 128 MB of headroom.
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-134217728}"

# Portable safety baseline; every value remains environment-overridable.
export MEMORIA_BATCH_SIZE="${MEMORIA_BATCH_SIZE:-16}"
export MEMORIA_RAM_WARN_PCT="${MEMORIA_RAM_WARN_PCT:-0.85}"
export MEMORIA_RAM_CRIT_PCT="${MEMORIA_RAM_CRIT_PCT:-0.95}"
export MEMORIA_RAM_FLOOR_PCT="${MEMORIA_RAM_FLOOR_PCT:-0.90}"
export MEMORIA_RESERVED_MB="${MEMORIA_RESERVED_MB:-2048}"

info "Starting Huey consumer (workers=${_HUEY_WORKERS}, backend=sqlite)"
(cd "$BACKEND_DIR" && \
    "$HUEY_CONSUMER" workers.huey \
        -w "${_HUEY_WORKERS}" \
        -k thread \
        --logfile "$LOG_DIR/huey.log" \
        -v \
    >> "$LOG_DIR/huey.log" 2>&1) &
HUEY_PID=$!
_write_pid_file huey "$HUEY_PID"
info "  huey     PID: $HUEY_PID  (log: logs/huey.log)"

# ── 8. Watcher ────────────────────────────────────────────────────────────────
# Start the consumer first so a large initial crawl can drain immediately.
# Shutdown uses the reverse dependency order: watcher before consumer.
info "Starting watcher.py"
(cd "$BACKEND_DIR" && \
    "$PYTHON" watcher.py \
    >> "$LOG_DIR/watcher.log" 2>&1) &
WATCHER_PID=$!
_write_pid_file watcher "$WATCHER_PID"
info "  watcher  PID: $WATCHER_PID  (log: logs/watcher.log)"

# -- 9. Next.js production/development server ------------------------------
# Production is the default; development enables hot reload for frontend work.
info "Starting Next.js ${MEMORIA_FRONTEND_MODE} server → ${MEMORIA_WEB_PROBE_URL} (bind: ${MEMORIA_WEB_HOST})"
if [ "$MEMORIA_FRONTEND_MODE" = "production" ]; then
    (cd "$STANDALONE_DIR" && \
        exec env HOSTNAME="$MEMORIA_WEB_HOST" PORT="$MEMORIA_WEB_PORT" \
            node server.js \
        >> "$LOG_DIR/frontend.log" 2>&1) &
else
    (cd "$FRONTEND_DIR" && \
        exec "$NEXT_BIN" dev \
            --hostname "$MEMORIA_WEB_HOST" \
            --port "$MEMORIA_WEB_PORT" \
        >> "$LOG_DIR/frontend.log" 2>&1) &
fi
FRONTEND_PID=$!
_write_pid_file frontend "$FRONTEND_PID"
info "  frontend PID: $FRONTEND_PID  (log: logs/frontend.log)"

echo "----------------------------------------------"
echo ""

# ── 10. Health check — verify services respond ────────────────────────────────
info "Waiting for services to become healthy..."
HEALTH_OK=false
for i in $(seq 1 30); do
    if curl -sf --noproxy '*' "${MEMORIA_API_PROBE_URL}/health" > /dev/null 2>&1; then
        HEALTH_OK=true
        break
    fi
    sleep 1
done

if $HEALTH_OK; then
    info "✓ Backend (uvicorn) is healthy at ${MEMORIA_API_PROBE_URL}"
else
    warn "Backend health check failed after 30s — check logs/uvicorn.log"
fi

# Quick check that Next.js is responding
FRONTEND_OK=false
for i in $(seq 1 30); do
    if curl -sf --noproxy '*' "$MEMORIA_WEB_PROBE_URL" > /dev/null 2>&1; then
        FRONTEND_OK=true
        break
    fi
    sleep 1
done

if $FRONTEND_OK; then
    info "✓ Frontend (Next.js) is healthy at ${MEMORIA_WEB_PROBE_URL}"
else
    warn "Frontend health check failed after 30s — check logs/frontend.log"
fi

SERVICES_ALIVE=true
for service_pid in "$UVICORN_PID" "$WATCHER_PID" "$HUEY_PID" "$FRONTEND_PID"; do
    if ! kill -0 "$service_pid" 2>/dev/null; then
        SERVICES_ALIVE=false
    fi
done

if [ "$HEALTH_OK" != true ] || [ "$FRONTEND_OK" != true ] \
    || [ "$SERVICES_ALIVE" != true ]; then
    warn "Native startup was incomplete; stopping only the processes launched by this run."
    for svc in watcher huey uvicorn frontend; do
        _kill_pid_file "$svc"
    done
    error "Memoria did not start cleanly. Inspect the four files under logs/ and retry."
fi

echo ""
info "All services are healthy. Run ./stop.sh to stop them."
