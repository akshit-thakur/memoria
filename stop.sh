#!/usr/bin/env bash
# stop.sh – Stop native Memoria services and/or the Docker Compose stack
# Usage: ./stop.sh [--docker|-d]
#
# In plain English: the graceful shutdown counterpart to start.sh. For
# each service (watcher, huey, uvicorn, frontend) it reads the PID file
# in .pids/, sends SIGTERM, waits up to 5 seconds for the process to
# exit, then sends SIGKILL if it still hasn't gone. The PID file is
# removed in either case so the next start.sh begins from a clean slate.
#
# `kill -0 <pid>` is a no-op signal that returns success only if the
# process exists — we use it as a quick "is this PID alive?" check.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_DIR="$ROOT_DIR/.pids"

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $0 [--docker|-d]

Native PID files are always inspected and stopped with SIGTERM, a five-second
grace period, and SIGKILL as a last resort. Docker Compose is also brought down
when --docker/-d is supplied or Memoria Compose containers are detected.
EOF
}

DOCKER_REQUESTED=false
while [ $# -gt 0 ]; do
    case "$1" in
        --docker|-d) DOCKER_REQUESTED=true ;;
        --help|-h) usage; exit 0 ;;
        *) error "Unknown argument: $1 (use --help for usage)." ;;
    esac
    shift
done

COMPOSE=(docker compose --project-directory "$ROOT_DIR" -f "$ROOT_DIR/compose.yaml")
DOCKER_AVAILABLE=false
DOCKER_ACTIVE=false
if command -v docker >/dev/null 2>&1 \
    && docker compose version >/dev/null 2>&1 \
    && docker info >/dev/null 2>&1; then
    DOCKER_AVAILABLE=true
    COMPOSE_IDS=$("${COMPOSE[@]}" ps -a -q 2>/dev/null || true)
    [ -n "$COMPOSE_IDS" ] && DOCKER_ACTIVE=true
fi

if [ "$DOCKER_REQUESTED" = true ] && [ "$DOCKER_AVAILABLE" = false ]; then
    error "--docker was requested, but Docker Engine or Docker Compose v2 is unavailable."
fi

STOPPED=0
SKIPPED=0

process_identity() {
    ps -p "$1" -o lstart= 2>/dev/null \
        | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

stop_service() {
    local name="$1"
    local pid_file="$PIDS_DIR/${name}.pid"

    if [ ! -f "$pid_file" ]; then
        warn "No PID file found for '$name' – already stopped or never started."
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    local pid stored_identity actual_identity
    IFS= read -r pid < "$pid_file" || pid=""
    stored_identity=$(sed -n '2p' "$pid_file")

    if ! [[ "$pid" =~ ^[0-9]+$ ]] || [ "$pid" -le 1 ]; then
        warn "Invalid PID file for '$name'; removing it without sending a signal."
        rm -f -- "$pid_file"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    if kill -0 "$pid" 2>/dev/null; then
        actual_identity=$(process_identity "$pid")
        if [ -z "$stored_identity" ] || [ "$stored_identity" = "unavailable" ] \
            || [ "$actual_identity" != "$stored_identity" ]; then
            warn "PID $pid in '$pid_file' has a different process identity; refusing to signal a possibly unrelated process."
            rm -f -- "$pid_file"
            SKIPPED=$((SKIPPED + 1))
            return
        fi
        info "Stopping $name (PID $pid)..."
        kill "$pid"
        # Wait up to 5 s for the process to exit cleanly
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 5 ]; do
            sleep 1
            waited=$((waited + 1))
        done
        # Force-kill if still alive
        if kill -0 "$pid" 2>/dev/null; then
            warn "$name (PID $pid) did not exit cleanly – sending SIGKILL."
            kill -9 "$pid" 2>/dev/null || true
        fi
        STOPPED=$((STOPPED + 1))
    else
        warn "$name (PID $pid) is not running (stale PID file)."
        SKIPPED=$((SKIPPED + 1))
    fi

    rm -f -- "$pid_file"
}

echo ""
info "Stopping Memoria services..."
echo "----------------------------------------------"

stop_service watcher
stop_service huey
stop_service uvicorn
stop_service frontend

if [ "$DOCKER_REQUESTED" = true ] || [ "$DOCKER_ACTIVE" = true ]; then
    info "Stopping the Memoria Docker Compose stack..."
    if "${COMPOSE[@]}" down; then
        STOPPED=$((STOPPED + 1))
    else
        warn "Docker Compose could not fully stop the Memoria stack."
        SKIPPED=$((SKIPPED + 1))
    fi
fi

echo "----------------------------------------------"
info "Done. Stopped: $STOPPED  Skipped/already-dead: $SKIPPED"
echo ""
