#!/usr/bin/env bash
# Wait for Memoria's API-reported work queue to drain, then stop the deployment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STOP_SCRIPT="$ROOT_DIR/stop.sh"

API_BASE="${MEMORIA_API:-http://127.0.0.1:${MEMORIA_API_PORT:-8000}}"
TIMEOUT_SECONDS="${MEMORIA_DRAIN_TIMEOUT_SECONDS:-18000}"
INTERVAL_SECONDS="${MEMORIA_DRAIN_INTERVAL_SECONDS:-60}"
EMPTY_CHECKS_REQUIRED="${MEMORIA_DRAIN_EMPTY_CHECKS:-2}"
DOCKER_MODE=false
DRY_RUN=false

usage() {
  cat <<EOF
Usage: $0 [options]

Poll Memoria's /api/v1/queue/stats endpoint until queued, scheduled, and
running work remains empty for consecutive checks, then call the repo-local
stop.sh. The same stop is performed when the safety timeout expires.

Options:
  --api-url URL       API base URL (default: $API_BASE)
  --timeout SECONDS   Safety timeout (default: $TIMEOUT_SECONDS)
  --interval SECONDS  Poll interval (default: $INTERVAL_SECONDS)
  --empty-checks N    Consecutive empty polls before stopping
                      (default: $EMPTY_CHECKS_REQUIRED)
  --docker, -d        Pass --docker to stop.sh
  --dry-run           Monitor normally, but print rather than run stop.sh
  --help, -h          Show this help

Environment equivalents: MEMORIA_API, MEMORIA_API_PORT,
MEMORIA_DRAIN_TIMEOUT_SECONDS, MEMORIA_DRAIN_INTERVAL_SECONDS, and
MEMORIA_DRAIN_EMPTY_CHECKS.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --api-url)
      [ "$#" -ge 2 ] || { echo "--api-url requires a value" >&2; exit 2; }
      API_BASE="$2"
      shift 2
      ;;
    --timeout)
      [ "$#" -ge 2 ] || { echo "--timeout requires a value" >&2; exit 2; }
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --interval)
      [ "$#" -ge 2 ] || { echo "--interval requires a value" >&2; exit 2; }
      INTERVAL_SECONDS="$2"
      shift 2
      ;;
    --empty-checks)
      [ "$#" -ge 2 ] || { echo "--empty-checks requires a value" >&2; exit 2; }
      EMPTY_CHECKS_REQUIRED="$2"
      shift 2
      ;;
    --docker|-d)
      DOCKER_MODE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] \
    || { echo "$name must be a positive integer: $value" >&2; exit 2; }
}

require_positive_integer "--timeout" "$TIMEOUT_SECONDS"
require_positive_integer "--interval" "$INTERVAL_SECONDS"
require_positive_integer "--empty-checks" "$EMPTY_CHECKS_REQUIRED"

command -v curl >/dev/null 2>&1 \
  || { echo "curl is required to monitor the Memoria API" >&2; exit 1; }
[ -x "$STOP_SCRIPT" ] \
  || { echo "Repo-local stop script is not executable: $STOP_SCRIPT" >&2; exit 1; }

API_BASE="${API_BASE%/}"
QUEUE_URL="$API_BASE/api/v1/queue/stats"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

extract_count() {
  local key="$1"
  local payload="$2"
  local value
  value=$(printf '%s\n' "$payload" | sed -n \
    "s/.*\"$key\"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p")
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$value"
}

stop_memoria() {
  local reason="$1"
  log "$reason"
  if [ "$DRY_RUN" = true ]; then
    printf '+ %q' "$STOP_SCRIPT"
    [ "$DOCKER_MODE" = false ] || printf ' %q' --docker
    printf '\n'
  elif [ "$DOCKER_MODE" = true ]; then
    "$STOP_SCRIPT" --docker
  else
    "$STOP_SCRIPT"
  fi
}

STARTED_AT=$(date +%s)
DEADLINE=$((STARTED_AT + TIMEOUT_SECONDS))
EMPTY_CHECKS=0

log "Monitoring $QUEUE_URL every ${INTERVAL_SECONDS}s (timeout ${TIMEOUT_SECONDS}s)."

while true; do
  RESPONSE=""
  if RESPONSE=$(curl --fail --silent --show-error \
      --connect-timeout 5 --max-time 15 "$QUEUE_URL" 2>&1); then
    PENDING=$(extract_count pending "$RESPONSE" || true)
    SCHEDULED=$(extract_count scheduled "$RESPONSE" || true)
    RUNNING=$(extract_count running "$RESPONSE" || true)
    FAILED=$(extract_count failed "$RESPONSE" || true)

    # ``scheduled`` was added after the original queue endpoint. Treat an
    # absent field as zero so the monitor remains usable during rolling updates.
    [ -n "$SCHEDULED" ] || SCHEDULED=0
    [ -n "$FAILED" ] || FAILED=0

    if [ -z "$PENDING" ] || [ -z "$RUNNING" ]; then
      log "Queue API returned an unexpected payload; waiting: $RESPONSE"
      EMPTY_CHECKS=0
    else
      TOTAL=$((PENDING + SCHEDULED + RUNNING))
      log "Queue depth: $TOTAL (pending=$PENDING scheduled=$SCHEDULED running=$RUNNING failed=$FAILED)"
      if [ "$TOTAL" -eq 0 ]; then
        EMPTY_CHECKS=$((EMPTY_CHECKS + 1))
        if [ "$EMPTY_CHECKS" -ge "$EMPTY_CHECKS_REQUIRED" ]; then
          stop_memoria "Queue remained empty for $EMPTY_CHECKS consecutive checks; stopping Memoria."
          exit 0
        fi
      else
        EMPTY_CHECKS=0
      fi
    fi
  else
    RESPONSE=${RESPONSE//$'\n'/ }
    log "Queue API is unavailable; waiting: $RESPONSE"
    EMPTY_CHECKS=0
  fi

  NOW=$(date +%s)
  if [ "$NOW" -ge "$DEADLINE" ]; then
    stop_memoria "Safety timeout reached; stopping Memoria."
    exit 0
  fi

  SLEEP_SECONDS=$INTERVAL_SECONDS
  REMAINING=$((DEADLINE - NOW))
  [ "$SLEEP_SECONDS" -le "$REMAINING" ] || SLEEP_SECONDS=$REMAINING
  sleep "$SLEEP_SECONDS"
done
