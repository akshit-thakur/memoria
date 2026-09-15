#!/bin/sh
set -eu

: "${DB_HOST:=db}"
: "${DB_PORT:=5432}"
: "${DB_NAME:=memoria}"
: "${DB_USER:=memoria}"
: "${DB_PASSWORD:?DB_PASSWORD must be set for Compose startup; use a unique value in .env}"
export DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD

if [ -z "${DATABASE_URL:-}" ]; then
    DATABASE_URL="$(python3 -c '
import os
from urllib.parse import quote

user = quote(os.environ["DB_USER"], safe="")
password = quote(os.environ["DB_PASSWORD"], safe="")
host = os.environ["DB_HOST"]
port = os.environ["DB_PORT"]
database = quote(os.environ["DB_NAME"], safe="")
print(f"postgresql://{user}:{password}@{host}:{port}/{database}")
')"
    export DATABASE_URL
fi

mkdir -p \
    "${MEMORIA_STATE_DIR:-/var/lib/memoria/state}" \
    "${MEMORIA_SIDECAR_ROOT:-/var/lib/memoria/sidecars}" \
    "$(dirname "${MEMORIA_SCAN_INDEX_DB:-/var/lib/memoria/sqlite/scan_index.db}")" \
    "$(dirname "${HUEY_DB:-/var/lib/memoria/sqlite/queue.db}")" \
    "${HOME:-/var/lib/memoria/home}" \
    /var/lib/memoria/cache \
    /var/lib/memoria/models

python3 /app/docker/bootstrap.py

exec /usr/bin/supervisord -c /app/docker/supervisord.conf
