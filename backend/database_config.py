"""Consistent PostgreSQL URL resolution for every Memoria process."""

from __future__ import annotations

import getpass
import os
from urllib.parse import quote


def resolve_database_url() -> str:
    """Return DATABASE_URL or build one from the shared DB_* settings.

    Container startup always supplies DATABASE_URL. The component fallback is
    for native services and admin scripts, where the current OS user and an
    empty password match PostgreSQL's normal local-auth defaults.
    """

    configured = (os.getenv("DATABASE_URL") or "").strip()
    if configured:
        # Older providers still emit ``postgres://``. psycopg accepts that
        # alias, but SQLAlchemy 2/Alembic does not, so normalize it once for
        # every process before the URL crosses those layers.
        if configured.startswith("postgres://"):
            return "postgresql://" + configured[len("postgres://") :]
        return configured

    user = os.getenv("DB_USER") or getpass.getuser()
    password = os.getenv("DB_PASSWORD") or ""
    host = os.getenv("DB_HOST") or "localhost"
    port = os.getenv("DB_PORT") or "5432"
    database = os.getenv("DB_NAME") or "memoria"

    encoded_user = quote(user, safe="")
    encoded_database = quote(database, safe="")
    authority = encoded_user
    if password:
        authority += ":" + quote(password, safe="")
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        host = f"[{host}]"
    return f"postgresql://{authority}@{host}:{port}/{encoded_database}"


DATABASE_URL = resolve_database_url()
