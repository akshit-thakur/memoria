"""Alembic migration environment for Memoria.

In plain English: this file is the bootstrap script that Alembic runs
before applying any migration. Its job is to figure out *which* database
to connect to (from DATABASE_URL, DB_* env vars, or alembic.ini, in that
order of precedence) and hand the connection to Alembic.

Memoria's migrations are hand-written raw SQL wrapped in
``op.execute("…")`` calls — we don't keep SQLAlchemy ORM models, so
``target_metadata`` stays None and Alembic's autogenerate is disabled.

Migrations use `op.execute()` with raw SQL rather than SQLAlchemy model
metadata — we don't maintain SQLAlchemy ORM models in this codebase, and
the existing migrations are hand-written Postgres DDL. No autogenerate
target_metadata is configured.
"""

from __future__ import annotations

import os
import getpass
from logging.config import fileConfig
from urllib.parse import quote

from alembic import context
from sqlalchemy import engine_from_config, pool

# Alembic Config object.
config = context.config

# Resolve the connection URL from env. Order of precedence:
#   1. DATABASE_URL (full URL, used by FastAPI / workers / CLI tools)
#   2. DB_USER / DB_PASSWORD / DB_HOST / DB_PORT / DB_NAME (composed here)
#   3. alembic.ini's sqlalchemy.url fallback
#
# Whatever we end up with, force the `postgresql+psycopg://` prefix so
# SQLAlchemy uses psycopg v3 (which is what's in requirements.txt).
_db_url = os.getenv("DATABASE_URL")

if not _db_url:
    _user = os.getenv("DB_USER") or getpass.getuser()
    # Lifecycle scripts normally supply their resolved DATABASE_URL. This
    # fallback mirrors native PostgreSQL local-auth defaults.
    _password = os.getenv("DB_PASSWORD", "")
    _host = os.getenv("DB_HOST", "localhost")
    _port = os.getenv("DB_PORT", "5432")
    _name = os.getenv("DB_NAME", "memoria")
    _encoded_user = quote(_user, safe="")
    _encoded_name = quote(_name, safe="")
    _url_host = _host
    if ":" in _url_host and not (
        _url_host.startswith("[") and _url_host.endswith("]")
    ):
        _url_host = f"[{_url_host}]"
    if _password:
        _auth = f"{_encoded_user}:{quote(_password, safe='')}@"
    else:
        _auth = f"{_encoded_user}@"
    _db_url = f"postgresql://{_auth}{_url_host}:{_port}/{_encoded_name}"

if _db_url.startswith("postgres://"):
    _db_url = "postgresql://" + _db_url[len("postgres://") :]
if _db_url.startswith("postgresql://") and "+" not in _db_url.split("://", 1)[0]:
    _db_url = _db_url.replace("postgresql://", "postgresql+psycopg://", 1)

# Alembic stores main options through ConfigParser interpolation. Percent
# escapes in URL-encoded credentials (for example `%40` for `@`) therefore
# have to be doubled here; ConfigParser restores the original URL when read.
config.set_main_option("sqlalchemy.url", _db_url.replace("%", "%%"))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL to stdout, don't connect."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
