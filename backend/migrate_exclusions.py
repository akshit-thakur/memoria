"""One-time migration: purge media rows that fall under excluded directories.

In plain English: cleanup script. When the exclusion list in
``constants.py`` grows — say someone adds a new directory name that
should never have been indexed — old rows that *were* indexed before
the change remain in the database. This script walks the ``media`` table
and deletes any row whose ``file_path`` passes through one of the
currently-excluded directories.

Run once after upgrading to the new EXCLUDED_DIRS watcher logic:

    python backend/migrate_exclusions.py

Safe to run multiple times (idempotent).
"""

import os
import sys
import time
from pathlib import Path

import psycopg

sys.path.insert(0, os.path.dirname(__file__))
from constants import EXCLUDED_DIRS
from database_config import DATABASE_URL


def _path_contains_excluded(file_path: str) -> bool:
    """Return True if any component of file_path matches an excluded directory name."""
    return any(part in EXCLUDED_DIRS for part in Path(file_path).parts)


def run_migration() -> None:
    t0 = time.perf_counter()
    print("Connecting to database …")
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, file_path FROM media")
            rows = cur.fetchall()

    print(f"Scanning {len(rows)} rows …")
    ids_to_delete = [row[0] for row in rows if _path_contains_excluded(row[1])]

    if not ids_to_delete:
        elapsed = time.perf_counter() - t0
        print(f"No excluded-path records found. Nothing to do. (scanned {len(rows)} rows in {elapsed:.2f}s)")
        return

    print(f"Found {len(ids_to_delete)} records in excluded paths. Deleting …")
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM media WHERE id = ANY(%s)", (ids_to_delete,))
        conn.commit()

    elapsed = time.perf_counter() - t0
    rate = len(ids_to_delete) / elapsed if elapsed > 0 else 0.0
    print(
        f"migrate_exclusions done: scanned={len(rows)} deleted={len(ids_to_delete)} "
        f"in {elapsed:.2f}s = {rate:.1f} rows/sec."
    )


if __name__ == "__main__":
    run_migration()
