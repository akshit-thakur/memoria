"""Re-queue media files that are on disk but not yet in Postgres.

In plain English: scans the filesystem, asks Postgres which file paths
already have rows, and enqueues a ``process_media`` task for every file
that exists on disk but is missing from the database. Use this after a
DB wipe, a schema migration, or to recover from a bug that lost rows.

Run once after the @huey.task pipeline decorator fix:
    cd backend && python requeue_missing.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import psycopg
from constants import EXCLUDED_DIR_SUFFIXES, EXCLUDED_DIRS, MEDIA_EXTENSIONS
from database_config import DATABASE_URL
from workers.tasks import process_media

CRAWL_ROOTS_ENV = os.getenv("CRAWL_ROOTS", "").strip()


def _is_excluded(path: str) -> bool:
    for part in Path(path).parts:
        if part in EXCLUDED_DIRS:
            return True
        if any(part.endswith(sfx) for sfx in EXCLUDED_DIR_SUFFIXES):
            return True
        if part.startswith("."):
            return True
    return False


def main() -> None:
    t0 = time.perf_counter()
    # Collect already-indexed paths.
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute("SELECT file_path FROM media").fetchall()
    indexed = {r[0] for r in rows}
    print(f"Already in Postgres: {len(indexed)}")

    # Determine roots to scan.
    if CRAWL_ROOTS_ENV:
        roots = [p.strip() for p in CRAWL_ROOTS_ENV.split(os.pathsep) if os.path.isdir(p.strip())]
    else:
        roots = [str(Path.home())]
    print(f"Scanning roots: {roots}")

    scanned = 0
    queued = 0
    scan_start = time.perf_counter()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = [
                d for d in dirnames
                if d not in EXCLUDED_DIRS
                and not any(d.endswith(sfx) for sfx in EXCLUDED_DIR_SUFFIXES)
                and not d.startswith(".")
            ]
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                scanned += 1
                if Path(full).suffix.lower() not in MEDIA_EXTENSIONS:
                    continue
                if _is_excluded(full):
                    continue
                if full in indexed:
                    continue
                process_media(full)
                queued += 1

    elapsed = time.perf_counter() - t0
    scan_elapsed = time.perf_counter() - scan_start
    scan_rate = scanned / scan_elapsed if scan_elapsed > 0 else 0.0
    print(
        f"requeue done: scanned={scanned} queued={queued} in {elapsed:.2f}s "
        f"(crawl {scan_rate:.0f} files/sec)."
    )


if __name__ == "__main__":
    main()
