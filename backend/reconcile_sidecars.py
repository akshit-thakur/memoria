#!/usr/bin/env python3
"""Memoria – Sidecar Reconciliation Script (Spec 02 C1).

In plain English: the recovery half of the two-phase sync. The worker
writes ``.memoria_metadata.json`` sidecars through the configured store and marks each
entry as ``"pending"`` before talking to Postgres, then flips them to
``"synced"`` once the DB upsert returns. If the worker dies between the
two — power loss, crash, OOM — the sidecar is left in ``"pending"`` (or
``"failed"`` if the DB call raised). This script walks the filesystem,
finds every such sidecar, and replays the DB upsert so the database
catches up.

Run with ``--dry-run`` first to see what would change.

Walks configured media roots and resolves each directory through the shared
sidecar abstraction. With ``MEMORIA_SIDECAR_ROOT`` this reads the mirrored
persistent store instead of trying to write into a read-only library.

Usage:
    python backend/reconcile_sidecars.py --dry-run   # preview
    python backend/reconcile_sidecars.py             # apply
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import psycopg

sys.path.insert(0, os.path.dirname(__file__))
from constants import is_excluded_path
from database_config import DATABASE_URL
from sidecar import (
    get_lock as sidecar_lock,
    load_sidecar,
    save_sidecar,
    sidecar_path,
)

CRAWL_ROOTS_ENV = os.getenv("CRAWL_ROOTS", "").strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("memoria.reconcile")


def _walk_sidecar_directories(roots: list[str]):
    """Yield source directories that have a sidecar in the selected store."""
    for root in roots:
        for dirpath, dirnames, _filenames in os.walk(root, topdown=True):
            dirnames[:] = [
                d for d in dirnames
                if not is_excluded_path(os.path.join(dirpath, d))
            ]
            path = sidecar_path(dirpath)
            if path.is_file():
                yield dirpath


def _upsert_to_postgres(conn, file_path: str, entry: dict):
    """Upsert a single sidecar entry to the media table."""
    p = Path(file_path)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO media (file_path, filename, directory, extension,
                               phash, blur_score, exif_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_path) DO UPDATE SET
                phash       = EXCLUDED.phash,
                blur_score  = EXCLUDED.blur_score,
                exif_data   = EXCLUDED.exif_data,
                updated_at  = NOW()
            """,
            (
                file_path,
                p.name,
                str(p.parent),
                p.suffix.lower().lstrip("."),
                entry.get("phash", "") or None,
                entry.get("blur", 0.0),
                json.dumps(entry.get("exif", {})),
            ),
        )
    conn.commit()


def reconcile(roots: list[str], dry_run: bool = False):
    t0 = time.perf_counter()
    scanned = 0
    fixed = 0
    failed = 0

    for directory in _walk_sidecar_directories(roots):
        # Hold the complete read-modify-write critical section. On POSIX the
        # shared helper also takes an advisory file lock, preventing a live
        # worker process from being overwritten by this admin process.
        with sidecar_lock(directory):
            sidecar = load_sidecar(directory)
            modified = False
            for filename, entry in list(sidecar.items()):
                state = entry.get("sidecar_state")
                if state not in ("pending", "failed"):
                    continue

                scanned += 1
                file_path = os.path.join(directory, filename)

                if dry_run:
                    logger.info(
                        "[DRY-RUN] Would reconcile: %s (state=%s)",
                        file_path,
                        state,
                    )
                    continue

                try:
                    with psycopg.connect(DATABASE_URL) as conn:
                        _upsert_to_postgres(conn, file_path, entry)
                    entry["sidecar_state"] = "synced"
                    modified = True
                    fixed += 1
                    logger.info("[FIXED] %s", file_path)
                except Exception as exc:
                    failed += 1
                    logger.error("[FAILED] %s: %s", file_path, exc)

            if modified and not dry_run:
                save_sidecar(directory, sidecar)

    elapsed = time.perf_counter() - t0
    rate = scanned / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Reconciliation complete: scanned=%d, fixed=%d, still_failed=%d in %.2fs = %.1f entries/sec.",
        scanned, fixed, failed, elapsed, rate,
    )


def main():
    parser = argparse.ArgumentParser(description="Reconcile pending/failed sidecars to Postgres.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes.")
    parser.add_argument(
        "roots",
        nargs="*",
        help="Media roots to scan (default: CRAWL_ROOTS, then home dir).",
    )
    args = parser.parse_args()

    if args.roots:
        roots = args.roots
    elif CRAWL_ROOTS_ENV:
        roots = [
            p.strip() for p in CRAWL_ROOTS_ENV.split(os.pathsep)
            if p.strip() and os.path.isdir(p.strip())
        ]
        if not roots:
            logger.warning(
                "CRAWL_ROOTS contained no readable directories; using home."
            )
            roots = [str(Path.home())]
    else:
        roots = [str(Path.home())]
    reconcile(roots, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
