#!/usr/bin/env python3
"""Memoria – Dead-Letter Drain Script (Spec 05 B2).

In plain English: when a pipeline task crashes hard (raised an exception
past the safety net) it leaves a row in the ``processing_log`` table with
``status='failed'`` and the error message. This script either lists those
failed entries (default) or re-enqueues them for another attempt — useful
after fixing a bug that caused the original failure.

Queries the processing_log for failed entries and optionally re-enqueues them.

Usage:
    python backend/drain_failed.py                # list failed entries
    python backend/drain_failed.py --requeue      # re-enqueue failed entries
"""

import argparse
import logging
import os
import time

import psycopg
from database_config import DATABASE_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("memoria.drain")


def drain(requeue: bool = False):
    t0 = time.perf_counter()
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, file_path, attempt, error_msg, updated_at "
                "FROM processing_log WHERE status = 'failed' ORDER BY updated_at DESC"
            )
            rows = cur.fetchall()

        if not rows:
            logger.info("No failed entries in processing_log.")
            logger.info("drain done in %.2fs (entries=0).", time.perf_counter() - t0)
            return

        logger.info("Found %d failed entries.", len(rows))
        for row in rows:
            log_id, file_path, attempt, error_msg, updated_at = row
            logger.info("  [%d] %s — attempts=%d, error=%s, last=%s", log_id, file_path, attempt, error_msg, updated_at)

        if requeue:
            from workers.tasks import process_media

            requeued = 0
            for row in rows:
                log_id, file_path, attempt, error_msg, updated_at = row
                if not os.path.exists(file_path):
                    logger.warning("  Skipping (file gone): %s", file_path)
                    continue
                try:
                    process_media(file_path)
                    requeued += 1
                    logger.info("  Re-enqueued: %s", file_path)
                except Exception as exc:
                    logger.error("  Failed to re-enqueue %s: %s", file_path, exc)

            # Reset status to 'pending' for re-enqueued entries.
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE processing_log SET status = 'pending', updated_at = now() WHERE status = 'failed'"
                )
            conn.commit()
            logger.info("Re-enqueued %d / %d failed entries.", requeued, len(rows))

    elapsed = time.perf_counter() - t0
    rate = len(rows) / elapsed if elapsed > 0 else 0.0
    logger.info(
        "drain done: entries=%d in %.2fs = %.1f entries/sec.",
        len(rows), elapsed, rate,
    )


def main():
    parser = argparse.ArgumentParser(description="List or re-enqueue failed processing_log entries.")
    parser.add_argument("--requeue", action="store_true", help="Re-enqueue failed entries.")
    args = parser.parse_args()
    drain(requeue=args.requeue)


if __name__ == "__main__":
    main()
