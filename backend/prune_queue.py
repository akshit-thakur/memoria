"""Memoria – Huey SQLite Queue Pruner.

In plain English: a maintenance tool. Sometimes the watcher enqueues
tasks for files that have since been deleted, moved into an excluded
folder, or were never supposed to be indexed (dotfile typo, mistake, …).
This script reads each pending Huey task, unpickles the target path, and
deletes the task if the path is excluded, hidden, or missing on disk.

Use ``--execute`` to actually delete; default is a dry-run preview.

Removes tasks from the Huey queue whose target file paths:
  - Fall inside an excluded directory (EXCLUDED_DIRS / EXCLUDED_DIR_SUFFIXES)
  - Fall inside a hidden directory (any path component starts with '.')
  - No longer exist on disk (stale / deleted files)

Run this script once to drain the bad backlog, then restart services.

Usage:
    cd backend
    python prune_queue.py              # dry-run (shows what would be removed)
    python prune_queue.py --execute    # actually deletes the rows

The script works with a hot or cold queue (services can be running).
"""

import argparse
import os
import pickle
import sqlite3
import sys
from pathlib import Path

# ── Resolve paths ────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from constants import EXCLUDED_DIR_SUFFIXES, EXCLUDED_DIRS
from runtime_paths import QUEUE_DB_PATH

QUEUE_DB = str(QUEUE_DB_PATH)


# ── Classification helpers ───────────────────────────────────────────────────
def _in_excluded(path: str) -> bool:
    """Return True if the path should not be indexed."""
    for part in Path(path).parts:
        if part in EXCLUDED_DIRS:
            return True
        if any(part.endswith(sfx) for sfx in EXCLUDED_DIR_SUFFIXES):
            return True
        if part.startswith("."):
            return True
    return False


def _extract_path(data: bytes) -> str | None:
    """Unpack a Huey Message blob and return the first positional arg (the file path)."""
    try:
        msg = pickle.loads(data)
        # Huey 2.x / 3.x: huey.registry.Message namedtuple with .args
        args = getattr(msg, "args", None)
        if args:
            return str(args[0])
        # Fallback: plain tuple / list
        if isinstance(msg, (tuple, list)) and msg:
            return str(msg[0])
    except Exception:
        pass
    return None


# ── Main pruning logic ───────────────────────────────────────────────────────
def prune(execute: bool) -> None:
    if not Path(QUEUE_DB).exists():
        print(f"Queue DB not found: {QUEUE_DB}")
        sys.exit(1)

    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"[{mode}] Opening queue: {QUEUE_DB}")

    con = sqlite3.connect(QUEUE_DB, timeout=10)
    con.isolation_level = None  # autocommit
    cur = con.cursor()

    cur.execute("SELECT id, data FROM task")
    rows = cur.fetchall()
    total = len(rows)
    print(f"[{mode}] Total queued tasks: {total:,}")

    ids_excluded: list[int] = []
    ids_missing: list[int] = []
    ids_unreadable: list[int] = []

    for row_id, blob in rows:
        raw = bytes(blob)
        file_path = _extract_path(raw)

        if file_path is None:
            ids_unreadable.append(row_id)
            continue

        if _in_excluded(file_path):
            ids_excluded.append(row_id)
        elif not os.path.exists(file_path):
            ids_missing.append(row_id)

    all_to_delete = ids_excluded + ids_missing + ids_unreadable
    print(
        f"[{mode}] To remove:\n"
        f"          excluded dirs/bundles : {len(ids_excluded):,}\n"
        f"          file no longer exists : {len(ids_missing):,}\n"
        f"          unreadable/corrupt    : {len(ids_unreadable):,}\n"
        f"          ────────────────────────\n"
        f"          total to prune        : {len(all_to_delete):,}\n"
        f"          remaining after prune : {total - len(all_to_delete):,}"
    )

    if not all_to_delete:
        print(f"[{mode}] Nothing to prune.")
        con.close()
        return

    if not execute:
        print(f"\n[DRY-RUN] No changes made. Re-run with --execute to apply.")
        # Show a sample of what would be removed
        sample_ids = (ids_excluded + ids_missing)[:5]
        con.row_factory = sqlite3.Row
        for sid in sample_ids:
            row = con.execute("SELECT data FROM task WHERE id=?", (sid,)).fetchone()
            if row:
                path = _extract_path(bytes(row[0]))
                print(f"  sample: {path}")
        con.close()
        return

    # Delete in batches of 500 to avoid SQLite max-variable limits.
    BATCH = 500
    deleted = 0
    for i in range(0, len(all_to_delete), BATCH):
        batch = all_to_delete[i : i + BATCH]
        placeholders = ",".join("?" * len(batch))
        cur.execute(f"DELETE FROM task WHERE id IN ({placeholders})", batch)
        deleted += len(batch)
        print(f"[EXECUTE] Deleted {deleted:,} / {len(all_to_delete):,} …", end="\r")

    print()
    print(f"[EXECUTE] Done. Pruned {deleted:,} tasks. {total - deleted:,} remain in queue.")
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune stale/excluded tasks from the Huey SQLite queue.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete the rows. Without this flag the script runs in dry-run mode.",
    )
    args = parser.parse_args()
    prune(execute=args.execute)
