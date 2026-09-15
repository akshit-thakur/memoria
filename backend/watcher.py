"""Memoria – System-Wide Media Watcher.

In plain English: this is the always-on filesystem agent. It does two
things at the same time:

  1. On startup, walk every directory under the configured roots (or the
     user's home directory by default) and find every image / video file.
     A persistent SQLite index ("scan_index.db") remembers which files
     have been seen with which mtime + size, so re-scans only enqueue
     files that are *new* or have *changed*.

  2. While running, listen for filesystem events (create, modify, delete,
     move) via the watchdog library. Anything new gets enqueued into the
     Huey task queue for the worker to process.

A small amount of bookkeeping protects the queue: a 30-second TTL dedup
set so a file save's "created" + "modified" event storm doesn't enqueue
twice; a 500 ms debounce that batches discovered paths together; an
exponential-backoff retry on the rare SQLite "queue is locked" error.

Autonomously crawls the OS for media files and monitors the filesystem for
new creations / modifications via ``watchdog``.  Discovered paths are pushed
to the Huey SQLite-backed task queue for the worker to process.
"""

import atexit
import logging
import os
import signal
import sqlite3
import stat as stat_module
import sys
import time
import threading
from collections import deque
from pathlib import Path

import psycopg
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import metrics
from database_config import DATABASE_URL
from runtime_paths import SCAN_INDEX_DB_PATH, SCAN_SESSION_PATH
from workers.tasks import delete_media, process_media, process_media_batch

logger = logging.getLogger("memoria.watcher")

# Only start the watcher snapshotter when running as the watcher process
# (`python watcher.py`). Importing watcher.py from another script — e.g. the
# bench harness — would otherwise overwrite watcher.json every 30 s.
_watcher_arg0 = (sys.argv[0] if sys.argv else "").lower()
if _watcher_arg0.endswith("watcher.py") or os.environ.get("MEMORIA_WATCHER") == "1":
    _metrics_snapshotter = metrics.PeriodicSnapshotter("watcher", interval_s=30.0)
    _metrics_snapshotter.start()
    atexit.register(_metrics_snapshotter.stop)


# ── Micro-batching config (Spec 05 A1) ───────────────────────────────────────
ENQUEUE_DEBOUNCE_MS = 500
_pending_paths: deque[str] = deque()
_pending_lock = threading.Lock()
_flush_timer: threading.Timer | None = None

# POSIX lifecycle scripts and Supervisor stop the watcher with SIGTERM. Keep
# handlers minimal: they only set this flag; normal control flow below owns
# queue writes, SQLite commits, and thread joins.
_shutdown_event = threading.Event()


def _request_shutdown(signum, frame) -> None:
    _shutdown_event.set()


def _install_shutdown_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)


# ── Batched-pipeline switch (Phase E6.3) ────────────────────────────────────
# MEMORIA_BATCH_SIZE controls whether the watcher dispatches per-file
# `process_media` tasks (the safety-net path) or chunked `process_media_batch`
# tasks (the high-throughput path). The default is 16; invalid, non-positive,
# or over-cap values fall back to that safe configured baseline.
try:
    _configured_batch_size = int(os.getenv("MEMORIA_BATCH_SIZE", "16"))
    if not 1 <= _configured_batch_size <= 50:
        raise ValueError("batch size must be between 1 and 50")
    MEMORIA_BATCH_SIZE = _configured_batch_size
except (TypeError, ValueError):
    MEMORIA_BATCH_SIZE = 16


def _safe_enqueue(path: str, max_retries: int = 5) -> bool:
    """Enqueue a single path with exponential backoff on SQLite lock errors (Spec 05 A2).

    Huey uses a SQLite database as its task queue; when the worker and the
    watcher race to write at the same time, SQLite raises an "OperationalError:
    database is locked". We catch that specific case and retry up to five
    times with exponentially growing pauses (0.1 → 0.2 → 0.4 → … capped at
    10 s). Any other exception is logged once and reported to the caller so
    scan state and live-event TTL claims can be rolled back for a later retry.
    """
    delay = 0.1
    for attempt in range(max_retries):
        try:
            with metrics.timer("watcher.enqueue"):
                process_media(path)
            metrics.count("watcher.enqueue.ok")
            return True
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                logger.warning("SQLite locked on enqueue for %s — retrying in %.1fs (attempt %d/%d)", path, delay, attempt + 1, max_retries)
                metrics.count("watcher.enqueue.lock_retry")
                time.sleep(delay)
                delay = min(delay * 2, 10.0)
            else:
                logger.error("Failed to enqueue %s after %d attempts: %s", path, attempt + 1, e)
                metrics.count("watcher.enqueue.error")
                return False
        except Exception as exc:
            logger.error("Failed to enqueue %s: %s", path, exc)
            metrics.count("watcher.enqueue.error")
            return False
    return False


def _safe_enqueue_batch(paths: list[str], max_retries: int = 5) -> bool:
    """Enqueue a chunk via process_media_batch with the same SQLite-lock
    backoff as :func:`_safe_enqueue` (Phase E6.3)."""
    if not paths:
        return True
    delay = 0.1
    for attempt in range(max_retries):
        try:
            with metrics.timer("watcher.enqueue_batch", n=len(paths)):
                process_media_batch(paths)
            metrics.count("watcher.enqueue_batch.ok")
            metrics.record("watcher.batch_size", len(paths))
            return True
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                logger.warning(
                    "SQLite locked on batch enqueue (n=%d) — retrying in %.1fs (attempt %d/%d)",
                    len(paths), delay, attempt + 1, max_retries,
                )
                metrics.count("watcher.enqueue_batch.lock_retry")
                time.sleep(delay)
                delay = min(delay * 2, 10.0)
            else:
                logger.error(
                    "Failed to batch-enqueue %d paths after %d attempts: %s",
                    len(paths), attempt + 1, e,
                )
                metrics.count("watcher.enqueue_batch.error")
                return False
        except Exception as exc:
            logger.error("Failed to batch-enqueue %d paths: %s", len(paths), exc)
            metrics.count("watcher.enqueue_batch.error")
            return False
    return False


def _safe_enqueue_delete(path: str, max_retries: int = 5) -> bool:
    """Durably enqueue a metadata deletion, retrying SQLite lock races.

    The boolean result lets the scan-index reconciler retain a stale row when
    queue persistence fails, so a later scan can retry instead of forgetting
    a database row that still needs removal.
    """
    delay = 0.1
    for attempt in range(max_retries):
        try:
            delete_media(path)
            metrics.count("watcher.enqueue_delete.ok")
            return True
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < max_retries - 1:
                logger.warning(
                    "SQLite locked on delete enqueue for %s — retrying in "
                    "%.1fs (attempt %d/%d)",
                    path,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                metrics.count("watcher.enqueue_delete.lock_retry")
                time.sleep(delay)
                delay = min(delay * 2, 10.0)
                continue
            logger.error(
                "Failed to enqueue metadata deletion for %s after %d "
                "attempts: %s",
                path,
                attempt + 1,
                exc,
            )
            metrics.count("watcher.enqueue_delete.error")
            return False
        except Exception as exc:
            logger.error("Failed to enqueue metadata deletion for %s: %s", path, exc)
            metrics.count("watcher.enqueue_delete.error")
            return False
    return False


def _dispatch_paths(paths: list[str]) -> set[str]:
    """Enqueue *paths* according to MEMORIA_BATCH_SIZE.

    When MEMORIA_BATCH_SIZE == 1, each path goes as its own
    `process_media` task — preserves the existing single-image safety net.
    When > 1 (the default is 16), chunk and dispatch `process_media_batch`.
    The return value contains only paths whose queue write was accepted.
    """
    if not paths:
        return set()
    accepted: set[str] = set()
    if MEMORIA_BATCH_SIZE <= 1:
        for p in paths:
            if _safe_enqueue(p):
                accepted.add(p)
        return accepted
    n = MEMORIA_BATCH_SIZE
    for i in range(0, len(paths), n):
        chunk = paths[i:i + n]
        if _safe_enqueue_batch(chunk):
            accepted.update(chunk)
    return accepted


def _normalize_dispatch_result(result, attempted: list[str]) -> set[str]:
    """Convert callback results while keeping legacy ``None`` compatibility."""
    attempted_set = set(attempted)
    if result is None or result is True:
        return attempted_set
    if result is False:
        return set()
    try:
        return attempted_set.intersection(result)
    except TypeError:
        logger.error("Invalid enqueue callback result %r; treating as failure", result)
        return set()


def _forget_scan_index_paths(paths: set[str], max_retries: int = 5) -> bool:
    """Remove failed live dispatches so the next scan retries them.

    The live debounce timer can race the initial crawl's SQLite transaction.
    Retry lock conflicts rather than leaving a matching mtime/size row that
    would make the failed path look fully indexed on every later startup.
    """
    if not paths:
        return True
    index_path = Path(SCAN_INDEX_DB_PATH)
    if not index_path.exists():
        return True

    delay = 0.1
    for attempt in range(max_retries):
        try:
            with sqlite3.connect(str(index_path), timeout=10) as db:
                db.executemany(
                    "DELETE FROM scan_index WHERE path = ?",
                    ((path,) for path in paths),
                )
                db.commit()
            return True
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            # The initial crawl may have created the database file but not
            # the table yet. In that state there cannot be a row to forget.
            if "no such table" in message:
                return True
            if ("locked" in message or "busy" in message) and attempt < max_retries - 1:
                logger.warning(
                    "Scan index locked while clearing %d failed live path(s) "
                    "- retrying in %.1fs (attempt %d/%d)",
                    len(paths),
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
                delay = min(delay * 2, 10.0)
                continue
            logger.warning(
                "Could not clear %d failed live path(s) from scan index: %s",
                len(paths),
                exc,
            )
            return False
        except Exception as exc:
            logger.warning(
                "Could not clear %d failed live path(s) from scan index: %s",
                len(paths),
                exc,
            )
            return False
    return False


def _flush_pending():
    """Flush all pending paths in one burst, each with backoff protection."""
    global _flush_timer
    with _pending_lock:
        batch = list(_pending_paths)
        _pending_paths.clear()
        _flush_timer = None
    try:
        result = _dispatch_paths(batch)
        accepted = _normalize_dispatch_result(result, batch)
    except Exception as exc:
        logger.error("Pending watcher dispatch failed: %s", exc)
        accepted = set()
    failed = set(batch) - accepted
    for path in failed:
        _enqueue_seen.discard(path)
    if failed:
        if _shutdown_event.is_set():
            # Native stop grants five seconds and Supervisor grants thirty. Do
            # not enter the scan-index lock retry loop during that grace window:
            # new paths have no row, modified paths retain their old mtime/size,
            # and startup-overlap paths are conservatively omitted by the crawl.
            logger.warning(
                "Shutdown interrupted %d live enqueue(s); deferring them to "
                "the next incremental scan",
                len(failed),
            )
            metrics.count("watcher.enqueue.shutdown_deferred", len(failed))
        else:
            _forget_scan_index_paths(failed)


def _debounced_enqueue(path: str):
    """Add path to the pending batch; schedule a flush after ENQUEUE_DEBOUNCE_MS.

    The watchdog observer often fires several events back-to-back for one
    file save (one for the temp file rename, one for the modify, one for
    the close). Rather than hitting the queue three times, we collect
    paths for 500 ms and flush as one burst.
    """
    global _flush_timer
    with _pending_lock:
        _pending_paths.append(path)
        if _flush_timer is None:
            _flush_timer = threading.Timer(ENQUEUE_DEBOUNCE_MS / 1000.0, _flush_pending)
            _flush_timer.daemon = True
            _flush_timer.start()


# ── Dedup Guard (Spec 02 D1) ─────────────────────────────────────────────────
# Prevents the watcher from double-enqueuing the same file during event storms.
# Entries expire after _DEDUP_TTL_SECS seconds.

_DEDUP_TTL_SECS = 30

class _TTLSet:
    """Thread-safe set with per-entry TTL expiry.

    In plain English: a Set-like container where every entry automatically
    "vanishes" after `ttl` seconds. Used here to remember "this file path
    has already been claimed for enqueue in the last 30 seconds, ignore
    duplicate events for it." A claim may still be waiting in the debounce
    buffer; only an accepted dispatch result proves queue durability. Old
    entries are pruned lazily on each `add()` call — no background thread.
    """

    def __init__(self, ttl: float):
        self._ttl = ttl
        self._entries: dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, key: str) -> bool:
        """Add key. Returns True if newly added, False if already present (not expired)."""
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            if key in self._entries:
                return False
            self._entries[key] = now + self._ttl
            return True

    def discard(self, key: str) -> None:
        """Release a claim immediately after durable dispatch fails."""
        with self._lock:
            self._entries.pop(key, None)

    def _prune(self, now: float):
        """Remove every entry whose expiry timestamp has passed.
        Called from inside add() under the same lock."""
        expired = [k for k, exp in self._entries.items() if exp <= now]
        for k in expired:
            del self._entries[k]


_enqueue_seen = _TTLSet(_DEDUP_TTL_SECS)


def _dispatch_discovered_paths(paths: list[str]) -> set[str]:
    """Dispatch crawl results through the same TTL guard as live events.

    The observer runs before the initial crawl. Whichever source sees a path
    first claims it here, preventing a create event and the overlapping crawl
    from adding duplicate work. A pre-existing claim is intentionally *not*
    reported as accepted: the observer may still have that path only in its
    debounce buffer. The scan therefore omits its index row until some later
    crawl can prove a durable queue write, while the observer remains free to
    finish its own dispatch.
    """
    claimed: list[str] = []
    for path in dict.fromkeys(paths):
        if _enqueue_seen.add(path):
            claimed.append(path)
        else:
            logger.debug(
                "Discovery claim for %s is not yet proven durable; leaving "
                "it out of the scan index",
                path,
            )

    try:
        result = _dispatch_paths(claimed) if claimed else set()
        accepted = _normalize_dispatch_result(result, claimed)
    except Exception as exc:
        logger.error("Crawl dispatch failed: %s", exc)
        accepted = set()
    for path in set(claimed) - accepted:
        _enqueue_seen.discard(path)
    return accepted

from constants import (
    EXCLUDE_PATH_PREFIXES,
    EXCLUDED_DIR_SUFFIXES,
    EXCLUDED_DIRS,
    MEDIA_EXTENSIONS,
)

# ── Configurable crawl roots ────────────────────────────────────────────────
# Set the CRAWL_ROOTS env var to a colon-separated list of absolute paths to
# restrict which directories are crawled and watched. When unset, the home dir
# plus any mounted volumes under /media/$USER are used.
#
#   export CRAWL_ROOTS="$HOME/Pictures:/mnt/photos"
#
CRAWL_ROOTS_ENV = os.getenv("CRAWL_ROOTS", "")

# ── Scan session tracking ────────────────────────────────────────────────────
import json
import uuid
from datetime import datetime, timezone

_SCAN_SESSION_PATH = SCAN_SESSION_PATH


def _write_scan_session(data: dict) -> None:
    """Atomically write scan session JSON to ~/.memoria/scan_session.json."""
    _SCAN_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _SCAN_SESSION_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data))
        os.replace(tmp, _SCAN_SESSION_PATH)
    except Exception as exc:
        logger.warning("Failed to write scan session: %s", exc)


def is_media(path: str) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def _in_excluded_dir(path: str) -> bool:
    """Return True if *path* passes through an excluded directory name or bundle suffix."""
    parts = Path(path).parts
    for part in parts:
        if part in EXCLUDED_DIRS:
            return True
        # Check macOS bundle suffixes (e.g. 'Photos Library.photoslibrary')
        if any(part.endswith(sfx) for sfx in EXCLUDED_DIR_SUFFIXES):
            return True
    return False


def _in_hidden_dir(path: str) -> bool:
    """Return True if any component of *path* is a hidden directory (starts with '.')."""
    return any(part.startswith(".") for part in Path(path).parts)


def _in_excluded_path(path: str) -> bool:
    """Return True if *path* falls under any of the EXCLUDE_PATH_PREFIXES."""
    norm = os.path.normpath(os.path.abspath(path))
    for prefix in EXCLUDE_PATH_PREFIXES:
        p_norm = os.path.normpath(os.path.abspath(prefix))
        if norm == p_norm or norm.startswith(p_norm + os.sep):
            return True
    return False


# ── DB Cleanup ──────────────────────────────────────────────────────────────

# Keyset-pagination chunk size for the purge scans below. At 70k+ rows the
# old "SELECT id, file_path FROM media" loaded the entire table into Python;
# we now walk it in 5k chunks so RSS stays bounded.
_PURGE_CHUNK_SIZE = 5000


def _purge_in_chunks(
    label: str,
    predicate,
    shutdown_event: threading.Event | None = None,
) -> None:
    """Stream media rows in keyset chunks and delete those matching predicate(file_path).

    predicate :: str -> bool
    """
    last_id = 0
    deleted_total = 0
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            break
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, file_path FROM media WHERE id > %s ORDER BY id LIMIT %s",
                    (last_id, _PURGE_CHUNK_SIZE),
                )
                rows = cur.fetchall()
        if not rows:
            break
        last_id = rows[-1][0]
        ids_to_delete = [r[0] for r in rows if predicate(r[1])]
        if ids_to_delete:
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM media WHERE id = ANY(%s)",
                        (ids_to_delete,),
                    )
                conn.commit()
            deleted_total += len(ids_to_delete)
    if deleted_total:
        logger.info("%s: purged %d entries.", label, deleted_total)
    else:
        logger.info("%s: no matching entries found.", label)


def purge_hidden_dir_entries(
    shutdown_event: threading.Event | None = None,
) -> None:
    """Delete any DB rows whose file_path passes through a hidden or excluded directory."""
    logger.info("Purging hidden/excluded-directory entries from database …")
    _purge_in_chunks(
        "hidden/excluded purge",
        lambda fp: _in_hidden_dir(fp) or _in_excluded_dir(fp),
        shutdown_event=shutdown_event,
    )

# ── Incremental Filesystem Scanner (Spec 06) ─────────────────────────────────

SCAN_INDEX_DB = str(SCAN_INDEX_DB_PATH)
SCAN_BATCH_SIZE = 200
SCAN_BATCH_DELAY = 0.5  # seconds between enqueue batches
_STALE_RECONCILE_CHUNK_SIZE = 1000


def _init_scan_index(db: sqlite3.Connection):
    """Create the scan_index table if it doesn't exist."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS scan_index (
            path          TEXT PRIMARY KEY,
            mtime         REAL NOT NULL,
            size          INTEGER NOT NULL,
            last_seen_at  REAL NOT NULL
        )
    """)
    db.commit()


def _is_scan_excluded(entry_path: str, entry_name: str) -> bool:
    """Return True if a scandir entry should be skipped."""
    if entry_name in EXCLUDED_DIRS or entry_name.startswith("."):
        return True
    if any(entry_name.endswith(sfx) for sfx in EXCLUDED_DIR_SUFFIXES):
        return True
    norm = os.path.normpath(os.path.abspath(entry_path))
    for prefix in EXCLUDE_PATH_PREFIXES:
        p_norm = os.path.normpath(os.path.abspath(prefix))
        if norm == p_norm or norm.startswith(p_norm + os.sep):
            return True
    return False


def _stage_stale_index_paths(
    db: sqlite3.Connection,
    root: str,
    scan_started_at: float,
    unreadable_prefixes: set[str],
) -> int:
    """Snapshot safely stale paths into a temporary table.

    Only paths below the completed root are staged. Any directory or file
    prefix that raised during traversal is excluded, so an unavailable mount
    or permission boundary is never interpreted as an offline deletion.
    """
    root_prefix = root if root.endswith(os.sep) else root + os.sep
    db.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS scan_unreadable_prefixes (
            path TEXT PRIMARY KEY
        )
        """
    )
    db.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS scan_stale_candidates (
            path TEXT PRIMARY KEY
        )
        """
    )
    db.execute("DELETE FROM scan_unreadable_prefixes")
    db.execute("DELETE FROM scan_stale_candidates")
    if unreadable_prefixes:
        db.executemany(
            "INSERT OR IGNORE INTO scan_unreadable_prefixes(path) VALUES (?)",
            ((path,) for path in unreadable_prefixes),
        )
    db.execute(
        """
        INSERT OR IGNORE INTO scan_stale_candidates(path)
        SELECT scan_index.path
        FROM scan_index
        WHERE scan_index.last_seen_at < ?
          AND (
              scan_index.path = ?
              OR substr(scan_index.path, 1, ?) = ?
          )
          AND NOT EXISTS (
              SELECT 1
              FROM scan_unreadable_prefixes AS unreadable
              WHERE scan_index.path = unreadable.path
                 OR substr(scan_index.path, 1, length(unreadable.path) + 1)
                    = unreadable.path || ?
          )
        """,
        (scan_started_at, root, len(root_prefix), root_prefix, os.sep),
    )
    row = db.execute("SELECT COUNT(*) FROM scan_stale_candidates").fetchone()
    db.commit()
    return int(row[0]) if row else 0


def _reconcile_stale_index_paths(
    db: sqlite3.Connection,
    delete_fn=None,
    shutdown_event: threading.Event | None = None,
) -> int:
    """Queue safe offline deletions and remove only acknowledged index rows."""
    scheduled = 0
    last_path = ""
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            break
        rows = db.execute(
            """
            SELECT path
            FROM scan_stale_candidates
            WHERE path > ?
            ORDER BY path
            LIMIT ?
            """,
            (last_path, _STALE_RECONCILE_CHUNK_SIZE),
        ).fetchall()
        if not rows:
            break
        last_path = rows[-1][0]
        reconciled: list[tuple[str]] = []
        for (path,) in rows:
            if shutdown_event is not None and shutdown_event.is_set():
                break
            # The observer is already active. A file may have been recreated
            # after its directory was scanned but before reconciliation. Do a
            # final stat so a queued create can never race behind a deletion.
            try:
                current = os.stat(path, follow_symlinks=False)
            except FileNotFoundError:
                current = None
            except OSError as exc:
                logger.warning(
                    "Preserving stale index entry that cannot be rechecked %s: %s",
                    path,
                    exc,
                )
                continue

            if current is not None and stat_module.S_ISREG(current.st_mode):
                db.execute(
                    """
                    UPDATE scan_index
                    SET mtime = ?, size = ?, last_seen_at = ?
                    WHERE path = ?
                    """,
                    (current.st_mtime, current.st_size, time.time(), path),
                )
                continue

            try:
                accepted = True if delete_fn is None else delete_fn(path)
            except Exception as exc:
                logger.error("Offline delete enqueue failed for %s: %s", path, exc)
                accepted = False
            if accepted is False:
                continue
            reconciled.append((path,))
            if delete_fn is not None:
                scheduled += 1

        if reconciled:
            db.executemany("DELETE FROM scan_index WHERE path = ?", reconciled)
            db.commit()

    db.execute("DELETE FROM scan_stale_candidates")
    db.execute("DELETE FROM scan_unreadable_prefixes")
    db.commit()
    return scheduled


def _dispatch_scan_batch(
    db: sqlite3.Connection,
    paths: list[str],
    enqueue_fn,
    on_progress=None,
) -> int:
    """Persist scan state only for paths whose queue dispatch was accepted."""
    if not paths:
        return 0
    db.commit()
    try:
        result = enqueue_fn(list(paths))
    except Exception as exc:
        logger.error("Scan enqueue callback failed for %d path(s): %s", len(paths), exc)
        accepted: set[str] = set()
    else:
        accepted = _normalize_dispatch_result(result, paths)

    failed = set(paths) - accepted
    if failed:
        db.executemany(
            "DELETE FROM scan_index WHERE path = ?",
            ((path,) for path in failed),
        )
        db.commit()
        metrics.count("watcher.scan.enqueue_rejected", len(failed))
    if accepted and on_progress is not None:
        on_progress(len(accepted))
    return len(accepted)


def incremental_scan(
    root: str,
    db: sqlite3.Connection,
    enqueue_fn,
    on_progress=None,
    delete_fn=None,
    shutdown_event: threading.Event | None = None,
):
    """Walk root using os.scandir with mtime-based skip.

    Only enqueues files that are new or modified since the last scan.
    Unchanged files get their last_seen_at updated but are not re-enqueued.

    In plain English: instead of asking the ML pipeline to re-process
    every photo on every boot, we keep a tiny SQLite index keyed by
    absolute path that remembers each file's mtime + size from the last
    scan. If both match this time around, the file is unchanged — skip
    it, just touch `last_seen_at`. If they differ (or the file is new),
    enqueue it.

    *enqueue_fn* is called with the entire ``batch`` list at each
    SCAN_BATCH_SIZE flush — pass ``_dispatch_paths`` (or any callable that
    accepts a list of absolute paths) to honor MEMORIA_BATCH_SIZE. It may
    return an iterable of accepted paths or a boolean; legacy ``None`` results
    mean the whole batch was accepted.

    *on_progress* is an optional callable(enqueued_delta: int) fired at
    every SCAN_BATCH_SIZE boundary and at the final flush.

    *delete_fn* is an optional callable(path) used to enqueue root-scoped
    offline deletions. Returning ``False`` preserves the scan-index row for a
    later retry; omitting it performs scan-index pruning without external work.

    *shutdown_event* cooperatively stops a partial walk. The current batch is
    still resolved through the durable queue callback, but stale-row deletion
    is skipped because an interrupted walk cannot prove which files vanished.
    """
    # Iterative depth-first walk using a stack — avoids Python's recursion
    # limit on deep directory trees and is friendlier to os.scandir's
    # generator semantics.
    root = os.path.normpath(os.path.abspath(root))
    stack = [root]
    now = time.time()
    enqueued = 0
    batch: list[str] = []

    # Preload the full scan_index into a dict so the per-file hot path is a
    # dict lookup instead of a SELECT. At 300k files this removes ~300k round
    # trips to SQLite and is the dominant cost of a second-run (no-op) scan.
    _preload_rows = db.execute("SELECT path, mtime, size FROM scan_index").fetchall()
    index_cache: dict[str, tuple[float, float]] = {r[0]: (r[1], r[2]) for r in _preload_rows}
    del _preload_rows

    # Durable scan telemetry feeds diagnostics and benchmark reports.
    scan_started = time.perf_counter()
    files_seen = 0
    files_skipped = 0
    files_new = 0
    dirs_walked = 0
    unreadable_prefixes: set[str] = set()
    cancelled = bool(shutdown_event and shutdown_event.is_set())

    while stack and not cancelled:
        current = stack.pop()
        dirs_walked += 1
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if shutdown_event is not None and shutdown_event.is_set():
                        cancelled = True
                        break
                    if entry.is_dir(follow_symlinks=False):
                        if not _is_scan_excluded(entry.path, entry.name):
                            stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext not in MEDIA_EXTENSIONS:
                            continue
                        try:
                            stat = entry.stat(follow_symlinks=False)
                        except OSError as exc:
                            unreadable_prefixes.add(
                                os.path.normpath(os.path.abspath(entry.path))
                            )
                            logger.warning(
                                "Skipping unreadable scan file %s: %s",
                                entry.path,
                                exc,
                            )
                            continue
                        files_seen += 1
                        row = index_cache.get(entry.path)
                        if row and row[0] == stat.st_mtime and row[1] == stat.st_size:
                            # Unchanged — update last_seen_at only.
                            db.execute(
                                "UPDATE scan_index SET last_seen_at=? WHERE path=?",
                                (now, entry.path),
                            )
                            files_skipped += 1
                        else:
                            # New or modified — enqueue.
                            batch.append(entry.path)
                            db.execute(
                                "INSERT OR REPLACE INTO scan_index(path,mtime,size,last_seen_at) VALUES(?,?,?,?)",
                                (entry.path, stat.st_mtime, stat.st_size, now),
                            )
                            enqueued += 1
                            files_new += 1

                            # Throttle: flush batch at SCAN_BATCH_SIZE.
                            # Sleeping briefly between flushes prevents the
                            # watcher from saturating the queue while the
                            # worker is still draining the previous burst.
                            if len(batch) >= SCAN_BATCH_SIZE:
                                attempted = len(batch)
                                accepted = _dispatch_scan_batch(
                                    db,
                                    batch,
                                    enqueue_fn,
                                    on_progress=on_progress,
                                )
                                enqueued -= attempted - accepted
                                batch.clear()
                                if enqueued > 0 and enqueued % 5000 == 0:
                                    logger.info("Scan: enqueued %d files so far …", enqueued)
                                if shutdown_event is None:
                                    time.sleep(SCAN_BATCH_DELAY)
                                elif shutdown_event.wait(SCAN_BATCH_DELAY):
                                    cancelled = True
                                    break
        except OSError as exc:
            # A directory can be unreadable or disappear while a removable
            # volume is being scanned. Protect its existing index entries
            # from the stale-row prune below; treating an incomplete walk as
            # an empty directory would cause needless re-indexing when it
            # becomes available again.
            unreadable_prefixes.add(os.path.normpath(os.path.abspath(current)))
            logger.warning("Skipping unreadable scan directory %s: %s", current, exc)
            continue

    # Flush remaining batch.
    if batch:
        attempted = len(batch)
        accepted = _dispatch_scan_batch(
            db,
            batch,
            enqueue_fn,
            on_progress=on_progress,
        )
        enqueued -= attempted - accepted
    if shutdown_event is not None and shutdown_event.is_set():
        cancelled = True

    scan_duration_ms = (time.perf_counter() - scan_started) * 1000.0
    metrics.record("watcher.scan.duration_ms", scan_duration_ms)
    metrics.count("watcher.scan.files_seen", files_seen)
    metrics.count("watcher.scan.files_skipped_mtime", files_skipped)
    metrics.count("watcher.scan.files_new", files_new)
    metrics.count("watcher.scan.dirs_walked", dirs_walked)
    logger.info(
        "[SCAN] %s: %d seen, %d new/changed, %d mtime-skipped, "
        "%d dirs in %.1fs",
        root, files_seen, files_new, files_skipped, dirs_walked,
        scan_duration_ms / 1000.0,
    )

    if cancelled:
        # Keep every unseen scan-index row. A later complete scan can safely
        # reconcile it; treating an interrupted traversal as complete would
        # enqueue false deletions for the unvisited portion of the library.
        db.commit()
        metrics.count("watcher.scan.cancelled")
        logger.info(
            "Incremental scan interrupted. Durably accepted %d new/modified "
            "files; stale reconciliation deferred.",
            enqueued,
        )
        return enqueued

    # Reconcile only paths proven absent from the root regions that were fully
    # traversed. The temporary snapshot makes queue dispatch retry-safe: an
    # index row is removed only after its delete task was accepted.
    db.commit()
    stale_count = _stage_stale_index_paths(
        db,
        root,
        now,
        unreadable_prefixes,
    )
    deletes_scheduled = _reconcile_stale_index_paths(
        db,
        delete_fn=delete_fn,
        shutdown_event=shutdown_event,
    )
    metrics.count("watcher.scan.stale_candidates", stale_count)
    metrics.count("watcher.scan.offline_deletes_scheduled", deletes_scheduled)
    db.commit()

    if shutdown_event is not None and shutdown_event.is_set():
        metrics.count("watcher.scan.cancelled")
        logger.info(
            "Incremental scan interrupted during stale reconciliation after "
            "durably accepting %d new/modified files.",
            enqueued,
        )
        return enqueued

    logger.info(
        "Incremental scan complete. Enqueued %d new/modified files; "
        "scheduled %d offline deletions.",
        enqueued,
        deletes_scheduled,
    )
    return enqueued


def crawl_system(
    roots: list[str] | None = None,
    shutdown_event: threading.Event | None = None,
):
    """Incremental system-wide scan using scandir + mtime-based skip (Spec 06).

    Replaces the old os.walk crawl. Uses a persistent SQLite scan_index
    to skip unchanged files across restarts.
    """
    # Ensure data directory exists.
    Path(SCAN_INDEX_DB).parent.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(SCAN_INDEX_DB, timeout=10)
    _init_scan_index(db)

    roots = list(_discover_roots() if roots is None else roots)
    logger.info("Starting incremental scan across roots: %s", roots)

    session_id = uuid.uuid4().hex[:12]
    session_started_at = datetime.now(timezone.utc).isoformat()
    session: dict = {
        "session_id": session_id,
        "started_at": session_started_at,
        "roots": roots,
        "enqueued": 0,
        "finished_at": None,
        "cancelled": False,
    }
    _write_scan_session(session)

    def _on_progress(delta: int) -> None:
        session["enqueued"] += delta
        _write_scan_session(session)

    total_enqueued = 0
    try:
        for root in roots:
            if shutdown_event is not None and shutdown_event.is_set():
                break
            total_enqueued += incremental_scan(
                root,
                db,
                _dispatch_discovered_paths,
                on_progress=_on_progress,
                delete_fn=_safe_enqueue_delete,
                shutdown_event=shutdown_event,
            )

        session["enqueued"] = total_enqueued
        session["cancelled"] = bool(
            shutdown_event is not None and shutdown_event.is_set()
        )
        session["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_scan_session(session)
    finally:
        db.close()
    logger.info("System scan complete. %d total files enqueued.", total_enqueued)


def discover_roots() -> list[str]:
    """Public wrapper for configured crawl roots used by API browse/scan."""
    return _discover_roots()


def scan_single_root(root: str) -> int:
    """Run an incremental scan for one directory tree and enqueue new/changed files."""
    Path(SCAN_INDEX_DB).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(SCAN_INDEX_DB, timeout=10)
    _init_scan_index(db)
    try:
        return incremental_scan(
            root,
            db,
            _dispatch_discovered_paths,
            delete_fn=_safe_enqueue_delete,
        )
    finally:
        db.close()


def _discover_roots():
    """Return a list of root directories to scan.

    If the CRAWL_ROOTS env var is set (colon-separated absolute paths), those
    are used exclusively – no default crawl occurs. Invalid entries are
    skipped; if none are valid, automatic crawling is disabled rather than
    unexpectedly widening the scan to the user's home directory.

    When CRAWL_ROOTS is unset, defaults to ``$HOME`` plus any mounted volumes
    under ``/media/$USER``.
    """
    if CRAWL_ROOTS_ENV:
        roots = []
        for raw in CRAWL_ROOTS_ENV.split(os.pathsep):
            configured = raw.strip()
            if not configured:
                continue
            p = os.path.normpath(os.path.abspath(configured))
            if os.path.isdir(p):
                roots.append(p)
            else:
                logger.warning("CRAWL_ROOTS entry is not a valid directory, skipping: %s", p)
        if roots:
            logger.info("Using CRAWL_ROOTS from env: %s", roots)
            return roots
        logger.error(
            "CRAWL_ROOTS was set but contained no valid directories; "
            "automatic crawling is disabled until the configuration is fixed."
        )
        return []

    home = Path.home()
    media = Path("/media") / home.name
    roots = [str(home)]
    if media.exists():
        roots.extend(str(v) for v in media.iterdir() if v.is_dir())
    return roots


# ── Realtime Watcher ─────────────────────────────────────────────────────────
class MediaEventHandler(FileSystemEventHandler):
    """Enqueue newly created / modified media files into the Huey task queue."""

    def _should_process(self, path: str) -> bool:
        return (
            not _in_hidden_dir(path)
            and not _in_excluded_dir(path)
            and not _in_excluded_path(path)
        )

    def _enqueue(self, path: str):
        """Enqueue a file for processing, respecting dedup + micro-batching (Spec 05)."""
        if not _enqueue_seen.add(path):
            logger.debug("Dedup: skipping recently enqueued %s", path)
            return
        _debounced_enqueue(path)

    def on_created(self, event):
        if not event.is_directory and is_media(event.src_path) and self._should_process(event.src_path):
            logger.info("New media detected: %s", event.src_path)
            self._enqueue(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and is_media(event.src_path) and self._should_process(event.src_path):
            logger.info("Modified media detected: %s", event.src_path)
            self._enqueue(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and is_media(event.src_path) and not _in_excluded_path(event.src_path):
            logger.info("Media deleted: %s", event.src_path)
            _safe_enqueue_delete(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        # Old path: remove only if it was in a tracked (non-excluded) location.
        if is_media(event.src_path) and not _in_excluded_path(event.src_path):
            logger.info("Media moved from: %s", event.src_path)
            _safe_enqueue_delete(event.src_path)
        # New path: index only if it is a non-hidden, non-excluded media file.
        if is_media(event.dest_path) and self._should_process(event.dest_path):
            logger.info("Media moved to: %s", event.dest_path)
            self._enqueue(event.dest_path)


def _stop_observer(observer) -> None:
    """Best-effort observer shutdown used by normal and failed startup."""
    try:
        observer.stop()
    except Exception as exc:
        logger.warning("Failed to stop filesystem observer: %s", exc)
    try:
        observer.join()
    except (RuntimeError, TypeError) as exc:
        # ``join`` raises when scheduling/start failed before the thread ran.
        logger.debug("Filesystem observer was not joinable: %s", exc)
    try:
        # A create event may own the shared TTL slot while still waiting in
        # the 500 ms debounce buffer. Persist it before a startup failure exits.
        _flush_pending()
    except Exception as exc:
        logger.warning("Failed to flush pending watcher events at shutdown: %s", exc)


def start_observer(roots: list[str] | None = None):
    """Start a ``watchdog`` observer on the exact roots the crawl will use."""
    handler = MediaEventHandler()
    observer = Observer()
    roots = list(_discover_roots() if roots is None else roots)
    try:
        for root in roots:
            logger.info("Watching: %s", root)
            observer.schedule(handler, root, recursive=True)
        observer.start()
    except BaseException:
        _stop_observer(observer)
        raise
    return observer


def run_watcher(
    shutdown_event: threading.Event | None = None,
) -> None:
    """Observe first, then reconcile, eliminating the startup event gap."""
    shutdown_event = shutdown_event or _shutdown_event
    roots = _discover_roots()
    observer = start_observer(roots)
    try:
        purge_hidden_dir_entries(shutdown_event=shutdown_event)
        if not shutdown_event.is_set():
            crawl_system(roots, shutdown_event=shutdown_event)
        shutdown_event.wait()
    except KeyboardInterrupt:
        # Retained for callers/tests that invoke run_watcher without installing
        # the process signal handlers used by the executable entry point.
        shutdown_event.set()
    finally:
        if shutdown_event.is_set():
            logger.info("Watcher shutdown requested")
        _stop_observer(observer)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(
        "[WATCHER START] MEMORIA_BATCH_SIZE=%d (env=%r) -> %s mode",
        MEMORIA_BATCH_SIZE,
        os.environ.get("MEMORIA_BATCH_SIZE"),
        "BATCHED (process_media_batch)" if MEMORIA_BATCH_SIZE > 1 else "SINGLE (process_media)",
    )
    _install_shutdown_signal_handlers()
    run_watcher()
