"""Regression tests for watcher root discovery and scan-index isolation."""

from __future__ import annotations

import os
import signal
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


def _import_watcher_without_runtime_services():
    """Import scan helpers without opening queue state or loading service deps."""

    fake_psycopg = types.ModuleType("psycopg")
    fake_watchdog = types.ModuleType("watchdog")
    fake_watchdog.__path__ = []
    fake_observers = types.ModuleType("watchdog.observers")
    fake_events = types.ModuleType("watchdog.events")

    class FakeObserver:
        pass

    class FakeFileSystemEventHandler:
        pass

    fake_observers.Observer = FakeObserver
    fake_events.FileSystemEventHandler = FakeFileSystemEventHandler

    fake_workers = types.ModuleType("workers")
    fake_workers.__path__ = []
    fake_tasks = types.ModuleType("workers.tasks")
    fake_tasks.delete_media = lambda *_args, **_kwargs: None
    fake_tasks.process_media = lambda *_args, **_kwargs: None
    fake_tasks.process_media_batch = lambda *_args, **_kwargs: None

    stubs = {
        "psycopg": fake_psycopg,
        "watchdog": fake_watchdog,
        "watchdog.observers": fake_observers,
        "watchdog.events": fake_events,
        "workers": fake_workers,
        "workers.tasks": fake_tasks,
    }
    previous_watcher = sys.modules.pop("watcher", None)
    try:
        with mock.patch.dict(sys.modules, stubs):
            import watcher as watcher_module
    finally:
        # Leave other tests free to import the real service module and dependencies.
        sys.modules.pop("watcher", None)
        if previous_watcher is not None:
            sys.modules["watcher"] = previous_watcher
    return watcher_module


watcher = _import_watcher_without_runtime_services()


class WatcherRootIsolationTests(unittest.TestCase):
    def test_incremental_scan_prunes_only_the_scanned_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_a = Path(tmp) / "library-a"
            root_b = Path(tmp) / "library-b"
            root_a.mkdir()
            root_b.mkdir()

            db = sqlite3.connect(":memory:")
            self.addCleanup(db.close)
            watcher._init_scan_index(db)
            db.executemany(
                "INSERT INTO scan_index(path, mtime, size, last_seen_at) "
                "VALUES (?, 1, 1, 0)",
                [
                    (str(root_a / "deleted.jpg"),),
                    (str(root_b / "not-scanned.jpg"),),
                ],
            )
            db.commit()

            deleted: list[str] = []
            watcher.incremental_scan(
                str(root_a),
                db,
                lambda _paths: None,
                delete_fn=lambda path: deleted.append(path) or True,
            )

            remaining = {
                row[0] for row in db.execute("SELECT path FROM scan_index").fetchall()
            }
            self.assertNotIn(str(root_a / "deleted.jpg"), remaining)
            self.assertIn(str(root_b / "not-scanned.jpg"), remaining)
            self.assertEqual(deleted, [str(root_a / "deleted.jpg")])

    def test_invalid_explicit_crawl_roots_do_not_fall_back_to_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_a = str(Path(tmp) / "missing-a")
            missing_b = str(Path(tmp) / "missing-b")
            configured = os.pathsep.join((missing_a, missing_b))

            with mock.patch.object(watcher, "CRAWL_ROOTS_ENV", configured):
                self.assertEqual(watcher._discover_roots(), [])

    def test_unreadable_subtree_is_not_pruned_as_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            unreadable = root / "offline-volume"
            root.mkdir()
            unreadable.mkdir()

            db = sqlite3.connect(":memory:")
            self.addCleanup(db.close)
            watcher._init_scan_index(db)
            stale_accessible = str(root / "deleted.jpg")
            stale_unreadable = str(unreadable / "existing.jpg")
            db.executemany(
                "INSERT INTO scan_index(path, mtime, size, last_seen_at) "
                "VALUES (?, 1, 1, 0)",
                [(stale_accessible,), (stale_unreadable,)],
            )
            db.commit()

            real_scandir = os.scandir

            def guarded_scandir(path):
                if os.path.normpath(path) == os.path.normpath(unreadable):
                    raise PermissionError("test subtree is unavailable")
                return real_scandir(path)

            deleted: list[str] = []
            with mock.patch.object(watcher.os, "scandir", side_effect=guarded_scandir):
                watcher.incremental_scan(
                    str(root),
                    db,
                    lambda _paths: None,
                    delete_fn=lambda path: deleted.append(path) or True,
                )

            remaining = {
                row[0] for row in db.execute("SELECT path FROM scan_index").fetchall()
            }
            self.assertNotIn(stale_accessible, remaining)
            self.assertIn(stale_unreadable, remaining)
            self.assertEqual(deleted, [stale_accessible])

    def test_failed_offline_delete_enqueue_preserves_scan_index_for_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            stale = str(root / "deleted.jpg")

            db = sqlite3.connect(":memory:")
            self.addCleanup(db.close)
            watcher._init_scan_index(db)
            db.execute(
                "INSERT INTO scan_index(path, mtime, size, last_seen_at) "
                "VALUES (?, 1, 1, 0)",
                (stale,),
            )
            db.commit()

            watcher.incremental_scan(
                str(root),
                db,
                lambda _paths: None,
                delete_fn=lambda _path: False,
            )

            remaining = db.execute(
                "SELECT path FROM scan_index"
            ).fetchall()
            self.assertEqual(remaining, [(stale,)])

    def test_failed_scan_dispatch_removes_index_row_for_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            media = root / "new.jpg"
            media.write_bytes(b"image bytes")

            db = sqlite3.connect(":memory:")
            self.addCleanup(db.close)
            watcher._init_scan_index(db)

            enqueued = watcher.incremental_scan(
                str(root),
                db,
                lambda _paths: set(),
            )

            self.assertEqual(enqueued, 0)
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM scan_index WHERE path = ?",
                    (str(media),),
                ).fetchone()
            )

    def test_none_scan_dispatch_result_remains_backward_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            media = root / "new.jpg"
            media.write_bytes(b"image bytes")

            db = sqlite3.connect(":memory:")
            self.addCleanup(db.close)
            watcher._init_scan_index(db)

            enqueued = watcher.incremental_scan(
                str(root),
                db,
                lambda _paths: None,
            )

            self.assertEqual(enqueued, 1)
            self.assertEqual(
                db.execute(
                    "SELECT path FROM scan_index WHERE path = ?",
                    (str(media),),
                ).fetchone(),
                (str(media),),
            )

    def test_partial_scan_dispatch_keeps_only_accepted_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            accepted = root / "accepted.jpg"
            rejected = root / "rejected.jpg"
            accepted.write_bytes(b"accepted")
            rejected.write_bytes(b"rejected")

            db = sqlite3.connect(":memory:")
            self.addCleanup(db.close)
            watcher._init_scan_index(db)

            enqueued = watcher.incremental_scan(
                str(root),
                db,
                lambda _paths: {str(accepted)},
            )

            self.assertEqual(enqueued, 1)
            remaining = {
                row[0] for row in db.execute("SELECT path FROM scan_index").fetchall()
            }
            self.assertEqual(remaining, {str(accepted)})

    def test_cancelled_scan_flushes_batch_without_pruning_unvisited_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            media = root / "new.jpg"
            media.write_bytes(b"image bytes")
            stale = str(root / "possibly-still-present.jpg")

            db = sqlite3.connect(":memory:")
            self.addCleanup(db.close)
            watcher._init_scan_index(db)
            db.execute(
                "INSERT INTO scan_index(path, mtime, size, last_seen_at) "
                "VALUES (?, 1, 1, 0)",
                (stale,),
            )
            db.commit()

            shutdown_event = threading.Event()
            enqueued: list[str] = []
            deleted: list[str] = []

            def enqueue_then_stop(paths):
                enqueued.extend(paths)
                shutdown_event.set()
                return set(paths)

            accepted = watcher.incremental_scan(
                str(root),
                db,
                enqueue_then_stop,
                delete_fn=lambda path: deleted.append(path) or True,
                shutdown_event=shutdown_event,
            )

            self.assertEqual(accepted, 1)
            self.assertEqual(enqueued, [str(media)])
            self.assertEqual(deleted, [])
            remaining = {
                row[0] for row in db.execute("SELECT path FROM scan_index")
            }
            self.assertEqual(remaining, {str(media), stale})

    def test_file_recreated_after_walk_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            recreated = root / "recreated.jpg"

            db = sqlite3.connect(":memory:")
            self.addCleanup(db.close)
            watcher._init_scan_index(db)
            db.execute(
                "INSERT INTO scan_index(path, mtime, size, last_seen_at) "
                "VALUES (?, 1, 1, 0)",
                (str(recreated),),
            )
            db.commit()

            original_stage = watcher._stage_stale_index_paths

            def stage_then_recreate(*args, **kwargs):
                count = original_stage(*args, **kwargs)
                recreated.write_bytes(b"new image bytes")
                return count

            deleted: list[str] = []
            with mock.patch.object(
                watcher,
                "_stage_stale_index_paths",
                side_effect=stage_then_recreate,
            ):
                watcher.incremental_scan(
                    str(root),
                    db,
                    lambda _paths: None,
                    delete_fn=lambda path: deleted.append(path) or True,
                )

            self.assertEqual(deleted, [])
            row = db.execute(
                "SELECT size, last_seen_at FROM scan_index WHERE path = ?",
                (str(recreated),),
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], len(b"new image bytes"))
            self.assertGreater(row[1], 0)

    def test_pending_observer_claim_is_not_reported_as_durable(self):
        path = "/test-library/new.jpg"
        handler = watcher.MediaEventHandler()
        with (
            mock.patch.object(watcher, "_enqueue_seen", watcher._TTLSet(30)),
            mock.patch.object(watcher, "_dispatch_paths") as dispatch,
            mock.patch.object(watcher, "_debounced_enqueue") as debounce,
        ):
            handler._enqueue(path)
            accepted = watcher._dispatch_discovered_paths([path])

        debounce.assert_called_once_with(path)
        dispatch.assert_not_called()
        self.assertEqual(accepted, set())

    def test_pending_observer_failure_cannot_strand_scan_index_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            media = root / "new.jpg"
            media.write_bytes(b"image bytes")
            path = str(media)

            db = sqlite3.connect(":memory:")
            self.addCleanup(db.close)
            watcher._init_scan_index(db)

            claims = watcher._TTLSet(30)
            self.assertTrue(claims.add(path))
            with watcher._pending_lock:
                watcher._pending_paths.clear()
                watcher._pending_paths.append(path)

            try:
                with (
                    mock.patch.object(watcher, "_enqueue_seen", claims),
                    mock.patch.object(watcher, "_dispatch_paths") as crawl_dispatch,
                ):
                    enqueued = watcher.incremental_scan(
                        str(root),
                        db,
                        watcher._dispatch_discovered_paths,
                    )

                self.assertEqual(enqueued, 0)
                crawl_dispatch.assert_not_called()
                self.assertIsNone(
                    db.execute(
                        "SELECT 1 FROM scan_index WHERE path = ?",
                        (path,),
                    ).fetchone()
                )

                # Even if the observer's queue write and best-effort external
                # cleanup both fail, the crawl transaction already omitted the
                # row and the next scan can retry it.
                with (
                    mock.patch.object(watcher, "_enqueue_seen", claims),
                    mock.patch.object(watcher, "_dispatch_paths", return_value=set()),
                    mock.patch.object(
                        watcher,
                        "_forget_scan_index_paths",
                        return_value=False,
                    ) as forget,
                ):
                    watcher._flush_pending()

                forget.assert_called_once_with({path})
                self.assertTrue(claims.add(path))
                self.assertIsNone(
                    db.execute(
                        "SELECT 1 FROM scan_index WHERE path = ?",
                        (path,),
                    ).fetchone()
                )
            finally:
                with watcher._pending_lock:
                    watcher._pending_paths.clear()

    def test_failed_crawl_dispatch_releases_ttl_claim(self):
        path = "/test-library/retry.jpg"
        claims = watcher._TTLSet(30)
        with (
            mock.patch.object(watcher, "_enqueue_seen", claims),
            mock.patch.object(watcher, "_dispatch_paths", return_value=set()),
        ):
            accepted = watcher._dispatch_discovered_paths([path])
            self.assertEqual(accepted, set())
            self.assertTrue(claims.add(path))

    def test_none_crawl_dispatch_result_remains_backward_compatible(self):
        path = "/test-library/accepted.jpg"
        claims = watcher._TTLSet(30)
        with (
            mock.patch.object(watcher, "_enqueue_seen", claims),
            mock.patch.object(watcher, "_dispatch_paths", return_value=None),
        ):
            accepted = watcher._dispatch_discovered_paths([path])

        self.assertEqual(accepted, {path})
        self.assertFalse(claims.add(path))

    def test_failed_live_dispatch_releases_claim_and_forgets_scan_row(self):
        path = "/test-library/retry.jpg"
        claims = watcher._TTLSet(30)
        self.assertTrue(claims.add(path))
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "scan-index.sqlite"
            with sqlite3.connect(index_path) as db:
                watcher._init_scan_index(db)
                db.execute(
                    "INSERT INTO scan_index(path, mtime, size, last_seen_at) "
                    "VALUES (?, 1, 1, 1)",
                    (path,),
                )
                db.commit()

            with watcher._pending_lock:
                watcher._pending_paths.clear()
                watcher._pending_paths.append(path)
            try:
                with (
                    mock.patch.object(watcher, "_enqueue_seen", claims),
                    mock.patch.object(watcher, "_dispatch_paths", return_value=set()),
                    mock.patch.object(watcher, "SCAN_INDEX_DB_PATH", index_path),
                ):
                    watcher._flush_pending()
                    self.assertTrue(claims.add(path))

                with sqlite3.connect(index_path) as db:
                    self.assertIsNone(
                        db.execute(
                            "SELECT 1 FROM scan_index WHERE path = ?",
                            (path,),
                        ).fetchone()
                    )
            finally:
                with watcher._pending_lock:
                    watcher._pending_paths.clear()

    def test_shutdown_flush_skips_scan_index_lock_retries(self):
        path = "/test-library/retry-after-restart.jpg"
        claims = watcher._TTLSet(30)
        self.assertTrue(claims.add(path))
        shutdown_event = threading.Event()
        shutdown_event.set()
        with watcher._pending_lock:
            watcher._pending_paths.clear()
            watcher._pending_paths.append(path)

        try:
            with (
                mock.patch.object(watcher, "_enqueue_seen", claims),
                mock.patch.object(watcher, "_shutdown_event", shutdown_event),
                mock.patch.object(watcher, "_dispatch_paths", return_value=set()),
                mock.patch.object(watcher, "_forget_scan_index_paths") as forget,
            ):
                watcher._flush_pending()

            forget.assert_not_called()
            self.assertTrue(claims.add(path))
        finally:
            with watcher._pending_lock:
                watcher._pending_paths.clear()

    def test_dispatch_paths_reports_per_path_acceptance(self):
        paths = ["/test-library/ok.jpg", "/test-library/fail.jpg"]
        with (
            mock.patch.object(watcher, "MEMORIA_BATCH_SIZE", 1),
            mock.patch.object(
                watcher,
                "_safe_enqueue",
                side_effect=lambda path: path.endswith("ok.jpg"),
            ),
        ):
            self.assertEqual(watcher._dispatch_paths(paths), {paths[0]})

    def test_dispatch_paths_reports_partial_batch_acceptance(self):
        paths = [f"/test-library/{index}.jpg" for index in range(4)]
        with (
            mock.patch.object(watcher, "MEMORIA_BATCH_SIZE", 2),
            mock.patch.object(
                watcher,
                "_safe_enqueue_batch",
                side_effect=[True, False],
            ) as enqueue_batch,
        ):
            self.assertEqual(watcher._dispatch_paths(paths), set(paths[:2]))

        self.assertEqual(
            enqueue_batch.call_args_list,
            [mock.call(paths[:2]), mock.call(paths[2:])],
        )

    def test_safe_enqueue_helpers_report_queue_failures(self):
        with mock.patch.object(
            watcher,
            "process_media",
            side_effect=RuntimeError("queue unavailable"),
        ):
            self.assertFalse(watcher._safe_enqueue("/test-library/fail.jpg"))

        with mock.patch.object(
            watcher,
            "process_media_batch",
            side_effect=RuntimeError("queue unavailable"),
        ):
            self.assertFalse(
                watcher._safe_enqueue_batch(["/test-library/fail.jpg"])
            )

    def test_run_watcher_starts_observer_before_crawl_and_stops_cleanly(self):
        calls: list[str] = []
        observer = mock.Mock()
        shutdown_event = threading.Event()

        def start(roots):
            calls.append("observer")
            self.assertEqual(roots, ["/test-library"])
            return observer

        def crawl(roots, shutdown_event=None):
            calls.append("crawl")
            self.assertEqual(roots, ["/test-library"])
            self.assertIsNotNone(shutdown_event)
            shutdown_event.set()

        with (
            mock.patch.object(watcher, "_discover_roots", return_value=["/test-library"]),
            mock.patch.object(watcher, "start_observer", side_effect=start),
            mock.patch.object(watcher, "purge_hidden_dir_entries"),
            mock.patch.object(watcher, "crawl_system", side_effect=crawl),
        ):
            watcher.run_watcher(shutdown_event=shutdown_event)

        self.assertEqual(calls, ["observer", "crawl"])
        observer.stop.assert_called_once_with()
        observer.join.assert_called_once_with()

    def test_shutdown_signal_handler_only_sets_the_event(self):
        shutdown_event = threading.Event()
        with mock.patch.object(watcher, "_shutdown_event", shutdown_event):
            watcher._request_shutdown(signal.SIGTERM, None)

        self.assertTrue(shutdown_event.is_set())

    def test_watcher_installs_cooperative_int_and_term_handlers(self):
        with mock.patch.object(watcher.signal, "signal") as install:
            watcher._install_shutdown_signal_handlers()

        install.assert_has_calls(
            [
                mock.call(signal.SIGINT, watcher._request_shutdown),
                mock.call(signal.SIGTERM, watcher._request_shutdown),
            ]
        )

    def test_run_watcher_stops_observer_when_initial_crawl_fails(self):
        observer = mock.Mock()
        with (
            mock.patch.object(watcher, "_discover_roots", return_value=["/test-library"]),
            mock.patch.object(watcher, "start_observer", return_value=observer),
            mock.patch.object(watcher, "purge_hidden_dir_entries"),
            mock.patch.object(
                watcher,
                "crawl_system",
                side_effect=RuntimeError("crawl failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "crawl failed"):
                watcher.run_watcher()

        observer.stop.assert_called_once_with()
        observer.join.assert_called_once_with()

    def test_start_observer_cleans_up_partial_schedule_failure(self):
        observer = mock.Mock()
        observer.schedule.side_effect = RuntimeError("schedule failed")
        with mock.patch.object(watcher, "Observer", return_value=observer):
            with self.assertRaisesRegex(RuntimeError, "schedule failed"):
                watcher.start_observer(["/test-library"])

        observer.stop.assert_called_once_with()
        observer.join.assert_called_once_with()
        observer.start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
