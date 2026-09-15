import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from library_paths import (
    MAX_PICKED_FILE_SAMPLES,
    _filesystem_search_roots,
    browse_directory,
    discover_library_mounts,
    is_within_roots,
    resolve_picked_scan_root,
    validate_picked_scan_request,
    validate_readable_directory,
    validate_scan_path,
)


class LibraryPathsTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _path(self, *parts: str) -> str:
        target = os.path.join(self._tmpdir, *parts)
        return os.path.abspath(target)

    def test_validate_scan_path_rejects_outside_roots(self):
        roots = [self._path("library")]
        os.makedirs(roots[0], exist_ok=True)
        with self.assertRaises(ValueError):
            validate_scan_path("/etc/passwd", roots)

    def test_validate_readable_directory_accepts_any_readable_path(self):
        folder = self._path("picked")
        os.makedirs(folder, exist_ok=True)
        self.assertEqual(validate_readable_directory(folder), folder)

    def test_is_within_roots_accepts_nested_path(self):
        roots = ["/media/library"]
        self.assertTrue(is_within_roots("/media/library/album", roots))

    def test_browse_directory_lists_roots_when_no_path(self):
        root = self._path("library")
        os.makedirs(root, exist_ok=True)
        payload = browse_directory(None, [root])
        self.assertIsNone(payload["current"])
        self.assertEqual(payload["roots"], [root])
        self.assertEqual(payload["entries"], [])

    def test_browse_directory_lists_child_directories(self):
        root = self._path("library")
        album = self._path("library", "Vacation")
        nested = self._path("library", "Vacation", "2024")
        os.makedirs(nested, exist_ok=True)
        with open(os.path.join(album, "photo.jpg"), "wb") as handle:
            handle.write(b"test")

        payload = browse_directory(root, [root])
        self.assertEqual(payload["current"], root)
        self.assertEqual(len(payload["entries"]), 1)
        self.assertEqual(payload["entries"][0]["name"], "Vacation")
        self.assertEqual(payload["entries"][0]["path"], album)
        self.assertTrue(payload["entries"][0]["has_children"])

    def test_resolve_picked_scan_root_matches_nested_folder(self):
        root = self._path("library")
        album = self._path("library", "Vacation")
        os.makedirs(album, exist_ok=True)
        photo = os.path.join(album, "photo.jpg")
        with open(photo, "wb") as handle:
            handle.write(b"test")

        resolved = resolve_picked_scan_root(
            "Vacation",
            [{"relative_path": "photo.jpg", "size": 4, "mtime_ms": 1}],
            [root],
        )
        self.assertEqual(resolved, album)

    def test_validate_picked_scan_request_keeps_bounded_paths_and_total(self):
        folder_name, files, total = validate_picked_scan_request(
            " Vacation ",
            [{"relative_path": "nested\\photo.jpg", "size": 999}],
            123_456,
        )
        self.assertEqual(folder_name, "Vacation")
        self.assertEqual(files, [{"relative_path": "nested/photo.jpg"}])
        self.assertEqual(total, 123_456)

    def test_validate_picked_scan_request_rejects_oversized_sample(self):
        files = [
            {"relative_path": f"photo-{index}.jpg"}
            for index in range(MAX_PICKED_FILE_SAMPLES + 1)
        ]
        with self.assertRaisesRegex(ValueError, "at most"):
            validate_picked_scan_request("Vacation", files, len(files))

    def test_validate_picked_scan_request_rejects_unsafe_shape(self):
        with self.assertRaisesRegex(ValueError, "single directory name"):
            validate_picked_scan_request("../Vacation", [{"relative_path": "photo.jpg"}], 1)
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            validate_picked_scan_request("Vacation", [{"relative_path": "../photo.jpg"}], 1)
        with self.assertRaisesRegex(ValueError, "no smaller"):
            validate_picked_scan_request(
                "Vacation",
                [{"relative_path": "one.jpg"}, {"relative_path": "two.jpg"}],
                1,
            )

    def test_resolve_picked_scan_root_matches_mount_root(self):
        root = self._path("Photos")
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "photo.jpg"), "wb") as handle:
            handle.write(b"test")

        resolved = resolve_picked_scan_root(
            "Photos",
            [{"relative_path": "photo.jpg", "size": 4, "mtime_ms": 1}],
            [root],
        )
        self.assertEqual(resolved, root)

    def test_resolve_picked_scan_root_finds_nested_folder_outside_crawl_subset(self):
        mount = self._path("media", "library")
        crawl_subset = self._path("media", "library", "synced")
        album = self._path("media", "library", "Vacation")
        os.makedirs(crawl_subset, exist_ok=True)
        os.makedirs(album, exist_ok=True)
        with open(os.path.join(album, "photo.jpg"), "wb") as handle:
            handle.write(b"test")

        resolved = resolve_picked_scan_root(
            "Vacation",
            [{"relative_path": "photo.jpg", "size": 4, "mtime_ms": 1}],
            [mount],
        )
        self.assertEqual(resolved, album)

    def test_discover_library_mounts_prefers_env_override(self):
        override = self._path("override")
        os.makedirs(override, exist_ok=True)
        with mock.patch.dict(os.environ, {"MEMORIA_LIBRARY_MOUNTS": override}, clear=False):
            self.assertEqual(discover_library_mounts(), [override])

    def test_filesystem_search_roots_includes_home(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("library_paths._existing_directory") as existing:
                home = self._path("home")
                existing.side_effect = lambda path: home if str(path).endswith("home") else None
                with mock.patch("library_paths.Path.home", return_value=Path(home)):
                    with mock.patch("library_paths.platform.system", return_value="Darwin"):
                        mounts = _filesystem_search_roots()
        self.assertIn(home, mounts)

    def test_container_search_roots_never_include_container_root(self):
        visible = {"/host", "/media/library", "/"}
        with mock.patch.dict(os.environ, {"CRAWL_ROOTS": ""}, clear=True):
            with mock.patch(
                "library_paths._existing_directory",
                side_effect=lambda path: str(path) if str(path) in visible else None,
            ):
                with mock.patch("library_paths.platform.system", return_value="Linux"):
                    with mock.patch("library_paths._in_container", return_value=True):
                        mounts = _filesystem_search_roots()
        self.assertEqual(mounts, ["/host", "/media/library"])
        self.assertNotIn("/", mounts)


if __name__ == "__main__":
    unittest.main()
