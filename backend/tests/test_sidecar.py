"""Dependency-free checks for native and external sidecar storage modes."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sidecar
from runtime_paths import media_is_read_only


class SidecarStorageTests(unittest.TestCase):
    def test_native_mode_keeps_adjacent_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop("MEMORIA_SIDECAR_ROOT", None)
            media_dir = Path(temp) / "library" / "album"
            media_dir.mkdir(parents=True)
            payload = {"photo.jpg": {"sidecar_state": "synced"}}

            sidecar.save_sidecar(str(media_dir), payload)

            self.assertEqual(
                sidecar.sidecar_path(str(media_dir)),
                media_dir / sidecar.SIDECAR_FILENAME,
            )
            self.assertEqual(sidecar.load_sidecar(str(media_dir)), payload)

    def test_external_mode_is_private_deterministic_and_collision_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            store = base / "state" / "sidecars"
            first = base / "library-a" / "same-album-name"
            second = base / "library-b" / "same-album-name"
            first.mkdir(parents=True)
            second.mkdir(parents=True)

            with patch.dict(
                os.environ, {"MEMORIA_SIDECAR_ROOT": str(store)}, clear=False
            ):
                first_path = sidecar.sidecar_path(str(first))
                second_path = sidecar.sidecar_path(str(second))
                self.assertEqual(first_path, sidecar.sidecar_path(str(first)))
                self.assertNotEqual(first_path, second_path)
                self.assertTrue(first_path.is_relative_to(store.resolve()))
                self.assertNotIn("library-a", str(first_path))
                self.assertNotIn("same-album-name", str(first_path))

                relative = first_path.relative_to(store.resolve())
                self.assertEqual(relative.parts[0], "v1")
                digest = "".join(relative.parts[1:-1])
                self.assertEqual(len(digest), 64)
                self.assertTrue(all(c in "0123456789abcdef" for c in digest))

                payload = {"photo.jpg": {"sidecar_state": "pending"}}
                sidecar.save_sidecar(str(first), payload)
                self.assertEqual(sidecar.load_sidecar(str(first)), payload)
                self.assertFalse((first / sidecar.SIDECAR_FILENAME).exists())

    def test_corrupt_external_sidecar_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            media_dir = base / "library"
            media_dir.mkdir()
            with patch.dict(
                os.environ,
                {"MEMORIA_SIDECAR_ROOT": str(base / "sidecars")},
                clear=False,
            ):
                path = sidecar.sidecar_path(str(media_dir))
                path.parent.mkdir(parents=True)
                path.write_text("{not-json", encoding="utf-8")

                self.assertEqual(sidecar.load_sidecar(str(media_dir)), {})
                self.assertFalse(path.exists())
                self.assertEqual(
                    len(list(path.parent.glob(f"{path.name}.corrupt.*"))), 1
                )

    def test_external_save_never_writes_inside_source_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            media_dir = base / "read-only-library" / "album"
            media_dir.mkdir(parents=True)
            store = base / "writable-state" / "sidecars"
            adjacent = media_dir / sidecar.SIDECAR_FILENAME
            original_open = Path.open

            def reject_source_open(path: Path, *args, **kwargs):
                if path == adjacent or media_dir in path.parents:
                    raise AssertionError(f"attempted source-library write: {path}")
                return original_open(path, *args, **kwargs)

            with patch.dict(
                os.environ, {"MEMORIA_SIDECAR_ROOT": str(store)}, clear=False
            ), patch.object(Path, "open", reject_source_open):
                payload = {"photo.jpg": {"sidecar_state": "synced"}}
                with sidecar.get_lock(str(media_dir)):
                    sidecar.save_sidecar(str(media_dir), payload)
                self.assertEqual(sidecar.load_sidecar(str(media_dir)), payload)

            self.assertFalse(adjacent.exists())

    def test_media_read_only_flag_defaults_false_and_accepts_true_values(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMORIA_MEDIA_READ_ONLY", None)
            self.assertFalse(media_is_read_only())
            os.environ["MEMORIA_MEDIA_READ_ONLY"] = "yes"
            self.assertTrue(media_is_read_only())

    def test_api_read_only_guard_returns_clear_conflict(self) -> None:
        try:
            import routers
        except ImportError as exc:
            self.skipTest(f"backend runtime dependency unavailable: {exc}")

        with patch.dict(
            os.environ, {"MEMORIA_MEDIA_READ_ONLY": "1"}, clear=False
        ):
            response = routers._read_only_media_guard()

        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 409)
        body = json.loads(response.body)
        self.assertEqual(body["code"], "media_read_only")
        self.assertIn("read-only", body["detail"])

        with patch.dict(
            os.environ, {"MEMORIA_MEDIA_READ_ONLY": "true"}, clear=False
        ):
            blurry = asyncio.run(routers.blurry_delete(1, None))
            dedup = asyncio.run(routers.dedup_delete_duplicates(1, None))
        self.assertEqual(blurry.status_code, 409)
        self.assertEqual(dedup.status_code, 409)


if __name__ == "__main__":
    unittest.main()
