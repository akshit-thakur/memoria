"""Per-directory sidecar location, load, and atomic-save helpers.

By default, Memoria preserves its native layout and stores
``.memoria_metadata.json`` beside the media in each source directory. A
container can instead set ``MEMORIA_SIDECAR_ROOT`` to a writable persistent
mount while keeping source libraries strictly read-only.

External sidecars live below ``<root>/v1/<sha256 fanout>/``. The digest is of
the canonical source directory, so two libraries with the same directory
names cannot collide, and no absolute source path is exposed in the sidecar
store. The JSON schema remains unchanged and continues to be keyed only by
the media filename.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

from runtime_paths import STATE_DIR

try:  # POSIX production path; Windows keeps the in-process lock fallback.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows hosts
    _fcntl = None

SIDECAR_FILENAME = ".memoria_metadata.json"

logger = logging.getLogger("memoria.sidecar")

# Per-sidecar locks serialize callers' load -> modify -> save cycles. Huey is
# deliberately single-threaded today, but the helper remains safe if an admin
# task and a worker touch different or identical directories concurrently.


class _SidecarLock:
    """Thread lock plus a crash-safe POSIX advisory lock.

    Reconciliation is normally run while the worker is quiet, but the file
    lock prevents a concurrently running admin process from overwriting a
    worker's newer sidecar. Lock filenames are hashes, so the state tree does
    not disclose source paths.
    """

    def __init__(self, key: str, lock_root: Path) -> None:
        self._thread_lock = threading.Lock()
        digest = hashlib.sha256(os.fsencode(key)).hexdigest()
        self._lock_path = lock_root / "locks" / "sidecars" / f"{digest}.lock"
        self._fd: int | None = None

    def __enter__(self) -> "_SidecarLock":
        self._thread_lock.acquire()
        try:
            if _fcntl is not None:
                self._lock_path.parent.mkdir(parents=True, exist_ok=True)
                self._fd = os.open(
                    self._lock_path,
                    os.O_CREAT | os.O_RDWR,
                    0o600,
                )
                _fcntl.flock(self._fd, _fcntl.LOCK_EX)
            return self
        except Exception:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            self._thread_lock.release()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self._fd is not None:
                fd = self._fd
                self._fd = None
                try:
                    if _fcntl is not None:
                        _fcntl.flock(fd, _fcntl.LOCK_UN)
                finally:
                    os.close(fd)
        finally:
            self._thread_lock.release()


_dir_locks: dict[str, _SidecarLock] = {}
_dir_locks_mutex = threading.Lock()


def configured_sidecar_root() -> Path | None:
    """Return the configured external sidecar root, or ``None``.

    Relative values are made absolute against the process working directory.
    Production configuration should use an absolute path, but accepting a
    relative one keeps local tooling convenient.
    """
    raw = os.getenv("MEMORIA_SIDECAR_ROOT", "").strip()
    if not raw:
        return None
    # Resolve the root as well as the source directory.  macOS commonly
    # exposes temporary paths through /var -> /private/var; keeping both
    # sides canonical makes containment checks and lock keys deterministic.
    return Path(raw).expanduser().resolve()


def _canonical_directory(directory: str | os.PathLike[str]) -> str:
    expanded = os.path.expanduser(os.fspath(directory))
    absolute = os.path.abspath(expanded)
    # realpath collapses symlink aliases; normcase makes equivalent Windows
    # spellings share one sidecar without changing Linux case semantics.
    return os.path.normcase(os.path.realpath(absolute))


def _external_sidecar_path(
    root: Path, directory: str | os.PathLike[str]
) -> Path:
    digest = hashlib.sha256(
        os.fsencode(_canonical_directory(directory))
    ).hexdigest()
    return root / "v1" / digest[:2] / digest[2:4] / digest[4:] / SIDECAR_FILENAME


def sidecar_path(directory: str | os.PathLike[str]) -> Path:
    """Return the sidecar path used for *directory* under current config."""
    root = configured_sidecar_root()
    if root is None:
        return Path(directory) / SIDECAR_FILENAME
    return _external_sidecar_path(root, directory)


def get_lock(directory: str) -> _SidecarLock:
    """Return the per-sidecar lock for *directory*, creating it if needed."""
    # Key by actual output path so lexical aliases that resolve to the same
    # external sidecar cannot race each other.
    key = os.path.normcase(str(sidecar_path(directory)))
    lock_root = configured_sidecar_root() or STATE_DIR
    with _dir_locks_mutex:
        if key not in _dir_locks:
            _dir_locks[key] = _SidecarLock(key, lock_root)
        return _dir_locks[key]


def load_sidecar(directory: str) -> dict:
    """Load *directory*'s sidecar; return ``{}`` when absent or corrupt.

    Corrupt files are preserved under a unique ``.corrupt`` suffix where the
    selected sidecar store is writable, rather than being silently deleted.
    """
    path = sidecar_path(directory)
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("sidecar root must be a JSON object")
            return data
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("Corrupt sidecar at %s, resetting: %s", path, exc)
            try:
                corrupt = path.with_name(
                    f"{path.name}.corrupt.{os.getpid()}.{time.time_ns()}"
                )
                os.replace(path, corrupt)
            except OSError:
                pass
    return {}


def save_sidecar(directory: str, data: dict) -> None:
    """Atomically persist *directory*'s sidecar in the selected store."""
    root = configured_sidecar_root()
    path = (
        Path(directory) / SIDECAR_FILENAME
        if root is None
        else _external_sidecar_path(root, directory)
    )
    # Adjacent mode intentionally does not create missing source directories.
    # External mode owns its tree and may safely create the digest fanout.
    if root is not None:
        path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
