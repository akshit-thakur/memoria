"""Helpers for on-demand library folder browse and scan requests."""

from __future__ import annotations

import os
import platform
import string
from pathlib import Path

from constants import EXCLUDED_DIRS, EXCLUDED_DIR_SUFFIXES, is_excluded_path

DEFAULT_CONTAINER_LIBRARY_MOUNT = "/media/library"
MAX_RESOLVE_DEPTH = 24
MAX_PICKED_FILE_SAMPLES = 50
MAX_LIBRARY_SCAN_BODY_BYTES = 512 * 1024
MAX_PICKED_FOLDER_NAME_LENGTH = 512
MAX_PICKED_RELATIVE_PATH_LENGTH = 4096
MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991

WINDOWS_SKIP_DIRNAMES = frozenset({
    "$Recycle.Bin",
    "Boot",
    "PerfLogs",
    "Program Files",
    "Program Files (x86)",
    "ProgramData",
    "Recovery",
    "System Volume Information",
    "WinSxS",
    "Windows",
})


def _normalize(path: str) -> str:
    return os.path.normpath(os.path.abspath(path.strip()))


def _canonical(path: str | Path, *, strict: bool = True) -> Path:
    """Canonicalize a filesystem path, resolving symlinks component-wise."""
    return Path(path).expanduser().resolve(strict=strict)


def _existing_directory(path: Path | str) -> str | None:
    norm = _normalize(str(path))
    return norm if os.path.isdir(norm) else None


def _crawl_roots_from_env() -> list[str]:
    raw = os.getenv("CRAWL_ROOTS", "").strip()
    if not raw:
        return []
    roots: list[str] = []
    for part in raw.split(os.pathsep):
        resolved = _existing_directory(part.strip())
        if resolved:
            roots.append(resolved)
    return roots


def _filesystem_search_roots() -> list[str]:
    """Return every filesystem root Memoria can search by default."""
    mounts: list[str] = []
    seen: set[str] = set()

    def add(path: Path | str) -> None:
        resolved = _existing_directory(path)
        if resolved and resolved not in seen:
            seen.add(resolved)
            mounts.append(resolved)

    system = platform.system()
    if system == "Windows":
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            add(drive)
            add(Path(drive) / "Users")
    elif system == "Darwin":
        add(Path("/"))
        add(Path("/Users"))
        add(Path("/Volumes"))
        add(Path.home())
    else:
        if _in_container():
            # Never recursively probe the container root. Apart from being
            # expensive, it walks application/runtime state that cannot be a
            # source library. Compose exposes only the explicit browse bind
            # and the configured crawl bind here.
            add(Path("/host"))
            add(Path(DEFAULT_CONTAINER_LIBRARY_MOUNT))
        else:
            add(Path("/"))
            add(Path("/home"))
            add(Path("/media"))
            add(Path("/mnt"))
            add(Path.home())

    for root in _crawl_roots_from_env():
        add(root)

    return mounts


def discover_library_mounts() -> list[str]:
    """Return directories searchable by the on-demand folder picker.

    This is intentionally separate from ``CRAWL_ROOTS``, which controls the
    automatic watcher and startup crawl scope.

    In a container, the unset default is limited to ``/host`` and
    ``/media/library``. Native runtimes search the platform's common roots.
    """
    library_mounts_env = os.getenv("MEMORIA_LIBRARY_MOUNTS", "").strip()
    if library_mounts_env:
        mounts: list[str] = []
        for raw in library_mounts_env.split(os.pathsep):
            if not raw.strip():
                continue
            resolved = _existing_directory(raw.strip())
            if resolved:
                mounts.append(resolved)
        # An explicit mount list is an authority boundary. Never fall back to
        # platform-wide discovery when it is present but unusable.
        return mounts

    return _filesystem_search_roots()


def is_within_roots(path: str, roots: list[str]) -> bool:
    """Return True when *path* is equal to or nested under any configured root."""
    norm = Path(os.path.realpath(_normalize(path)))
    for root in roots:
        try:
            root_norm = Path(os.path.realpath(_normalize(root)))
            norm.relative_to(root_norm)
            if not is_excluded_path(str(norm)):
                return True
        except (OSError, RuntimeError, ValueError):
            continue
    return False


def validate_readable_directory(path: str) -> str:
    """Resolve *path* and ensure it is a readable directory."""
    if not path or not isinstance(path, str):
        raise ValueError("path is required")
    try:
        abs_path = _normalize(path)
    except (OSError, RuntimeError, ValueError):
        raise ValueError("path is not a readable directory")
    if is_excluded_path(str(Path(os.path.realpath(abs_path)))) or not os.path.isdir(abs_path):
        raise ValueError("path is not a readable directory")
    return abs_path


def validate_scan_path(path: str, roots: list[str]) -> str:
    """Resolve *path* and ensure it is a directory under configured browse roots."""
    if not roots:
        raise ValueError("no library roots are configured")
    abs_path = validate_readable_directory(path)
    if not is_within_roots(abs_path, roots):
        raise ValueError("path is outside configured library roots")
    return abs_path


def validate_media_source(path: str, roots: list[str] | None = None) -> str:
    """Validate a database-backed source file against current library roots."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("media source is invalid")
    configured = roots if roots is not None else discover_library_mounts()
    if not configured or not is_within_roots(path, configured):
        raise ValueError("media source is outside configured library roots")
    try:
        resolved = _canonical(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("media source is unavailable") from exc
    if is_excluded_path(str(resolved)) or not resolved.is_file():
        raise ValueError("media source is unavailable")
    return str(resolved)


def validate_picked_scan_request(
    folder_name: object,
    files: object,
    total_files: object = None,
) -> tuple[str, list[dict], int]:
    """Validate the bounded browser-folder fingerprint sent to the API.

    Browsers cannot reveal a native absolute path. The client therefore sends
    at most ``MAX_PICKED_FILE_SAMPLES`` relative paths so the server can match
    the selected folder against its configured mounts, plus the full media
    count as scalar metadata. File size and timestamp metadata are neither
    needed nor trusted for this lookup.
    """
    if not isinstance(folder_name, str):
        raise ValueError("folder_name must be a string")
    normalized_name = folder_name.strip()
    if not normalized_name:
        raise ValueError("folder_name is required")
    if len(normalized_name.encode("utf-8")) > MAX_PICKED_FOLDER_NAME_LENGTH:
        raise ValueError("folder_name is too long")
    if normalized_name in {".", ".."} or any(
        separator in normalized_name for separator in ("/", "\\", "\x00")
    ):
        raise ValueError("folder_name must be a single directory name")

    if not isinstance(files, list):
        raise ValueError("files must be a list")
    if not files:
        raise ValueError("the selected folder contains no supported media files")
    if len(files) > MAX_PICKED_FILE_SAMPLES:
        raise ValueError(
            f"files must contain at most {MAX_PICKED_FILE_SAMPLES} sample paths"
        )

    normalized_files: list[dict] = []
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise ValueError(f"files[{index}] must be an object")
        relative_path = item.get("relative_path")
        if not isinstance(relative_path, str):
            raise ValueError(f"files[{index}].relative_path must be a string")
        relative_path = relative_path.strip().replace("\\", "/")
        if not relative_path:
            raise ValueError(f"files[{index}].relative_path is required")
        if len(relative_path.encode("utf-8")) > MAX_PICKED_RELATIVE_PATH_LENGTH:
            raise ValueError(f"files[{index}].relative_path is too long")
        parts = relative_path.split("/")
        if (
            relative_path.startswith("/")
            or "\x00" in relative_path
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError(
                f"files[{index}].relative_path must be a safe relative path"
            )
        normalized_files.append({"relative_path": relative_path})

    if total_files is None:
        normalized_total = len(normalized_files)
    elif (
        isinstance(total_files, bool)
        or not isinstance(total_files, int)
        or total_files < len(normalized_files)
        or total_files > MAX_SAFE_JSON_INTEGER
    ):
        raise ValueError(
            "total_files must be an integer no smaller than the sample size"
        )
    else:
        normalized_total = total_files

    return normalized_name, normalized_files, normalized_total


def _should_skip_dirname(name: str) -> bool:
    if name.startswith("."):
        return True
    if name in EXCLUDED_DIRS:
        return True
    if platform.system() == "Windows" and name in WINDOWS_SKIP_DIRNAMES:
        return True
    return any(name.endswith(suffix) for suffix in EXCLUDED_DIR_SUFFIXES)


def _in_container() -> bool:
    return os.path.isfile("/.dockerenv") or os.getenv("MEMORIA_MEDIA_READ_ONLY", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _collect_folder_candidates(mount: str, folder_name: str) -> list[str]:
    """Find directories named *folder_name* under *mount*."""
    mount_norm = _normalize(mount)
    if not os.path.isdir(mount_norm):
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _add_candidate(path: str) -> None:
        norm = _normalize(path)
        if norm in seen or not os.path.isdir(norm):
            return
        seen.add(norm)
        candidates.append(norm)

    _add_candidate(mount_norm)
    _add_candidate(os.path.join(mount_norm, folder_name))

    try:
        with os.scandir(mount_norm) as iterator:
            for entry in iterator:
                if entry.is_dir(follow_symlinks=False) and entry.name == folder_name:
                    _add_candidate(entry.path)
    except OSError:
        pass

    for dirpath, dirnames, _ in os.walk(mount_norm, topdown=True):
        if dirpath != mount_norm:
            rel_depth = dirpath[len(mount_norm) :].count(os.sep)
            if rel_depth >= MAX_RESOLVE_DEPTH:
                dirnames[:] = []
                continue

        dirnames[:] = [
            name for name in dirnames if not _should_skip_dirname(name)
        ]
        if folder_name in dirnames:
            _add_candidate(os.path.join(dirpath, folder_name))

    return candidates


def resolve_picked_scan_root(
    folder_name: str,
    files: list[dict],
    mounts: list[str] | None = None,
) -> str:
    """Map a browser-picked folder onto a server directory under library mounts."""
    search_mounts = mounts if mounts is not None else discover_library_mounts()
    if not search_mounts:
        raise ValueError("no library mounts are configured")
    if not files:
        raise ValueError("the selected folder contains no supported media files")

    sample_paths = [
        str(item.get("relative_path", "")).replace("\\", "/")
        for item in files[:MAX_PICKED_FILE_SAMPLES]
        if item.get("relative_path")
    ]
    if not sample_paths:
        raise ValueError("the selected folder contains no supported media files")

    candidates: list[str] = []
    seen: set[str] = set()
    for mount in search_mounts:
        for candidate in _collect_folder_candidates(mount, folder_name):
            try:
                norm = _normalize(candidate)
            except (OSError, RuntimeError, ValueError):
                continue
            if norm not in seen:
                seen.add(norm)
                candidates.append(norm)

    best_root: str | None = None
    best_hits = 0
    for candidate in candidates:
        hits = 0
        for rel in sample_paths:
            candidate_file = os.path.join(candidate, rel.replace("/", os.sep))
            try:
                candidate_file = str(_canonical(candidate_file))
            except (OSError, RuntimeError, ValueError):
                continue
            if is_within_roots(candidate_file, search_mounts) and os.path.isfile(candidate_file):
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_root = candidate

    if best_root and best_hits > 0:
        return best_root

    if _in_container():
        raise ValueError(
            "could not locate the selected folder inside this container. "
            "Browsers cannot send full file paths, so Memoria matches your pick "
            "against the read-only MEMORIA_MEDIA_PATH and MEMORIA_HOST_MOUNT "
            "binds. Set MEMORIA_HOST_MOUNT in .env to a narrow parent of the "
            "folder (use forward slashes on Windows), allow that path in "
            "Docker Desktop when prompted, then restart Compose."
        )

    raise ValueError(
        "could not locate the selected folder on this system. "
        "Ensure the folder contains supported media files and is readable by Memoria."
    )


def browse_directory(path: str | None, roots: list[str]) -> dict:
    """Return immediate child directories for the folder picker UI."""
    if not roots:
        raise ValueError("no library roots are configured")

    normalized_roots = [_normalize(root) for root in roots]
    if path:
        current = validate_scan_path(path, roots)
        parent = str(Path(current).parent)
        if not is_within_roots(parent, roots):
            parent = None
    else:
        current = None
        parent = None

    entries: list[dict] = []
    targets = [current] if current else normalized_roots
    seen: set[str] = set()

    for target in targets:
        try:
            with os.scandir(target) as iterator:
                for entry in iterator:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    if entry.name.startswith("."):
                        continue
                    child_path = _normalize(entry.path)
                    if not is_within_roots(child_path, roots):
                        continue
                    if child_path in seen:
                        continue
                    seen.add(child_path)
                    has_children = False
                    try:
                        with os.scandir(child_path) as child_iter:
                            has_children = any(
                                child.is_dir(follow_symlinks=False)
                                and not child.name.startswith(".")
                                for child in child_iter
                            )
                    except OSError:
                        has_children = False
                    entries.append(
                        {
                            "name": entry.name,
                            "path": child_path,
                            "has_children": has_children,
                        }
                    )
        except OSError as exc:
            raise ValueError(f"cannot read directory: {exc}") from exc

    entries.sort(key=lambda item: item["name"].lower())
    return {
        "current": current,
        "parent": parent,
        "roots": normalized_roots,
        "entries": entries,
    }
