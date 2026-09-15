"""Shared filesystem exclusion + media-extension constants.

In plain English: this file is the single source of truth for two things:
  1. Which file extensions count as "media" (and whether they're an image or
     a video). Used by the watcher to decide what to enqueue, by reconcile
     scripts to decide what to look at, and by the API to filter results.
  2. Which directories the crawler must never walk into (system folders,
     bundle packages, dotfiles like .git/.memoria, etc.).

Every scanner and reconciler imports from this module. Don't redeclare these
sets locally anywhere else — drift between copies has silently broken
reconciliation before.
"""

from __future__ import annotations

from pathlib import Path

# ── Image format taxonomy ───────────────────────────────────────────────────
# The image extensions are split into three tiers because they each need a
# different decoder. The union (IMAGE_EXTENSIONS) is what the pipeline
# actually exposes to callers; the individual sets exist so image_utils can
# route to the right decoder.

# Standard formats Pillow / OpenCV can read directly.
PILLOW_BASIC_EXTS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
})

# Apple/HEVC-family. Need pillow-heif registered (handled in image_utils).
HEIC_EXTS: frozenset[str] = frozenset({
    ".heic", ".heif", ".avif",
})

# Camera RAW. Need rawpy/libraw to decode (lazy import in image_utils).
RAW_EXTS: frozenset[str] = frozenset({
    ".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng",
    ".rw2", ".orf", ".pef", ".raf", ".srw",
})

# Union of formats the pipeline can actually decode.
IMAGE_EXTENSIONS: frozenset[str] = PILLOW_BASIC_EXTS | HEIC_EXTS | RAW_EXTS

# Video — the watcher discovers them but ``process_media`` short-circuits
# (keyframe extraction is a future phase).
VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv",
})

# File suffixes we consider "media" and therefore index. Lowercase with leading
# dot, matched via Path(p).suffix.lower(). Watcher gates on this; the worker
# differentiates via is_image / is_video below.
MEDIA_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# Directory-name components that short-circuit a walk. Any path part matching
# one of these is treated as opaque and skipped. Also note: every file-system
# scanner additionally skips names that start with a dot.
EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "Library", "System", "Applications",
    "virmach", ".memoria", "venv", "__pycache__", ".github", ".vscode",
    "cache",
})

# Bundle / package suffixes. macOS .photoslibrary, .app, etc. are opaque —
# walking into them yields thousands of internal files we never want to index.
EXCLUDED_DIR_SUFFIXES: frozenset[str] = frozenset({
    ".photoslibrary", ".app", ".bundle", ".framework", ".plugin",
    ".xcodeproj", ".xcworkspace",
})

# Absolute-path prefixes. Paths beginning with any of these are skipped
# outright, belt-and-braces alongside the dir-name check above.
EXCLUDE_PATH_PREFIXES: frozenset[str] = frozenset({
    str(Path.home() / "Library"),
})


def is_media(path: str) -> bool:
    """True if *path* has a known media extension (image or video)."""
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def is_image(path: str) -> bool:
    """True if *path* is a still image we can decode (jpeg / heic / raw / etc.)."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: str) -> bool:
    """True if *path* is a video file (no image-pipeline path yet)."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_excluded_path(path: str) -> bool:
    """Return True if *path* should not be indexed.

    A path is excluded when any of these is true:
      • one of its components matches `EXCLUDED_DIRS` (exact name);
      • one of its components ends with a known bundle suffix
        (e.g. `Photos Library.photoslibrary`);
      • any component starts with a dot (hidden file/folder convention);
      • the absolute path starts with one of `EXCLUDE_PATH_PREFIXES`.
    """
    parts = Path(path).parts
    for part in parts:
        if part in EXCLUDED_DIRS:
            return True
        if any(part.endswith(sfx) for sfx in EXCLUDED_DIR_SUFFIXES):
            return True
        # Dotfiles / dot-dirs convention — ".git", ".cache", ".config", etc.
        # Catches anything that begins with `.`, which is intentional.
        if part.startswith("."):
            return True
    for prefix in EXCLUDE_PATH_PREFIXES:
        if path.startswith(prefix):
            return True
    return False
