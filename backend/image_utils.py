"""Shared image-processing primitives + cache-path constants.

In plain English: every other file in the backend that needs to *open* a
photo on disk goes through this module. It knows which decoder to use
based on the file extension:

  • Ordinary JPEG / PNG / TIFF / WebP / etc. → Pillow.
  • HEIC / HEIF / AVIF (iPhone-style)        → Pillow + the pillow-heif
                                                plugin registered below.
  • Camera RAW (CR2 / NEF / ARW / DNG / …)    → the rawpy library.

It also keeps the on-disk cache layout constants (where thumbnails,
previews, and face crops live on disk) so both the FastAPI server and the
worker process agree on the paths.

Every callable here is blocking (Pillow / rawpy reads the full image into
memory). Route through an executor or a Huey task.

Format-aware decoders (Phase E1):
  * decode_pil(path)       → PIL.Image (RGB)
  * decode_cv2_bgr(path)   → np.ndarray (BGR)
Both route by extension and handle JPEG/PNG/TIFF + HEIC/HEIF/AVIF (via
``pillow-heif``) + camera RAW (via ``rawpy``). Standalone scripts can also
``from image_utils import decode_pil`` if they need the same coverage.

The cache directory constants live here so both the FastAPI server (for
FileResponse) and the worker (for cache writes) agree on the layout without
needing to import each other.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from runtime_paths import CACHE_DIR as _RUNTIME_CACHE_DIR

logger = logging.getLogger("memoria.image_utils")

# ── Optional decoder libraries (registered once, at import time) ────────────
# pillow-heif teaches PIL.Image.open to decode .heic / .heif / .avif. The
# register call is idempotent; safe to import multiple times.
_HAS_HEIC = False
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HAS_HEIC = True
except ImportError:
    logger.warning(
        "pillow-heif not installed — HEIC/HEIF/AVIF files will fail to decode."
    )

_HAS_RAWPY = False
try:
    import rawpy  # noqa: F401  (probed; imported lazily inside decode_pil)
    _HAS_RAWPY = True
except ImportError:
    logger.warning(
        "rawpy not installed — camera RAW files (CR2 / NEF / ARW / DNG / …) will fail to decode."
    )


# ── Decode-time size cap (Phase E2) ─────────────────────────────────────────
# Big files (50 MP DSLR shots, 8K RAWs) decoded to RGB hold ~600 MB of
# transient memory each. Most pipeline steps (face, scene model, blur FFT) are
# happy with a downsampled view, so we clamp the longest edge here unless
# the caller explicitly asks for full resolution.
#
#   max_dim=None  → use MEMORIA_MAX_DECODE_DIM env var (default 4096)
#   max_dim=0     → no cap (full resolution)
#   max_dim=int   → cap longest edge to this value
try:
    _MAX_DECODE_DIM_DEFAULT = int(os.environ.get("MEMORIA_MAX_DECODE_DIM", "4096"))
except ValueError:
    _MAX_DECODE_DIM_DEFAULT = 4096


# ── Cache layout ────────────────────────────────────────────────────────────
# Persistent, reboot-stable cache. Sweep/eviction is the server's job
# (see _sweep_cache_if_over_budget in main.py).
CACHE_ROOT = _RUNTIME_CACHE_DIR
THUMB_CACHE_DIR = CACHE_ROOT / "thumbs"
PREVIEW_CACHE_DIR = CACHE_ROOT / "previews"
FACE_CROP_CACHE_DIR = CACHE_ROOT / "face_crops"

# ── Size / quality constants ────────────────────────────────────────────────
THUMB_SIZE = (240, 240)
PREVIEW_MAX_DIM = 2048
FACE_CROP_SIZE = (160, 160)
# Bounds check for user-supplied coordinates on the face-crop endpoint.
FACE_CROP_MAX_DIM = 16384
# Extra context to include around the detected face bbox, as a fraction
# of the bbox width/height.
_FACE_CROP_PADDING = 0.4

# Pillow can resize these extensions safely (no decoder library required).
# Kept for backwards-compat callers; new code should import IMAGE_EXTENSIONS
# from constants.py instead.
PILLOW_EXTS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
})

# Local copies of the extension sets so this module doesn't depend on
# constants.py (avoids an import cycle with anything that already imports
# from here). Keep these in lockstep with constants.RAW_EXTS / HEIC_EXTS.
_RAW_EXTS: frozenset[str] = frozenset({
    ".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng",
    ".rw2", ".orf", ".pef", ".raf", ".srw",
})
_HEIC_EXTS: frozenset[str] = frozenset({".heic", ".heif", ".avif"})


def _ext(path: str | Path) -> str:
    """Return the lowercase file extension, including the leading dot."""
    return Path(path).suffix.lower()


# ── Format-aware decoders (Phase E1) ────────────────────────────────────────

def _apply_max_dim(img, max_dim: int | None):
    """In-place downsample if the longest edge exceeds *max_dim*.

    *max_dim* semantics: ``None`` → use env-var default; ``0`` → no cap;
    positive int → clamp longest edge to that value (preserving aspect).
    """
    from PIL import Image

    if max_dim is None:
        max_dim = _MAX_DECODE_DIM_DEFAULT
    if max_dim and max_dim > 0 and max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return img


def decode_pil(path: str | Path, *, max_dim: int | None = None):
    """Decode any supported image format to a PIL.Image in RGB mode.

    Routes by extension:
      * RAW (CR2 / NEF / ARW / DNG / …): rawpy.imread().postprocess() → PIL
      * HEIC / HEIF / AVIF: PIL.Image.open() (pillow-heif registered globally)
      * Everything else: PIL.Image.open()

    Optionally clamps the longest edge to *max_dim* (default from
    ``MEMORIA_MAX_DECODE_DIM``; pass ``0`` to disable). The clamp happens
    AFTER decode, so it bounds peak transient memory per call but doesn't
    speed up the decode itself — that's a libraw / Pillow problem.

    Raises ``RuntimeError`` if a required decoder library is missing for the
    detected format, so callers see a clean message instead of a cryptic
    Pillow ``UnidentifiedImageError``.
    """
    from PIL import Image

    e = _ext(path)
    s = str(path)

    if e in _RAW_EXTS:
        if not _HAS_RAWPY:
            raise RuntimeError(
                f"rawpy is required to decode RAW files (got {path}); "
                f"install via `pip install rawpy>=0.19`."
            )
        import rawpy  # local import keeps module load light
        # `postprocess` is the demosaic-and-render step: turn the raw sensor
        # readings into ordinary RGB pixels. Settings: respect the camera's
        # white balance, allow auto-brightness, output 8-bit-per-channel
        # (matches what every downstream model expects).
        with rawpy.imread(s) as raw:
            arr = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=8,
            )
        img = Image.fromarray(arr)  # rawpy postprocess returns RGB
        return _apply_max_dim(img, max_dim)

    if e in _HEIC_EXTS and not _HAS_HEIC:
        raise RuntimeError(
            f"pillow-heif is required to decode HEIC/HEIF/AVIF (got {path})."
        )

    img = Image.open(s)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _apply_max_dim(img, max_dim)


def decode_cv2_bgr(path: str | Path, *, max_dim: int | None = None) -> np.ndarray:
    """Decode any supported image format to an OpenCV BGR ndarray.

    ``cv2.imread`` is HEIC- and RAW-blind, so this helper falls back to
    ``decode_pil`` (RGB) and converts to BGR for those formats. For plain
    JPEG/PNG/TIFF/etc. it stays on the fast cv2 path.

    Same *max_dim* semantics as :func:`decode_pil`.
    """
    import cv2

    e = _ext(path)
    s = str(path)

    if e in _RAW_EXTS or e in _HEIC_EXTS:
        pil = decode_pil(path, max_dim=max_dim)
        rgb = np.array(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    img = cv2.imread(s, cv2.IMREAD_COLOR)
    if img is not None:
        return _resize_cv2_max_dim(img, max_dim)

    # Last-ditch fallback via Pillow (covers oddball TIFFs cv2 can't read).
    try:
        pil = decode_pil(path, max_dim=max_dim)
        rgb = np.array(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise RuntimeError(f"could not decode image {path}: {exc}") from exc


def _resize_cv2_max_dim(img: np.ndarray, max_dim: int | None) -> np.ndarray:
    """Mirror of _apply_max_dim for cv2 BGR ndarrays."""
    import cv2

    if max_dim is None:
        max_dim = _MAX_DECODE_DIM_DEFAULT
    if not max_dim or max_dim <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / float(longest)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ── Producers (now format-aware via decode_pil) ─────────────────────────────

def make_thumbnail(src: str, dest: Path) -> None:
    """Resize *src* to THUMB_SIZE, save as low-quality JPEG to *dest*.

    Low quality (55) is deliberate — thumbnails feed a 240px virtualized
    grid; any more detail is wasted bandwidth.
    """
    from PIL import Image
    # Pre-cap to the decode default to bound transient memory; the
    # subsequent thumbnail() takes care of the final 240×240 resize.
    img = decode_pil(src)
    img.thumbnail(THUMB_SIZE, Image.LANCZOS)
    img.save(dest, "JPEG", quality=55, optimize=True)


def make_preview(src: str, dest: Path) -> None:
    """Resize *src* to at most PREVIEW_MAX_DIM on its longest side, save as WebP."""
    from PIL import Image
    img = decode_pil(src)
    img.thumbnail((PREVIEW_MAX_DIM, PREVIEW_MAX_DIM), Image.LANCZOS)
    img.save(dest, "WEBP", quality=82, method=4)


def make_face_crop(src: str, dest: Path, x: int, y: int, w: int, h: int) -> None:
    """Crop a padded region around (x, y, w, h), resize to FACE_CROP_SIZE, save as JPEG.

    Bbox coordinates are in original-image pixels, so we decode at full
    resolution (max_dim=0) to avoid silently scaling them out from under
    the caller.
    """
    from PIL import Image
    img = decode_pil(src, max_dim=0)
    iw, ih = img.size
    # Expand the bbox by 40% on each side so the crop includes hair and
    # background context, not just the face rectangle. `max(0, …)` and
    # `min(iw, …)` then clamp the result inside the image — without this
    # the crop would wrap or raise on faces near the edges.
    pad_x = int(w * _FACE_CROP_PADDING)
    pad_y = int(h * _FACE_CROP_PADDING)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(iw, x + w + pad_x)
    bottom = min(ih, y + h + pad_y)
    crop = img.crop((left, top, right, bottom))
    crop = crop.resize(FACE_CROP_SIZE, Image.LANCZOS)
    crop.save(dest, "JPEG", quality=70, optimize=True)
