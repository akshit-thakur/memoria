"""
blur_detect.py — Standalone blur-detection scoring module.

In plain English: given a photo file, return a number between 0 and 1
that says how blurry it is. Six classic CV "is this sharp?" measurements
are computed, each is normalised to 0–1, and the weighted average becomes
the final score. The six measurements complement each other — a single
metric can be fooled by certain image types (e.g. low-texture skies look
"blurry" to Laplacian but not to FFT), so combining them is more robust.

Runs on macOS (Apple Silicon / CPU) and Linux (NVIDIA CUDA).
Exposes score_image(path) → float and score_image_detailed(path) → dict.
Usable as a CLI tool or imported as a library.

Dependencies:
    Required: opencv-python-headless, numpy
    Optional: torch (GPU-accelerated FFT on CUDA / MPS)
"""

import argparse
import csv
import gc
import json
import logging
import math
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

# HEIC/HEIF support via Pillow + pillow-heif (optional)
_HAS_PILLOW_HEIF = False
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HAS_PILLOW_HEIF = True
except ImportError:
    pass

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ---------------------------------------------------------------------------
# Device detection — resolved once at module import
# ---------------------------------------------------------------------------

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

# Public alias
HAS_TORCH: bool = _HAS_TORCH

_HAS_CV2_CUDA = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        _HAS_CV2_CUDA = True
except AttributeError:
    pass


def _detect_device() -> str:
    if _HAS_TORCH:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


DEVICE: str = _detect_device()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

if DEVICE != "cpu" and not _HAS_CV2_CUDA:
    log.warning(
        "torch detected %s but cv2.cuda is not available; "
        "spatial filters will use CPU OpenCV.",
        DEVICE,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif",
}

WEIGHTS = {
    "laplacian": 0.30,
    "tenengrad": 0.20,
    "fft":       0.15,
    "cpbd":      0.15,
    "brenner":   0.10,
    "local_std": 0.10,
}

BLUR_THRESHOLD = float(os.environ.get("MEMORIA_BLUR_THRESHOLD", "0.75"))

# FFT resize cap (longest edge)
_FFT_MAX_PX = 2048

# Batch breathing interval
_BATCH_BREATHE_EVERY = 50

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _resize_for_fft(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= _FFT_MAX_PX:
        return gray
    scale = _FFT_MAX_PX / longest
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _clear_gpu_cache() -> None:
    if not _HAS_TORCH:
        return
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


def _classify(score: float, threshold: float = BLUR_THRESHOLD) -> str:
    """Classify a blur score. Higher score = blurrier."""
    if score > threshold:
        return "blurry"
    if score > 0.50:
        return "soft"
    return "sharp"


# Public alias
classify = _classify


# ---------------------------------------------------------------------------
# Model 1: Laplacian Variance  (weight 0.30)
# ---------------------------------------------------------------------------

def _score_laplacian(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    return min(var / 1000.0, 1.0)


# ---------------------------------------------------------------------------
# Model 2: Tenengrad / Sobel gradient magnitude  (weight 0.20)
# ---------------------------------------------------------------------------

def _score_tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    tenengrad = float(np.mean(gx ** 2 + gy ** 2))
    return min(tenengrad / 5000.0, 1.0)


# ---------------------------------------------------------------------------
# Model 3: Brenner gradient  (weight 0.10)
# ---------------------------------------------------------------------------

def _score_brenner(gray: np.ndarray) -> float:
    gray_f = gray.astype(np.float64)
    diff = gray_f[:, 2:] - gray_f[:, :-2]
    brenner = float(np.mean(diff ** 2))
    return min(brenner / 2000.0, 1.0)


# ---------------------------------------------------------------------------
# Model 4: FFT high-frequency energy ratio  (weight 0.15)
# ---------------------------------------------------------------------------

def _score_fft(gray: np.ndarray, device: str | None = None) -> float:
    dev = device or DEVICE
    small = _resize_for_fft(gray).astype(np.float32)
    h, w = small.shape
    diag = math.sqrt(h * h + w * w)
    radius = diag * 0.10

    if dev in ("cuda", "mps") and _HAS_TORCH:
        t = torch.from_numpy(small).to(dev)
        fft = torch.fft.fft2(t)
        magnitude = torch.abs(fft)
        mag_np = magnitude.cpu().numpy()
        del t, fft, magnitude
        _clear_gpu_cache()
    else:
        fft = np.fft.fft2(small)
        mag_np = np.abs(fft)

    # Shift zero-frequency to centre
    mag_shifted = np.fft.fftshift(mag_np)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    total_energy = float(np.sum(mag_shifted ** 2))
    if total_energy == 0:
        return 0.0
    high_energy = float(np.sum(mag_shifted[dist > radius] ** 2))
    ratio = high_energy / total_energy
    # Natural images have most energy at low frequencies (1/f² spectrum),
    # so even sharp photos have a small ratio. Normalise to practical range.
    return min(ratio / 0.10, 1.0)


# ---------------------------------------------------------------------------
# Model 5: Local standard deviation  (weight 0.10)
# ---------------------------------------------------------------------------

def _score_local_std(gray: np.ndarray) -> float:
    gray_f = gray.astype(np.float64)
    k = (15, 15)
    mean = cv2.blur(gray_f, k)
    mean_sq = cv2.blur(gray_f ** 2, k)
    var_map = mean_sq - mean ** 2
    # Clamp tiny negative values from floating point error
    var_map = np.maximum(var_map, 0.0)
    std_map = np.sqrt(var_map)
    mean_local_std = float(np.mean(std_map))
    return min(mean_local_std / 30.0, 1.0)


# ---------------------------------------------------------------------------
# Model 6: CPBD — Cumulative Probability of Blur Detection  (weight 0.15)
#
# Inline implementation (~40 lines) to avoid an external dependency.
# Based on Marziliano et al. (2002) edge-width measure + Ferzli & Karam
# (2009) CPBD formulation.
# ---------------------------------------------------------------------------

def _score_cpbd(gray: np.ndarray) -> float:
    # Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    if np.count_nonzero(edges) == 0:
        return 0.0

    # Gradient magnitude (Sobel)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    # Measure edge widths along horizontal gradient direction
    edge_widths: list[int] = []
    rows, cols = edges.shape
    for r in range(rows):
        c = 0
        while c < cols:
            if edges[r, c] == 0:
                c += 1
                continue
            # Walk left to find start of edge transition
            left = c
            while left > 0 and mag[r, left - 1] > mag[r, left] * 0.5:
                left -= 1
                if c - left > 20:  # cap to avoid runaway
                    break
            # Walk right to find end of edge transition
            right = c
            while right < cols - 1 and mag[r, right + 1] > mag[r, right] * 0.5:
                right += 1
                if right - c > 20:
                    break
            width = right - left + 1
            edge_widths.append(width)
            c = right + 1

    if not edge_widths:
        return 0.0

    widths = np.array(edge_widths, dtype=np.float64)

    # "Just noticeable blur" threshold — edges narrower than this are perceived sharp
    jnb = 5.0

    # CPBD: fraction of edges whose width < JNB
    cpbd_val = float(np.mean(widths < jnb))
    return cpbd_val


# ---------------------------------------------------------------------------
# Ensemble scoring
# ---------------------------------------------------------------------------

_RAW_EXTS_BLUR = {
    ".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng",
    ".rw2", ".orf", ".pef", ".raf", ".srw",
}


def _read_image(path: str) -> np.ndarray:
    """Read an image with HEIC/HEIF (Pillow) and camera-RAW (rawpy) fallbacks."""
    ext = os.path.splitext(path)[1].lower()

    # Camera RAW — try rawpy first; cv2.imread can't decode CR2/NEF/etc.
    if ext in _RAW_EXTS_BLUR:
        try:
            import rawpy
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=False,
                    output_bps=8,
                )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except ImportError:
            log.warning("rawpy not installed; cannot read RAW file %s", path)
            raise FileNotFoundError(f"Cannot read RAW image (rawpy missing): {path}")
        except Exception as exc:
            log.warning("rawpy failed on %s: %s", path, exc)
            # fall through to other paths just in case

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # cv2.imread failed — try Pillow (handles HEIC/HEIF if pillow-heif is installed)
    if _HAS_PIL:
        try:
            pil_img = PILImage.open(path)
            pil_img = pil_img.convert("RGB")
            arr = np.array(pil_img)
            # RGB → BGR for OpenCV consistency
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    raise FileNotFoundError(f"Cannot read image: {path}")


def score_image_detailed(path: str, *, device: str | None = None) -> dict:
    """Score a single image, returning per-model breakdown."""
    img = _read_image(path)

    gray = _to_gray(img)

    models = {
        "laplacian": _score_laplacian(gray),
        "tenengrad": _score_tenengrad(gray),
        "brenner":   _score_brenner(gray),
        "fft":       _score_fft(gray, device=device),
        "local_std": _score_local_std(gray),
        "cpbd":      _score_cpbd(gray),
    }

    # Invert: models measure sharpness (high = sharp) but we want
    # blur_score where high = blurry, so: blur = 1 - sharpness.
    sharpness = sum(models[m] * WEIGHTS[m] for m in models)
    blur_score = 1.0 - sharpness
    label = _classify(blur_score)

    # Invert per-model scores too so they all read as blur amount
    inverted_models = {k: round(1.0 - v, 6) for k, v in models.items()}

    # Free image memory eagerly
    del img, gray

    return {
        "blur_score": round(blur_score, 6),
        "label": label,
        "models": inverted_models,
    }


def score_image(path: str, *, device: str | None = None) -> float:
    """Score a single image → float ∈ [0, 1]."""
    return score_image_detailed(path, device=device)["blur_score"]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _collect_images(target: str, recursive: bool) -> list[str]:
    p = Path(target)
    if p.is_file():
        return [str(p)]
    if not p.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {target}")
    glob_fn = p.rglob if recursive else p.glob
    return sorted(
        str(f) for f in glob_fn("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def _print_csv_header(writer: csv.DictWriter) -> None:
    writer.writeheader()


def _run_cli(args: argparse.Namespace) -> None:
    if args.device_info:
        print(f"Device: {DEVICE}")
        if _HAS_TORCH:
            print(f"  torch version: {torch.__version__}")
            if DEVICE == "cuda":
                print(f"  CUDA device:   {torch.cuda.get_device_name(0)}")
        print(f"  cv2.cuda:      {'yes' if _HAS_CV2_CUDA else 'no'}")
        return

    images = _collect_images(args.target, args.recursive)
    if not images:
        log.error("No images found at %s", args.target)
        sys.exit(1)

    threshold = args.threshold

    # CSV mode setup
    csv_writer = None
    if args.csv:
        fieldnames = ["file", "blur_score", "label", "laplacian", "tenengrad",
                       "brenner", "fft", "local_std", "cpbd"]
        csv_writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        _print_csv_header(csv_writer)

    for i, img_path in enumerate(images, 1):
        try:
            result = score_image_detailed(img_path)
        except Exception as exc:
            log.error("Failed on %s: %s", img_path, exc)
            continue

        if args.json:
            out = {"file": img_path, **result}
            if threshold != BLUR_THRESHOLD:
                out["label"] = _classify(result["blur_score"], threshold)
            print(json.dumps(out))

        elif args.csv:
            row = {
                "file": img_path,
                "blur_score": result["blur_score"],
                "label": _classify(result["blur_score"], threshold),
                **result["models"],
            }
            csv_writer.writerow(row)

        elif args.verbose:
            parts = "  ".join(f"{k}: {v:.4f}" for k, v in result["models"].items())
            label = _classify(result["blur_score"], threshold)
            print(f"{img_path}  score={result['blur_score']:.4f}  [{label}]  {parts}")

        else:
            label = _classify(result["blur_score"], threshold)
            print(f"{img_path}  {result['blur_score']:.4f}  [{label}]")

        # Batch breathing
        if i % _BATCH_BREATHE_EVERY == 0:
            gc.collect()
            _clear_gpu_cache()

    # Final cleanup
    gc.collect()
    _clear_gpu_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blur detection — score images using a 6-model weighted ensemble.",
    )
    parser.add_argument(
        "target", nargs="?", default=None,
        help="Image file or directory to score.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument("--json", action="store_true", help="Output JSON per image.")
    parser.add_argument("--csv", action="store_true", help="Output CSV (for calibration).")
    parser.add_argument("--verbose", action="store_true", help="Print per-model score breakdown.")
    parser.add_argument(
        "--threshold", type=float, default=BLUR_THRESHOLD,
        help=f"Blurry/soft boundary (default: {BLUR_THRESHOLD}).",
    )
    parser.add_argument("--device-info", action="store_true", help="Print detected device and exit.")

    args = parser.parse_args()

    if not args.device_info and args.target is None:
        parser.error("target is required (or use --device-info)")

    _run_cli(args)


if __name__ == "__main__":
    main()
