"""
Face detection & embedding extraction pipeline.

In plain English: every photo flows through three independent face-finders.
A box only counts as a "real" face if at least two of them agree the box
is in roughly the same place. That consensus step is what keeps false
positives (hands, posters of faces, busy backgrounds) out of the library.
Once a face is confirmed we pull its 512-dimensional ArcFace embedding —
a numeric fingerprint that lets us cluster the same person across photos.

The three detectors and what they're good at:
  • MTCNN (`facenet-pytorch`) — small + accurate, weak on low-light.
  • YuNet (OpenCV DNN)         — fastest, great recall.
  • InsightFace (ONNX)         — heaviest, also produces the embeddings.

Multi-model ensemble with consensus voting to eliminate false positives.
Runs 3 models — MTCNN + YuNet + InsightFace — and only accepts detections
confirmed by ≥2 of them.

Hardware compatibility:
  - MTCNN       : CPU, CUDA.  MPS → forced CPU (adaptive_avg_pool2d unsupported).
  - YuNet       : CPU, CUDA.  Always CPU on macOS (OpenCV DNN backend).
  - InsightFace : CPU, CUDA (CUDAExecutionProvider).  MPS → CPU fallback.
                  ★ Best results on NVIDIA CUDA — run on your GPU for full speed.

Can be used as a library or run as a CLI:

    python -m backend.face_pipeline --image photo.jpg
    python -m backend.face_pipeline --reprocess --batch-size 10
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import sys
import tempfile
import threading
import urllib.request
import ssl
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ml.base import ModelSpec
    from ml.face import (
        FACE_VECTOR_DIMENSION,
        FaceAnalysisEngine,
        reconcile_face_onnx_execution,
    )
    from ml.providers import ExecutionPlan, resolve_execution
except ImportError:
    _backend_dir = Path(__file__).resolve().parent.parent
    if str(_backend_dir) not in sys.path:
        sys.path.insert(0, str(_backend_dir))
    from ml.base import ModelSpec
    from ml.face import (
        FACE_VECTOR_DIMENSION,
        FaceAnalysisEngine,
        reconcile_face_onnx_execution,
    )
    from ml.providers import ExecutionPlan, resolve_execution

try:
    from runtime_paths import TRT_ENGINE_CACHE_DIR, YUNET_MODEL_PATH
except ImportError:
    _backend_dir = Path(__file__).resolve().parent.parent
    if str(_backend_dir) not in sys.path:
        sys.path.insert(0, str(_backend_dir))
    from runtime_paths import TRT_ENGINE_CACHE_DIR, YUNET_MODEL_PATH

logger = logging.getLogger("memoria.ml.face_pipeline")

# Optional metrics — only enabled when the metrics module is importable. When
# face_pipeline is run as a CLI outside the backend package the import may
# fail; in that case we substitute a no-op shim so timings simply don't get
# recorded (CLI runs aren't metered anyway).
try:
    import metrics as _metrics  # type: ignore
except ImportError:
    try:
        import sys as _sys
        _backend = Path(__file__).resolve().parent.parent
        if str(_backend) not in _sys.path:
            _sys.path.insert(0, str(_backend))
        import metrics as _metrics  # type: ignore
    except Exception:  # pragma: no cover — CLI fallback
        from contextlib import contextmanager as _ctx

        class _NoopMetrics:
            @staticmethod
            @_ctx
            def timer(*_a, **_k):
                yield {}

            @staticmethod
            def count(*_a, **_k):
                pass

            @staticmethod
            def record(*_a, **_k):
                pass

        _metrics = _NoopMetrics()

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_hardware() -> str:
    """Return 'cuda', 'mps', or 'cpu' — the best compute device available."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _clear_cache(hw: str) -> None:
    """Run Python's garbage collector and ask the GPU driver to release
    cached blocks. The pipeline calls this between heavy steps so VRAM
    headroom stays predictable. CPU path is a no-op past the gc.collect()."""
    import torch

    gc.collect()
    if hw == "cuda":
        torch.cuda.empty_cache()
    elif hw == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# SSL fix for macOS
# ---------------------------------------------------------------------------

def _fix_ssl() -> None:
    """Ensure SSL certificates work on macOS Python installs."""
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass


_fix_ssl()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DOWNLOAD_ATTEMPTS = 3  # Skip model entirely after 3 failed download attempts

# Minimum confidence per model to even consider a detection
MODEL_CONFIDENCE_THRESHOLDS = {
    "mtcnn": 0.93,          # MTCNN is false-positive-prone on skin-tone regions; set VERY high
    "yunet": 0.80,          # YuNet has low FP rate; moderate threshold is fine
    "insightface": 0.50,    # InsightFace det_score is already calibrated; trust it
}

# Minimum face dimensions in pixels
MIN_FACE_SIZE = 30

# Aspect ratio bounds for face bounding boxes (w/h)
# A human face is typically ~0.65–1.0 w/h (taller than wide).
# Hands, arms, and other body parts tend to be wider or narrower.
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 1.5

# IoU threshold for matching detections across models
IOU_MATCH_THRESHOLD = 0.3

# Hard minimum — NEVER accept a detection from a single model alone,
# even if only one model loaded. This is the key FP prevention measure.
MIN_CONSENSUS = 2

# Model weights for weighted ensemble scoring (embedding similarity)
MODEL_WEIGHTS: dict[str, float] = {
    "insightface": 0.55,
    "mtcnn": 0.25,
    "yunet": 0.20,
}

SAME_PERSON_THRESHOLD = 0.55

# Where to cache model files
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_BUNDLED_YUNET_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)
_YUNET_MIN_BYTES = 10_000


def _yunet_model_is_usable(path: Path) -> bool:
    """Validate a cached YuNet graph instead of trusting file size alone."""
    try:
        if path.stat().st_size <= _YUNET_MIN_BYTES:
            return False
        probe = cv2.FaceDetectorYN.create(
            str(path),
            "",
            (320, 320),
            score_threshold=MODEL_CONFIDENCE_THRESHOLDS["yunet"],
            nms_threshold=0.3,
            top_k=50,
        )
        return probe is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Download helper with retry
# ---------------------------------------------------------------------------

def _download_with_retry(
    url: str,
    dest: Path,
    max_attempts: int = MAX_DOWNLOAD_ATTEMPTS,
    min_bytes: int = 1,
) -> bool:
    """Atomically download a file with bounded retries."""
    import time as _time

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(
            "  Cannot create model directory %s: %s",
            dest.parent,
            exc,
        )
        return False
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        # ssl.create_default_context() already uses the operating system trust
        # store. Never turn certificate verification off merely because the
        # optional certifi package is absent.
        pass

    for attempt in range(1, max_attempts + 1):
        temporary_path: Path | None = None
        try:
            logger.info("  Download attempt %d/%d: %s", attempt, max_attempts, url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                expected_header = resp.headers.get("Content-Length")
                with tempfile.NamedTemporaryFile(
                    prefix=f".{dest.name}.",
                    suffix=".part",
                    dir=dest.parent,
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    shutil.copyfileobj(resp, temporary)
                    temporary.flush()
                    os.fsync(temporary.fileno())

            downloaded_bytes = temporary_path.stat().st_size
            expected_bytes = int(expected_header) if expected_header else None
            if downloaded_bytes < min_bytes:
                raise RuntimeError(
                    f"download was unexpectedly small ({downloaded_bytes} bytes)"
                )
            if expected_bytes is not None and downloaded_bytes != expected_bytes:
                raise RuntimeError(
                    "download was incomplete "
                    f"({downloaded_bytes}/{expected_bytes} bytes)"
                )
            os.replace(temporary_path, dest)
            temporary_path = None
            logger.info("  Downloaded: %s (%d bytes)", dest.name, dest.stat().st_size)
            return True
        except Exception as e:
            logger.warning("  Attempt %d failed: %s", attempt, e)
            if attempt < max_attempts:
                _time.sleep(2 * attempt)  # backoff
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
    logger.error("  Giving up on download after %d attempts: %s", max_attempts, url)
    return False


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _is_valid_face_bbox(x: int, y: int, w: int, h: int) -> bool:
    """Check if a bounding box has plausible face geometry."""
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False
    aspect = w / max(h, 1)
    if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
        return False
    return True


def _iou(box_a: dict, box_b: dict) -> float:
    """Compute intersection-over-union between two bbox dicts."""
    ax, ay, aw, ah = box_a["x"], box_a["y"], box_a["w"], box_a["h"]
    bx, by, bw, bh = box_b["x"], box_b["y"], box_b["w"], box_b["h"]

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter

    return inter / max(union, 1e-6)


# ---------------------------------------------------------------------------
# Individual detector wrappers
# ---------------------------------------------------------------------------

class _MTCNNDetector:
    """MTCNN face detector from facenet-pytorch.

    + Good at detecting small/angled faces, multi-scale cascade
    - High false positive rate on skin-tone regions (hands, neck, shoulders)
    → Mitigated by very tight cascade thresholds + high min_face_size

    Works on: CPU, CUDA.  MPS → forced CPU (adaptive_avg_pool2d unsupported).
    """

    def __init__(self, hw: str):
        import torch
        from facenet_pytorch import MTCNN

        # MPS doesn't support adaptive_avg_pool2d for non-divisible sizes
        device = torch.device(
            "cuda" if hw == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self._mtcnn = MTCNN(
            keep_all=True,
            device=device,
            post_process=False,
            min_face_size=60,                 # ignore tiny blobs
            thresholds=[0.85, 0.90, 0.95],    # very strict cascade (P→R→O nets)
        )
        self.name = "mtcnn"

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        from PIL import Image

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        boxes, probs = self._mtcnn.detect(pil_img)
        if boxes is None:
            return []

        h, w = img_bgr.shape[:2]
        results = []
        threshold = MODEL_CONFIDENCE_THRESHOLDS["mtcnn"]

        for box, prob in zip(boxes, probs):
            if prob is None or float(prob) < threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if not _is_valid_face_bbox(x1, y1, bw, bh):
                continue
            results.append({
                "bbox": {"x": x1, "y": y1, "w": bw, "h": bh},
                "confidence": float(prob),
                "detection_model": "mtcnn",
            })
        return results

    def close(self) -> None:
        """Release the Torch model so accelerator memory can be reclaimed."""
        model = getattr(self, "_mtcnn", None)
        if model is not None:
            del self._mtcnn


class _YuNetDetector:
    """OpenCV YuNet face detector (DNN-based, built into OpenCV 4.8+).

    + Very fast, low false positive rate, 5-point landmark detection
    + Small ONNX model (~330 KB)
    - Installs the bundled model atomically; bounded download is a fallback

    Works on: CPU, CUDA (via OpenCV DNN).  Always CPU on macOS.
    """

    def __init__(self, hw: str):
        self._model_path = str(YUNET_MODEL_PATH)
        self._ensure_model()
        self._detector = None  # lazy per-frame due to input size
        self.name = "yunet"

    def _ensure_model(self) -> None:
        if _yunet_model_is_usable(YUNET_MODEL_PATH):
            return
        if YUNET_MODEL_PATH.exists():
            logger.warning("Cached YuNet model is incomplete or invalid: %s", YUNET_MODEL_PATH)
        if (
            _BUNDLED_YUNET_PATH.exists()
            and _yunet_model_is_usable(_BUNDLED_YUNET_PATH)
        ):
            YUNET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=f".{YUNET_MODEL_PATH.name}.",
                    suffix=".part",
                    dir=YUNET_MODEL_PATH.parent,
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    with _BUNDLED_YUNET_PATH.open("rb") as source:
                        shutil.copyfileobj(source, temporary)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_path, YUNET_MODEL_PATH)
                temporary_path = None
                if _yunet_model_is_usable(YUNET_MODEL_PATH):
                    logger.info("Installed bundled YuNet model at %s", YUNET_MODEL_PATH)
                    return
                logger.warning("Bundled YuNet model failed validation after installation")
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        logger.info("Downloading YuNet model...")
        if not _download_with_retry(
            YUNET_URL,
            YUNET_MODEL_PATH,
            min_bytes=_YUNET_MIN_BYTES,
        ):
            raise RuntimeError(f"Failed to download YuNet model after {MAX_DOWNLOAD_ATTEMPTS} attempts")
        if not _yunet_model_is_usable(YUNET_MODEL_PATH):
            raise RuntimeError("Downloaded YuNet model failed OpenCV validation")

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        h, w = img_bgr.shape[:2]
        detector = cv2.FaceDetectorYN.create(
            self._model_path, "", (w, h),
            score_threshold=MODEL_CONFIDENCE_THRESHOLDS["yunet"],
            nms_threshold=0.3,
            top_k=50,
        )
        _, faces = detector.detect(img_bgr)
        if faces is None:
            return []

        results = []
        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            conf = float(face[14]) if len(face) > 14 else float(face[-1])
            x, y = max(0, x), max(0, y)
            if not _is_valid_face_bbox(x, y, fw, fh):
                continue
            results.append({
                "bbox": {"x": x, "y": y, "w": fw, "h": fh},
                "confidence": conf,
                "detection_model": "yunet",
            })
        return results

    def close(self) -> None:
        """YuNet creates its OpenCV detector per frame; no model is retained."""
        self._detector = None


class _InsightFaceDetector:
    """InsightFace buffalo_l (SCRFD detection + ArcFace R100 embeddings).

    + Best overall accuracy (99.83% LFW); produces 512-d face embeddings
    + Very low false positive rate — SCRFD is trained on WIDER FACE
    + Provides gender/age estimation as bonus
    - Needs onnxruntime; ~1.3 GB VRAM on CUDA; ~280 MB model download

    Works on:
      ★ CUDA  — CUDAExecutionProvider (fastest, recommended for production)
      ● CPU   — CPUExecutionProvider  (works everywhere, slower)
      ▲ MPS   — NO native MPS support. Falls back to CPU automatically.
                For best performance, run on NVIDIA GPU.

    Model download is retried up to 3 times. InsightFace auto-downloads
    buffalo_l (~280MB) on first use.
    """

    def __init__(
        self,
        hw: str,
        provider: str = "auto",
        precision: str = "auto",
        model_name: str = "buffalo_l",
    ):
        import os
        from pathlib import Path
        from insightface.app import FaceAnalysis

        providers: list = []
        try:
            import onnxruntime as ort
            ort_providers = set(ort.get_available_providers())
        except Exception:
            ort_providers = {"CPUExecutionProvider"}
        use_cuda = (
            hw == "cuda"
            and provider in {"auto", "cuda", "onnx-cuda", "tensorrt"}
            and "CUDAExecutionProvider" in ort_providers
        )
        use_trt = False
        reported_precision = "fp32"
        if use_cuda:
            # Opt-in TensorRT execution. ~1.5–2× faster on ArcFace embedding
            # vs plain CUDA EP (with fp16 enabled), at the cost of a one-time
            # ~30 s – 2 min per-model engine compile on first run. The
            # compiled engines are cached on disk and reused across restarts.
            #
            # Enable: set MEMORIA_INSIGHTFACE_USE_TRT=1 in the environment
            # before launching the Huey worker.
            legacy_trt = os.environ.get(
                "MEMORIA_INSIGHTFACE_USE_TRT", ""
            ).strip().lower() in {"1", "true", "yes"}
            use_trt = (
                provider == "tensorrt" or (provider == "auto" and legacy_trt)
            ) and "TensorrtExecutionProvider" in ort_providers
            if use_trt:
                trt_cache_dir = TRT_ENGINE_CACHE_DIR
                trt_cache_dir.mkdir(parents=True, exist_ok=True)
                # This pipeline has an explicit TensorRT FP16 path only;
                # there is no active TensorRT bf16 option to report.
                reported_precision = (
                    "fp32" if precision in {"fp32", "float32"} else "fp16"
                )
                providers.append((
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": reported_precision == "fp16",
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(trt_cache_dir),
                    },
                ))
                logger.info(
                    "InsightFace TensorRT execution precision=%s; engine cache=%s",
                    reported_precision,
                    trt_cache_dir,
                )
            providers.append("CUDAExecutionProvider")
        # MPS has no ONNX provider — always fall back to CPU
        providers.append("CPUExecutionProvider")

        if hw == "mps":
            logger.info("  ⚠ InsightFace has NO MPS support — running on CPU.")
            logger.info("    For best speed, run on NVIDIA CUDA GPU.")

        self._app = FaceAnalysis(name=model_name, providers=providers)
        self._app.prepare(ctx_id=0 if use_cuda else -1, det_size=(640, 640))
        recognition_model = getattr(self._app, "models", {}).get("recognition")
        session_models = (
            [recognition_model]
            if recognition_model is not None
            else list(getattr(self._app, "models", {}).values())
        )
        active_providers: list[str] = []
        for model in session_models:
            session = getattr(model, "session", None)
            get_providers = getattr(session, "get_providers", None)
            if not callable(get_providers):
                continue
            try:
                active_providers.extend(get_providers())
            except Exception as exc:
                logger.warning(
                    "Could not inspect InsightFace ONNX Runtime providers: %s",
                    exc,
                )
        self.active_session_providers = tuple(dict.fromkeys(active_providers))
        self.tensorrt_fp16_enabled = bool(use_trt and reported_precision == "fp16")
        if "TensorrtExecutionProvider" in self.active_session_providers:
            self.actual_provider = "tensorrt"
            self.actual_precision = (
                "fp16" if self.tensorrt_fp16_enabled else "fp32"
            )
        elif "CUDAExecutionProvider" in self.active_session_providers:
            self.actual_provider, self.actual_precision = "onnx-cuda", "fp32"
        elif "CPUExecutionProvider" in self.active_session_providers:
            self.actual_provider, self.actual_precision = "cpu", "fp32"
        else:
            self.actual_provider, self.actual_precision = provider, precision
            logger.warning(
                "InsightFace recognition session did not expose its provider "
                "chain; retaining planned provider=%s precision=%s",
                provider,
                precision,
            )
        if use_trt and self.actual_provider != "tensorrt":
            logger.warning(
                "TensorRT requested but InsightFace loaded providers %s; "
                "reporting provider=%s precision=%s",
                self.active_session_providers,
                self.actual_provider,
                self.actual_precision,
            )
        self.name = "insightface"

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        faces = self._app.get(img_bgr)
        results = []
        threshold = MODEL_CONFIDENCE_THRESHOLDS["insightface"]

        for f in faces:
            conf = float(f.det_score)
            if conf < threshold:
                continue
            x1, y1, x2, y2 = f.bbox.astype(int)
            bw, bh = int(x2 - x1), int(y2 - y1)
            if not _is_valid_face_bbox(int(x1), int(y1), bw, bh):
                continue

            emb = f.embedding / (np.linalg.norm(f.embedding) + 1e-10)
            results.append({
                "bbox": {"x": int(x1), "y": int(y1), "w": bw, "h": bh},
                "confidence": conf,
                "detection_model": "insightface",
                "embedding": emb.tolist(),
            })
        return results

    def close(self) -> None:
        """Drop ONNX Runtime sessions held by InsightFace."""
        app = getattr(self, "_app", None)
        if app is not None:
            del self._app


# ---------------------------------------------------------------------------
# Face embedding extractor (separate from detectors)
# ---------------------------------------------------------------------------

class _FaceNetEmbedder:
    """FaceNet InceptionResnetV1 for 512-d face embeddings.

    Used to add embeddings to detections from models that don't produce them
    (MTCNN, YuNet, RetinaFace).
    """

    def __init__(self, hw: str = "cpu"):
        import torch
        from facenet_pytorch import InceptionResnetV1

        self._device = (
            "cuda" if hw == "cuda" and torch.cuda.is_available() else "cpu"
        )
        try:
            self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self._device)
        except Exception as e:
            logger.warning("Could not load pretrained FaceNet weights: %s", e)
            self._model = None

    def embed(self, img_rgb: np.ndarray, bbox: dict) -> list[float] | None:
        """Extract 512-d embedding from a face crop."""
        if self._model is None:
            return None

        import torch
        from PIL import Image
        from torchvision import transforms

        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        ih, iw = img_rgb.shape[:2]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)

        crop = img_rgb[y:y2, x:x2]
        if crop.size == 0:
            return None

        preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        face_pil = Image.fromarray(crop)
        tensor = preprocess(face_pil).unsqueeze(0).to(self._device)

        with torch.no_grad():
            emb = self._model(tensor).cpu().numpy().flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb.tolist()

    def close(self) -> None:
        model = getattr(self, "_model", None)
        if model is not None:
            del self._model
        self._model = None


# ---------------------------------------------------------------------------
# Consensus logic
# ---------------------------------------------------------------------------

def _merge_detections(
    all_detections: list[list[dict]],
    min_consensus: int = MIN_CONSENSUS,
) -> list[dict]:
    """Merge detections from multiple models using IoU-based consensus.

    Only keeps detections confirmed by at least `min_consensus` models.
    Returns merged detections with averaged bboxes and max confidence.

    In plain English: each model produced its own list of "I see a face
    here" boxes. Two boxes from *different* models that overlap by at
    least IOU_MATCH_THRESHOLD (default 0.3) are treated as the same face.
    A face passes only if at least `min_consensus` distinct models voted
    for it. The output box is the average of the agreeing boxes, with
    the highest confidence among them.

    IoU = intersection-over-union: the area where two boxes overlap
    divided by the area they cover combined. 1.0 = identical, 0.0 = no
    overlap at all.
    """
    if not all_detections:
        return []

    # Flatten all detections with source tracking
    all_dets = []
    for dets in all_detections:
        all_dets.extend(dets)

    if not all_dets:
        return []

    # Build groups: for each detection, find overlapping detections from other models
    used = [False] * len(all_dets)
    merged = []

    for i, det_a in enumerate(all_dets):
        if used[i]:
            continue

        group = [det_a]
        group_models = {det_a["detection_model"]}
        used[i] = True

        for j, det_b in enumerate(all_dets):
            if used[j]:
                continue
            if det_b["detection_model"] in group_models:
                continue  # only count one detection per model
            if _iou(det_a["bbox"], det_b["bbox"]) >= IOU_MATCH_THRESHOLD:
                group.append(det_b)
                group_models.add(det_b["detection_model"])
                used[j] = True

        if len(group_models) >= min_consensus:
            # Average the bboxes, take max confidence
            avg_x = int(np.mean([d["bbox"]["x"] for d in group]))
            avg_y = int(np.mean([d["bbox"]["y"] for d in group]))
            avg_w = int(np.mean([d["bbox"]["w"] for d in group]))
            avg_h = int(np.mean([d["bbox"]["h"] for d in group]))
            max_conf = max(d["confidence"] for d in group)
            models_used = sorted(group_models)

            # If any detection has an embedding, keep it
            embedding = None
            for d in group:
                if "embedding" in d:
                    embedding = d["embedding"]
                    break

            result = {
                "bbox": {"x": avg_x, "y": avg_y, "w": avg_w, "h": avg_h},
                "confidence": max_conf,
                "detection_model": "+".join(models_used),
                "consensus": len(group_models),
                "models_agreed": models_used,
            }
            if embedding:
                result["embedding"] = embedding
            merged.append(result)

    return merged


# ---------------------------------------------------------------------------
# FaceDetector — main public interface
# ---------------------------------------------------------------------------

class FaceDetector(FaceAnalysisEngine):
    """Multi-model ensemble face detector.

    Runs multiple detection models and uses consensus voting to eliminate
    false positives. A detection must be confirmed by at least 2 models
    to be accepted.

    3 models (hard limit):
      - MTCNN         : CPU / CUDA  (fast cascade, strict thresholds)
      - YuNet         : CPU         (OpenCV DNN, low FP rate)
      - InsightFace   : CPU / CUDA  (best accuracy + embeddings)

    MPS note: None of these models have native MPS acceleration.
    MTCNN and InsightFace fall back to CPU on Apple Silicon.
    For best performance, run on NVIDIA CUDA.

    Usage:
        detector = FaceDetector()
        faces = detector.detect(image_bgr)
        faces, annotated = detector.detect_for_display(image_bgr)
    """

    def __init__(
        self,
        hw_mode: str | None = None,
        models: list[str] | None = None,
        min_consensus: int = MIN_CONSENSUS,
        use_embeddings: bool = True,
        *,
        embedder: str = "insightface",
        provider: str = "auto",
        precision: str = "auto",
        engine_key: str = "ensemble-v1",
        pipeline_key: str | None = None,
        embedding_model_key: str | None = None,
        insightface_model: str = "buffalo_l",
        batch_concurrency: int = 2,
        model_spec: ModelSpec | None = None,
        execution: Any | None = None,
    ):
        self.min_consensus = max(int(min_consensus), MIN_CONSENSUS)
        self.use_embeddings = use_embeddings
        self.embedder_name = embedder.strip().lower() if use_embeddings else "none"
        if self.embedder_name in {"arcface", "insightface-arcface"}:
            self.embedder_name = "insightface"
        if self.embedder_name in {"facenet-vggface2", "vggface2"}:
            self.embedder_name = "facenet"
        if self.embedder_name not in {"insightface", "facenet", "none"}:
            raise ValueError(f"Unknown face embedder: {embedder!r}")
        self.engine_key = engine_key
        self.insightface_model = insightface_model
        self.batch_concurrency = max(1, int(batch_concurrency))
        self._detectors: list[Any] = []
        self._embedder: _FaceNetEmbedder | None = None
        self._loaded = False
        self._load_lock = threading.RLock()
        self._detector_locks: dict[str, threading.RLock] = {}
        self._embedder_lock = threading.RLock()

        # Determine which models to load
        if models:
            self._requested_models = list(
                dict.fromkeys(m.strip().lower() for m in models)
            )
        else:
            self._requested_models = self._default_models()
        unknown = sorted(
            set(self._requested_models) - {"mtcnn", "yunet", "insightface"}
        )
        if unknown:
            raise ValueError(f"Unknown face detector(s): {', '.join(unknown)}")
        if len(self._requested_models) < self.min_consensus:
            raise ValueError(
                "Face detector quorum cannot be met: "
                f"{len(self._requested_models)} configured, "
                f"{self.min_consensus} required"
            )

        detector_key = "+".join(self._requested_models)
        self.detection_pipeline_key = (
            pipeline_key
            or (model_spec.model_key if model_spec is not None else None)
            or f"face/{engine_key}:{detector_key}"
        )
        if embedding_model_key:
            self.embedding_model_key = embedding_model_key
        elif self.embedder_name == "insightface":
            model_slug = insightface_model.strip().lower().replace("_", "-")
            self.embedding_model_key = f"face/insightface-{model_slug}-arcface"
        elif self.embedder_name == "facenet":
            self.embedding_model_key = "face/facenet-vggface2"
        else:
            self.embedding_model_key = None
        self.embedding_dimension = (
            FACE_VECTOR_DIMENSION if self.embedding_model_key else None
        )

        if model_spec is None:
            if self.embedder_name == "insightface":
                provider_preferences = ("tensorrt", "onnx-cuda", "cpu")
            elif self.embedder_name == "facenet":
                provider_preferences = ("cuda", "cpu")
            else:
                provider_preferences = ("onnx-cuda", "cuda", "cpu")
            model_spec = ModelSpec(
                model_key=self.detection_pipeline_key,
                task="face",
                driver="face-ensemble",
                model_id=engine_key,
                input_resolution=(640, 640),
                embedding_dim=self.embedding_dimension,
                default_precision="auto",
                provider_preferences=provider_preferences,
                metadata={
                    "detectors": list(self._requested_models),
                    "embedder": self.embedder_name,
                    "embedding_model_key": self.embedding_model_key,
                    "detection_pipeline_key": self.detection_pipeline_key,
                    "fixed_storage_dimension": FACE_VECTOR_DIMENSION,
                },
            )
        if model_spec.model_key != self.detection_pipeline_key:
            raise ValueError(
                "Face pipeline provenance key does not match its registered "
                f"model spec: {self.detection_pipeline_key!r} != "
                f"{model_spec.model_key!r}"
            )
        if model_spec.embedding_dim not in {None, FACE_VECTOR_DIMENSION}:
            raise ValueError(
                "Face storage is fixed at 512 dimensions; "
                f"model declares {model_spec.embedding_dim}"
            )
        if execution is None:
            execution = resolve_execution(
                model_spec,
                requested_provider=None if provider == "auto" else provider,
                requested_precision=None if precision == "auto" else precision,
            )
        if execution.actual_provider == "tensorrt":
            # The provider options below expose only TensorRT FP16. Normalize
            # direct FaceDetector construction as well as the configured
            # facade path so runtime provenance can never claim bf16 here.
            reported_precision = (
                "fp32" if execution.requested_precision == "fp32" else "fp16"
            )
            if execution.actual_precision != reported_precision:
                execution = ExecutionPlan(
                    requested_provider=execution.requested_provider,
                    actual_provider=execution.actual_provider,
                    device=execution.device,
                    requested_precision=execution.requested_precision,
                    actual_precision=reported_precision,
                    available_providers=execution.available_providers,
                    requested_providers=execution.requested_providers,
                    fallback_reason=execution.fallback_reason,
                )
        super().__init__(model_spec, execution)
        self.hw = hw_mode or execution.device
        self.provider = execution.actual_provider
        self.precision = execution.actual_precision

        # For API compatibility
        self.model_name = "ensemble"
        self.active_models: list[str] = []

    @property
    def model_spec(self) -> ModelSpec:
        """Return the registry descriptor used to construct this engine."""
        return self.spec

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _default_models(self) -> list[str]:
        """Return the 3 models to load (hard limit: 3 max)."""
        # All 3 models run on every platform.
        # On MPS/CPU: MTCNN→CPU, YuNet→CPU, InsightFace→CPUExecutionProvider.
        # On CUDA:    MTCNN→CUDA, YuNet→CPU, InsightFace→CUDAExecutionProvider.
        return ["mtcnn", "yunet", "insightface"]

    def _lazy_load(self) -> None:
        with self._load_lock:
            if self._closed:
                raise RuntimeError(f"face engine {self.spec.model_key!r} is closed")
            if self._loaded:
                return
            self._lazy_load_unlocked()

    def _lazy_load_unlocked(self) -> None:
        if self._loaded:
            return

        logger.info("Loading model(s): %s on %s", self._requested_models, self.hw)

        for model_name in self._requested_models:
            try:
                if model_name == "mtcnn":
                    det = _MTCNNDetector(self.hw)
                elif model_name == "yunet":
                    det = _YuNetDetector(self.hw)
                elif model_name == "insightface":
                    det = _InsightFaceDetector(
                        self.hw,
                        provider=self.provider,
                        precision=self.precision,
                        model_name=self.insightface_model,
                    )
                    if self.embedder_name == "insightface":
                        reconciled = reconcile_face_onnx_execution(
                            self.execution,
                            det.active_session_providers,
                            tensorrt_fp16_enabled=det.tensorrt_fp16_enabled,
                        )
                        if reconciled != self.execution:
                            logger.warning(
                                "Face runtime reconciled after load: provider "
                                "%s -> %s, precision %s -> %s",
                                self.execution.actual_provider,
                                reconciled.actual_provider,
                                self.execution.actual_precision,
                                reconciled.actual_precision,
                            )
                            self.execution = reconciled
                            self.provider = reconciled.actual_provider
                            self.precision = reconciled.actual_precision
                else:
                    raise ValueError(f"Unknown face detector: {model_name}")
                self._detectors.append(det)
                self._detector_locks[det.name] = threading.RLock()
                self.active_models.append(model_name)
                logger.info("  ✓ %s loaded", model_name)
            except Exception as e:
                logger.warning("  ✗ %s failed to load: %s", model_name, e)

        if len(self._detectors) < self.min_consensus:
            loaded = list(self.active_models)
            self.reset()
            raise RuntimeError(
                "Face detector quorum unavailable: "
                f"loaded {loaded}, require {self.min_consensus} detector(s)"
            )

        if self.embedder_name == "insightface" and "insightface" not in self.active_models:
            self.reset()
            raise RuntimeError(
                "Configured InsightFace ArcFace embedder is unavailable; "
                "refusing to mix a fallback vector space"
            )
        if self.embedder_name == "facenet":
            try:
                self._embedder = _FaceNetEmbedder(self.hw)
                if getattr(self._embedder, "_model", None) is None:
                    raise RuntimeError("pretrained FaceNet weights are unavailable")
                logger.info("  loaded FaceNet VGGFace2 embedder")
            except Exception as exc:
                self.reset()
                raise RuntimeError(f"FaceNet embedder failed to load: {exc}") from exc

        self.model_name = "+".join(self.active_models) if len(self.active_models) > 1 else self.active_models[0]
        self._loaded = True
        logger.info("Ready. Active models: %s", self.active_models)

    def detect(self, image: np.ndarray | str | Path) -> list[dict]:
        """Detect faces in an image (BGR ndarray or file path).

        Returns list of dicts with keys:
            bbox, confidence, detection_model, consensus, models_agreed
            (and optionally 'embedding')
        """
        self._lazy_load()

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        else:
            img = image

        # Run all detectors
        all_detections = []
        successful_detectors = 0
        failed_detectors: list[str] = []
        for det in self._detectors:
            try:
                with _metrics.timer("face.detector", model=det.name):
                    # Different runtime libraries make different thread-safety
                    # guarantees. One lock per shared driver keeps each model
                    # safe while batch threads may overlap different drivers.
                    with self._detector_locks[det.name]:
                        dets = det.detect(img)
                all_detections.append(dets)
                successful_detectors += 1
                _metrics.record("face.detections", len(dets), model=det.name)
            except Exception as e:
                logger.warning("Detector %s failed on frame: %s", det.name, e)
                _metrics.count("face.detector.error", model=det.name)
                failed_detectors.append(det.name)
                all_detections.append([])

        # Apply consensus — NEVER drop below 2 even if only 1 model loaded.
        # A single-model detection is unreliable (hands, shoulders, etc.).
        effective_min = max(self.min_consensus, MIN_CONSENSUS)
        if successful_detectors < effective_min:
            failed = ", ".join(failed_detectors) or "unknown"
            raise RuntimeError(
                "Face detector quorum failed during inference: "
                f"{successful_detectors} succeeded, {effective_min} required "
                f"(failed: {failed})"
            )
        with _metrics.timer("face.merge"):
            faces = _merge_detections(all_detections, min_consensus=effective_min)

        # A configured FaceNet embedder defines a different vector space, so
        # replace rather than mix InsightFace-provided vectors.
        if self.embedder_name == "facenet" and self._embedder and faces:
            with _metrics.timer("face.embed"):
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for face in faces:
                    face.pop("embedding", None)
                    with self._embedder_lock:
                        emb = self._embedder.embed(rgb, face["bbox"])
                    if emb:
                        face["embedding"] = emb
        elif self.embedder_name == "none":
            for face in faces:
                face.pop("embedding", None)

        for face in faces:
            agreed = list(face.get("models_agreed", []))
            face["detector_models"] = agreed
            face["detection_pipeline_key"] = self.detection_pipeline_key
            embedding = face.get("embedding")
            if embedding is None:
                continue
            try:
                validated = self.validate_face_embedding(embedding)
                vector = np.asarray(validated, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if norm <= 0:
                    raise ValueError("embedding has zero norm")
                vector /= norm
                face["embedding"] = vector.tolist()
                face["embedding_model_key"] = self.embedding_model_key
                face["embedding_dimension"] = FACE_VECTOR_DIMENSION
                face["runtime_provider"] = self.execution.actual_provider
                face["runtime_precision"] = self.execution.actual_precision
            except (TypeError, ValueError) as exc:
                logger.warning("Discarding incompatible face embedding: %s", exc)
                face.pop("embedding", None)
                face.pop("embedding_model_key", None)
                face.pop("embedding_dimension", None)
                face.pop("runtime_provider", None)
                face.pop("runtime_precision", None)

        return faces

    def detect_batch(
        self,
        images: list[np.ndarray | str | Path],
        max_workers: int = 2,
    ) -> list[list[dict]]:
        """Run :meth:`detect` on multiple images concurrently (Phase E6.1).

        Uses a small thread pool so multiple GPU dispatches stay in flight
        at once — CUDA kernels are async per stream, so while one thread is
        waiting on its result the next thread's work is already queued. We
        get most of the benefit of true tensor batching without reaching
        into InsightFace / MTCNN internals.

        ``max_workers`` is intentionally small (default 2). With InsightFace,
        MTCNN, and CLIP weights co-resident, additional concurrent detection
        calls can cause accelerator-memory contention. Tune only after
        measuring headroom in the actual deployment.

        Inference failures propagate to the batch caller. This prevents a
        transient detector outage from being persisted as a successful
        zero-face result.
        """
        self._lazy_load()
        if not images:
            return []

        from concurrent.futures import ThreadPoolExecutor

        # The engine-owned setting remains the memory-safety ceiling even when
        # a legacy caller requests higher concurrency.
        workers = max(1, min(max_workers, self.batch_concurrency, len(images)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.detect, images))

    def close(self) -> None:
        """Release loaded models and ask the active runtime to free caches."""
        with self._load_lock:
            for detector in self._detectors:
                try:
                    close = getattr(detector, "close", None)
                    if close is not None:
                        close()
                except Exception as exc:
                    logger.debug("Face detector close failed: %s", exc)
            self._detectors.clear()
            self._detector_locks.clear()
            if self._embedder is not None:
                try:
                    self._embedder.close()
                except Exception as exc:
                    logger.debug("Face embedder close failed: %s", exc)
            self._embedder = None
            self.active_models.clear()
            self._loaded = False
            self.model_name = "ensemble"
            self._closed = True
            try:
                _clear_cache(self.hw)
            except Exception as exc:
                logger.debug("Face runtime cache cleanup failed: %s", exc)

    def reset(self) -> None:
        """Unload now; the next inference lazily reconstructs the engine."""
        self.close()
        self._closed = False

    def detect_for_display(self, image: np.ndarray) -> tuple[list[dict], np.ndarray]:
        """Detect faces and return annotated image with bounding boxes drawn."""
        faces = self.detect(image)
        annotated = image.copy()

        for face in faces:
            bb = face["bbox"]
            x, y, w, h = bb["x"], bb["y"], bb["w"], bb["h"]
            conf = face["confidence"]
            consensus = face.get("consensus", 1)
            models = face.get("models_agreed", [face.get("detection_model", "?")])

            # Color by consensus strength
            if consensus >= 3:
                color = (0, 255, 0)     # green — strong consensus
            elif consensus >= 2:
                color = (0, 200, 255)   # orange — moderate
            else:
                color = (0, 100, 255)   # red — single model (only if min_consensus=1)

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label: confidence + model count
            label = f"{conf:.0%} [{consensus}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 4, y), color, -1)
            cv2.putText(annotated, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return faces, annotated


# ---------------------------------------------------------------------------
# Model selection helper (for plan.md compatibility)
# ---------------------------------------------------------------------------

def select_face_model(hw_mode: str, preference: str = "auto") -> str:
    if preference == "accurate":
        return "insightface"
    if preference == "fast":
        return "yunet"
    if preference == "fallback":
        return "mtcnn"
    return "ensemble"


# ---------------------------------------------------------------------------
# Weighted ensemble similarity (for clustering)
# ---------------------------------------------------------------------------

def weighted_similarity(
    embeddings_a: dict[str, np.ndarray],
    embeddings_b: dict[str, np.ndarray],
) -> float:
    """Weighted cosine similarity across all available models."""
    score, total_w = 0.0, 0.0
    for model, w in MODEL_WEIGHTS.items():
        if model in embeddings_a and model in embeddings_b:
            score += w * float(np.dot(embeddings_a[model], embeddings_b[model]))
            total_w += w
    return score / total_w if total_w > 0 else 0.0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Face detection pipeline (multi-model ensemble)")
    parser.add_argument("--image", type=str, help="Single image path for testing")
    parser.add_argument("--reprocess", action="store_true", help="Re-run detection on all media")
    parser.add_argument("--model", type=str, nargs="+", default=None,
                        help="Model(s) to use: mtcnn yunet insightface")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--min-consensus", type=int, default=2,
                        help="Min models that must agree (default: 2)")
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--sample-pct", type=int, default=5)
    parser.add_argument("--save-annotated", type=str, default=None,
                        help="Save annotated image to this path")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    hw = detect_hardware()
    logger.info("Hardware: %s", hw)

    detector = FaceDetector(
        hw_mode=hw,
        models=args.model,
        min_consensus=args.min_consensus,
    )

    if args.image:
        faces, annotated = detector.detect_for_display(args.image)
        print(f"\nDetected {len(faces)} face(s):")
        for i, f in enumerate(faces):
            bb = f["bbox"]
            print(f"  Face {i+1}:")
            print(f"    BBox: ({bb['x']}, {bb['y']}) {bb['w']}x{bb['h']}")
            print(f"    Confidence: {f['confidence']:.1%}")
            print(f"    Models agreed: {f.get('models_agreed', ['?'])}")
            print(f"    Consensus: {f.get('consensus', 1)}/{len(detector.active_models)}")

        if args.save_annotated:
            cv2.imwrite(args.save_annotated, annotated)
            print(f"\nAnnotated image saved to: {args.save_annotated}")
        return

    if args.reprocess:
        logger.info("Batch reprocess mode (batch_size=%d)", args.batch_size)
        print("Batch reprocessing requires a database connection.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
