"""Public dedup-embedding façade for the SSCD copy-detection model.

In plain English: SSCD is independent from the selected scene/search model. It is
*purpose-built* to recognise near-duplicate photos — same scene with a
crop, a watermark, a different JPEG quality, a recolour, etc. — even when
ordinary pHash misses them. This file owns the singleton SSCD model and
exposes ``encode_image`` / ``encode_images_batch`` to the pipeline.

The actual *duplicate grouping* (clustering nearby SSCD embeddings into
groups of duplicates) happens in ``routers.py::_refresh_duplicate_groups``
using pgvector's HNSW index over the embeddings — this file just produces
the vectors.

Mirrors ``backend/scene_engine.py``: thin lazy singleton wrapping the SSCD
TorchScript model + ImageNet preprocess transform so we initialize once per
process. Used by:
  * ``workers/tasks.py::compute_dedup_embedding`` during the live pipeline
  * ``workers/tasks.py::backfill_dedup_embeddings`` for initial / catch-up
  * ``routers.py::_refresh_duplicate_groups`` consumes the embeddings via SQL
    HNSW lookup, so it does not call into this module directly

SSCD = Self-Supervised Copy Detection (Meta Research). ResNet50 backbone,
512-d L2-normalized output, SOTA on DISC2021 for crops / recolors / watermarks
/ re-encodes — purpose-built for near-duplicate photo identification.

Weight checkpoint is a TorchScript ``.pt`` file. Path resolves via the
``MEMORIA_DEDUP_MODEL_PATH`` env var, defaulting to
``~/.memoria/models/sscd_disc_mixup.torchscript.pt``. If the file is missing,
``_ensure_loaded()`` returns ``None`` and callers soft-skip — same contract
as ``compute_scene_embedding`` when the scene engine fails.

Hardware selection mirrors ``scene_engine.py``: CUDA → MPS → CPU.

License note: SSCD weights are CC-BY-NC 4.0 (research / non-commercial).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from runtime_paths import MODELS_DIR

logger = logging.getLogger("memoria.ml.dedup")

_DEFAULT_PATH = str(MODELS_DIR / "sscd_disc_mixup.torchscript.pt")
_SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
_SSCD_MIN_BYTES = 1_000_000

_state: dict[str, Any] | None = None
_state_lock = threading.Lock()
_load_failed = False


def _checkpoint_is_plausible(path: Path) -> bool:
    """Reject legacy partial downloads before Torch tries to load them.

    TorchScript checkpoints use PyTorch's ZIP serialization.  The previous
    downloader wrote directly to the final path, so an interrupted transfer
    could leave a large (and therefore size-check-passing) truncated file.
    ``is_zipfile`` checks the archive footer without loading the model.
    """
    try:
        return path.stat().st_size >= _SSCD_MIN_BYTES and zipfile.is_zipfile(path)
    except OSError:
        return False


def _pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_model_present(target: str | None = None) -> str | None:
    """Download the SSCD checkpoint if it isn't already on disk.

    Returns the resolved path on success, or None if download fails (caller
    should soft-skip dedup encoding for that run). Safe to call from the
    backfill task or a startup hook — does nothing when the file exists.
    """
    path = target or os.environ.get("MEMORIA_DEDUP_MODEL_PATH", _DEFAULT_PATH)
    p = Path(path)
    if p.exists() and _checkpoint_is_plausible(p):
        return str(p)

    if p.exists():
        logger.warning("[DEDUP] cached SSCD checkpoint is incomplete or invalid: %s", p)

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(
            "[DEDUP] cannot create checkpoint directory %s: %s",
            p.parent,
            exc,
        )
        return None
    logger.info("[DEDUP] downloading SSCD checkpoint to %s …", p)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{p.name}.",
            suffix=".part",
            dir=p.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(_SSCD_URL, timeout=300) as response:
                expected_header = response.headers.get("Content-Length")
                shutil.copyfileobj(response, temporary, length=1024 * 1024)
            temporary.flush()
            os.fsync(temporary.fileno())

        downloaded_bytes = temporary_path.stat().st_size
        expected_bytes = int(expected_header) if expected_header else None
        if downloaded_bytes < _SSCD_MIN_BYTES:
            raise RuntimeError(
                f"download was unexpectedly small ({downloaded_bytes} bytes)"
            )
        if expected_bytes is not None and downloaded_bytes != expected_bytes:
            raise RuntimeError(
                "download was incomplete "
                f"({downloaded_bytes}/{expected_bytes} bytes)"
            )
        os.replace(temporary_path, p)
        temporary_path = None
    except Exception as exc:
        logger.error("[DEDUP] checkpoint download failed: %s", exc)
        return None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    if not p.exists() or not _checkpoint_is_plausible(p):
        logger.error("[DEDUP] checkpoint missing or invalid after download: %s", p)
        return None

    logger.info("[DEDUP] checkpoint ready: %s (%.1f MB)", p, p.stat().st_size / 1e6)
    return str(p)


def _build_preprocess():
    """Standard ImageNet preprocessing pipeline used by the SSCD ResNet50.
    Resize so the short edge is 288 px, centre-crop to 288×288, convert to
    a [0, 1] tensor, then normalise with the canonical ImageNet mean/std so
    the input distribution matches what SSCD was trained on. """
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(288),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _load() -> dict[str, Any] | None:
    """Load model + transforms once. Called inside the singleton lock.

    Returns None if the checkpoint is unreachable; callers must treat that
    as a soft-skip (same as scene_engine when its configured model fails).
    """
    import torch

    path = ensure_model_present()
    if path is None:
        return None

    device = _pick_device()
    logger.info("Loading SSCD %s on %s …", path, device)

    try:
        model = torch.jit.load(path, map_location=device).eval()
    except Exception as exc:
        logger.error("Failed to load SSCD TorchScript at %s: %s", path, exc)
        return None

    preprocess = _build_preprocess()

    # Probe output dim with a dummy 288×288 zero tensor.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 288, 288, device=device)
        out = model(dummy)
    dim = int(out.shape[-1])

    logger.info("SSCD loaded; embedding_dim=%d", dim)
    return {
        "model": model,
        "preprocess": preprocess,
        "device": device,
        "dim": dim,
    }


def _ensure_loaded() -> dict[str, Any] | None:
    global _state, _load_failed
    if _state is not None:
        return _state
    if _load_failed:
        return None
    with _state_lock:
        if _state is None and not _load_failed:
            _state = _load()
            if _state is None:
                _load_failed = True
    return _state


def _l2_normalize(t):
    """Scale a tensor so its rows have length 1. pgvector's cosine-distance
    index requires unit-norm vectors for correct ordering. The +1e-9 is a
    tiny epsilon to dodge division by zero on (unlikely) zero outputs."""
    return t / (t.norm(dim=-1, keepdim=True) + 1e-9)


def encode_image(pil_or_path) -> list[float] | None:
    """Encode a PIL Image or filesystem path to a unit-norm 512-d float list.

    Returns None if the model isn't loadable — caller should soft-skip.
    """
    import torch
    from PIL import Image

    s = _ensure_loaded()
    if s is None:
        return None

    if isinstance(pil_or_path, (str, bytes, os.PathLike)):
        img = Image.open(pil_or_path).convert("RGB")
    else:
        img = pil_or_path if pil_or_path.mode == "RGB" else pil_or_path.convert("RGB")

    tensor = s["preprocess"](img).unsqueeze(0).to(s["device"])
    with torch.no_grad():
        emb = s["model"](tensor)
        emb = _l2_normalize(emb)
    return emb[0].cpu().tolist()


def encode_images_batch(pil_images) -> list[list[float]] | None:
    """Encode a list of PIL Images. Returns one unit-norm list per input.

    Empty input returns []. Returns None if the model isn't loadable.
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor

    if not pil_images:
        return []

    s = _ensure_loaded()
    if s is None:
        return None

    preprocess = s["preprocess"]
    n_workers = min(len(pil_images), 8)

    def _prep(img):
        return preprocess(img if img.mode == "RGB" else img.convert("RGB"))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        tensor_list = list(pool.map(_prep, pil_images))

    tensors = torch.stack(tensor_list).to(s["device"])

    with torch.no_grad():
        emb = s["model"](tensors)
        emb = _l2_normalize(emb)

    return [vec.cpu().tolist() for vec in emb]


def embedding_dim() -> int | None:
    s = _ensure_loaded()
    return s["dim"] if s else None
