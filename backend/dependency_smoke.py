"""Fail fast when the intentionally managed ML dependency set drifts."""

from __future__ import annotations

import inspect
import sys
from importlib import metadata


def _canonical(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed[_canonical(name)] = distribution.version
    return installed


def _require_exact(installed: dict[str, str], name: str, version: str) -> None:
    actual = installed.get(_canonical(name))
    if actual != version:
        raise RuntimeError(f"expected {name}=={version}, found {actual or 'missing'}")


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"cpu", "gpu"}:
        raise SystemExit("usage: dependency_smoke.py <cpu|gpu>")
    profile = sys.argv[1]

    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError("Memoria supports Python 3.10 through 3.12")

    installed = _versions()
    expected = {
        "numpy": "1.26.4",
        "Pillow": "12.3.0",
        "pillow-heif": "1.5.0",
        "opencv-contrib-python-headless": "4.11.0.86",
        "onnx": "1.17.0",
        "facenet-pytorch": "2.6.0",
        "insightface": "1.0.1",
    }
    for name, version in expected.items():
        _require_exact(installed, name, version)

    opencv_distributions = sorted(
        name for name in installed if name.startswith("opencv-")
    )
    if opencv_distributions != ["opencv-contrib-python-headless"]:
        raise RuntimeError(
            "exactly one OpenCV distribution is allowed; found "
            f"{opencv_distributions}"
        )

    import cv2
    import numpy
    import onnxruntime
    import pillow_heif
    import torch
    import torchvision
    from facenet_pytorch import InceptionResnetV1, MTCNN
    from insightface.app import FaceAnalysis
    from PIL import Image

    del InceptionResnetV1, numpy, pillow_heif
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("OpenCV build does not contain YuNet/FaceDetectorYN")
    if not str(torch.__version__).startswith("2.2.2"):
        raise RuntimeError(f"expected torch 2.2.2, found {torch.__version__}")
    if not str(torchvision.__version__).startswith("0.17.2"):
        raise RuntimeError(
            f"expected torchvision 0.17.2, found {torchvision.__version__}"
        )
    if onnxruntime.__version__ != "1.23.2":
        raise RuntimeError(
            f"expected ONNX Runtime 1.23.2, found {onnxruntime.__version__}"
        )

    # Construction loads only the small weights bundled in the wheel and
    # exercises its Pillow/Torch imports without downloading model data.
    detector = MTCNN(keep_all=True, device="cpu")
    detector.detect(Image.new("RGB", (64, 64), color="black"))
    del detector
    prepare_parameters = inspect.signature(FaceAnalysis.prepare).parameters
    if "ctx_id" not in prepare_parameters or "det_size" not in prepare_parameters:
        raise RuntimeError("installed InsightFace FaceAnalysis API is incompatible")

    if profile == "cpu":
        _require_exact(installed, "onnxruntime", "1.23.2")
        forbidden = sorted(
            name
            for name in installed
            if name == "onnxruntime-gpu" or name.startswith("nvidia-")
        )
        if forbidden:
            raise RuntimeError(f"CPU profile contains GPU packages: {forbidden}")
        if torch.version.cuda is not None:
            raise RuntimeError(f"CPU torch reports CUDA {torch.version.cuda}")
        gpu_providers = {
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
        } & set(onnxruntime.get_available_providers())
        if gpu_providers:
            raise RuntimeError(
                f"CPU ONNX Runtime exposes GPU providers: {sorted(gpu_providers)}"
            )
    else:
        _require_exact(installed, "onnxruntime-gpu", "1.23.2")
        if "onnxruntime" in installed:
            raise RuntimeError("GPU profile also contains CPU onnxruntime")
        if torch.version.cuda is None:
            raise RuntimeError("GPU profile installed a CPU-only torch wheel")

    print(f"Memoria {profile} dependency smoke check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
