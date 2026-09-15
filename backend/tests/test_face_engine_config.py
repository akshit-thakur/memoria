"""DB- and model-weight-free tests for the Phase 2 face abstraction.

Runs under pytest, or standalone: ``python tests/test_face_engine_config.py``.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.base import ModelSpec
from ml.face import (
    DEFAULT_FACE_DETECTORS,
    FACE_VECTOR_DIMENSION,
    FaceAnalysisEngine,
    FaceEngineConfig,
)
from ml.providers import ExecutionPlan


def _expect_raises(error_type, fn) -> Exception:
    try:
        fn()
    except error_type as exc:
        return exc
    raise AssertionError(f"expected {error_type.__name__}")


def _cpu_plan() -> ExecutionPlan:
    return ExecutionPlan(
        requested_provider="cpu",
        actual_provider="cpu",
        device="cpu",
        requested_precision="fp32",
        actual_precision="fp32",
        available_providers=("cpu",),
    )


class _FakeFaceEngine(FaceAnalysisEngine):
    """Minimal engine used to exercise framework-free vector validation."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec, _cpu_plan())

    @property
    def loaded(self) -> bool:
        return False

    def detect(self, image):
        return []

    def detect_batch(self, images, max_workers: int = 2):
        return [[] for _ in images]

    def close(self) -> None:
        self._closed = True


def test_default_env_config_and_spec_are_self_describing_without_weights():
    with patch.dict(os.environ, {}, clear=True):
        config = FaceEngineConfig.from_env()
        spec = config.to_model_spec()

    assert config.engine_key == "ensemble-v1"
    assert config.detectors == DEFAULT_FACE_DETECTORS
    assert config.embedder == "insightface"
    assert config.min_consensus == 2
    assert config.batch_concurrency == 2
    assert config.embedding_model_key == "face/insightface-buffalo-l-arcface"
    # The stable default key is intentionally short; detector and runtime
    # shape details remain in the immutable descriptor metadata/fingerprint.
    assert config.detection_pipeline_key == "face/ensemble-v1"

    assert spec.model_key == "face/ensemble-v1"
    assert spec.task == "face"
    assert spec.driver == "face-ensemble"
    assert spec.input_resolution == (640, 640)
    assert spec.embedding_dim == FACE_VECTOR_DIMENSION
    assert spec.default_precision == "auto"
    assert spec.provider_preferences == ("tensorrt", "onnx-cuda", "cpu")
    assert spec.normalized is True
    assert spec.metadata["fixed_storage_dimension"] == FACE_VECTOR_DIMENSION


def test_detector_and_embedder_aliases_are_canonicalized():
    config = FaceEngineConfig(
        engine_key="default",
        detectors=("facenet-mtcnn", "opencv-yunet", "insightface-scrfd"),
        embedder="arcface",
    )
    assert config.engine_key == "ensemble-v1"
    assert config.detectors == ("mtcnn", "yunet", "insightface")
    assert config.embedder == "insightface"

    facenet = FaceEngineConfig(
        detectors=("facenet-mtcnn", "opencv-yunet"),
        embedder="facenet-vggface2",
    )
    assert facenet.detectors == ("mtcnn", "yunet")
    assert facenet.embedder == "facenet"
    assert facenet.embedding_model_key == "face/facenet-vggface2"


def test_env_aliases_and_separators_are_supported():
    env = {
        "MEMORIA_FACE_ENGINE": "ensemble",
        "MEMORIA_FACE_DETECTORS": "facenet-mtcnn + opencv-yunet, scrfd",
        "MEMORIA_FACE_EMBEDDER": "insightface-arcface",
        "MEMORIA_FACE_MIN_CONSENSUS": "3",
    }
    config = FaceEngineConfig.from_env(env)
    assert config.detectors == DEFAULT_FACE_DETECTORS
    assert config.embedder == "insightface"
    assert config.min_consensus == 3


def test_invalid_consensus_and_unknown_drivers_fail_before_loading():
    _expect_raises(
        ValueError,
        lambda: FaceEngineConfig(
            detectors=("mtcnn", "yunet"), embedder="facenet", min_consensus=1
        ),
    )
    _expect_raises(
        ValueError,
        lambda: FaceEngineConfig(
            detectors=("mtcnn", "yunet"), embedder="facenet", min_consensus=3
        ),
    )
    _expect_raises(
        ValueError,
        lambda: FaceEngineConfig(detectors=("mtcnn", "mystery", "yunet")),
    )
    _expect_raises(
        ValueError,
        lambda: FaceEngineConfig(embedder="unknown-embedder"),
    )
    _expect_raises(ValueError, lambda: FaceEngineConfig(engine_key="unknown-engine"))
    _expect_raises(
        ValueError,
        lambda: FaceEngineConfig(insightface_model="unregistered_pack"),
    )


def test_fixed_512_embedding_validation_rejects_bad_vectors():
    engine = _FakeFaceEngine(FaceEngineConfig().to_model_spec())
    valid = [float(index % 11) for index in range(FACE_VECTOR_DIMENSION)]
    assert engine.validate_face_embedding(valid) == valid

    _expect_raises(
        ValueError,
        lambda: engine.validate_face_embedding(valid[:-1]),
    )
    non_finite = valid.copy()
    non_finite[3] = math.nan
    _expect_raises(
        ValueError,
        lambda: engine.validate_face_embedding(non_finite),
    )
    non_finite[3] = math.inf
    _expect_raises(
        ValueError,
        lambda: engine.validate_face_embedding(non_finite),
    )


def test_face_validation_rejects_non_512_model_spec():
    wrong_spec = ModelSpec(
        model_key="face/test-256",
        task="face",
        driver="fake",
        model_id="fake-256",
        input_resolution=(32, 32),
        embedding_dim=256,
        provider_preferences=("cpu",),
    )
    engine = _FakeFaceEngine(wrong_spec)
    error = _expect_raises(
        RuntimeError,
        lambda: engine.validate_face_embedding([0.0] * 256),
    )
    assert "fixed at 512" in str(error)


def test_face_engine_model_spec_does_not_construct_runtime_engine():
    # Importing the facade and inspecting metadata must not import framework
    # drivers or call the lazy runtime constructor.
    heavy_modules = ("cv2", "torch", "insightface")
    already_loaded = {name: name in sys.modules for name in heavy_modules}
    sys.modules.pop("face_engine", None)
    facade = importlib.import_module("face_engine")

    with patch.object(
        facade,
        "get_face_engine",
        side_effect=AssertionError("runtime engine must not be constructed"),
    ):
        with patch.dict(os.environ, {}, clear=True):
            spec = facade.model_spec()

    assert spec.model_key == "face/ensemble-v1"
    assert spec.embedding_dim == FACE_VECTOR_DIMENSION
    for name, was_loaded in already_loaded.items():
        if not was_loaded:
            assert name not in sys.modules


if __name__ == "__main__":
    functions = [
        value for key, value in sorted(globals().items()) if key.startswith("test_")
    ]
    for function in functions:
        function()
        print(f"ok  {function.__name__}")
    print(f"\n{len(functions)} passed")
