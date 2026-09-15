"""DB-free tests for the Phase 2 model registry and scene primitives.

No test below constructs an engine, imports a model framework, downloads
weights, or touches PostgreSQL. Runs under pytest, or standalone with
``python tests/test_ml_models.py``.
"""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.base import validate_embedding
from ml import providers
from ml.scene import (
    HF_SIGLIP2_KEY,
    OPENCLIP_SIGLIP_KEY,
    OPENCLIP_VIT_L14_KEY,
    builtin_profiles,
    config_from_record,
    resolve_configured_scene_spec,
)
from ml.registry import model_registry


_SCENE_ENV_KEYS = (
    "MEMORIA_SCENE_MODEL",
    "MEMORIA_SCENE_PROFILE",
    "MEMORIA_SCENE_DRIVER",
    "MEMORIA_SCENE_WEIGHTS",
    "MEMORIA_SCENE_REVISION",
    "MEMORIA_SCENE_DIM",
    "MEMORIA_SCENE_INPUT_RESOLUTION",
    "MEMORIA_SCENE_PRECISION",
    "MEMORIA_SCENE_PROVIDERS",
    "MEMORIA_SCENE_PROVIDER",
    "MEMORIA_SCENE_DEVICE",
    "MEMORIA_CLIP_MODEL",
)


@contextmanager
def _isolated_scene_env(**overrides: str):
    saved = {key: os.environ.get(key) for key in _SCENE_ENV_KEYS}
    try:
        for key in _SCENE_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(overrides)
        yield
    finally:
        for key in _SCENE_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def _assert_raises(error_type, fn) -> None:
    try:
        fn()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_builtin_scene_profiles_are_self_describing():
    profiles = {spec.model_key: spec for spec in builtin_profiles()}
    assert profiles[OPENCLIP_VIT_L14_KEY].embedding_dim == 768
    assert profiles[OPENCLIP_VIT_L14_KEY].input_resolution == (224, 224)
    assert profiles[OPENCLIP_SIGLIP_KEY].embedding_dim == 1152
    assert profiles[OPENCLIP_SIGLIP_KEY].input_resolution == (384, 384)
    assert profiles[HF_SIGLIP2_KEY].embedding_dim == 1152

    descriptor = profiles[HF_SIGLIP2_KEY].as_dict()
    assert descriptor["kind"] == "scene"
    assert descriptor["model_name"] == "google/siglip2-so400m-patch14-384"
    assert descriptor["input_width"] == 384
    assert descriptor["input_height"] == 384
    assert descriptor["normalized"] is True
    assert descriptor["distance_metric"] == "cosine"
    assert len(descriptor["fingerprint"]) == 64


def test_legacy_memoria_clip_model_fallback():
    with _isolated_scene_env(
        MEMORIA_CLIP_MODEL="ViT-B-32/laion2b_s34b_b79k",
        MEMORIA_SCENE_DIM="512",
        MEMORIA_SCENE_INPUT_RESOLUTION="256",
    ):
        spec = resolve_configured_scene_spec()
    assert spec.driver == "open_clip"
    assert spec.model_id == "ViT-B-32"
    assert spec.weights == "laion2b_s34b_b79k"
    assert spec.embedding_dim == 512
    assert spec.input_resolution == (256, 256)
    assert spec.index_kind == "exact"


def test_embedding_validation_rejects_non_finite_and_wrong_length():
    assert validate_embedding((1, 2.5, -3), 3) == [1.0, 2.5, -3.0]
    _assert_raises(ValueError, lambda: validate_embedding([1.0, 2.0], 3))
    _assert_raises(ValueError, lambda: validate_embedding([1.0, math.nan], 2))
    _assert_raises(ValueError, lambda: validate_embedding([1.0, math.inf], 2))


def test_provider_resolution_falls_back_to_portable_cpu():
    spec = model_registry.resolve(OPENCLIP_VIT_L14_KEY)
    with _isolated_scene_env(
        MEMORIA_SCENE_PROVIDERS="cuda,cpu",
        MEMORIA_SCENE_PRECISION="fp16",
    ):
        with patch.object(providers, "detect_available_providers", return_value=("cpu",)):
            plan = providers.resolve_execution(
                spec,
                supported_providers=("cuda", "mps", "cpu"),
            )
    assert plan.requested_providers == ("cuda", "cpu")
    assert plan.actual_provider == "cpu"
    assert plan.device == "cpu"
    assert plan.actual_precision == "fp32"
    assert plan.as_dict()["actual_provider"] == "cpu"


def test_model_record_round_trip_without_loading_engine():
    record = {
        "model_key": "scene/hf/example/siglip-test",
        "kind": "scene",
        "driver": "hf",
        "model_name": "example/siglip-test",
        "weights": None,
        "revision": "test-revision",
        "embedding_dim": 1024,
        "input_width": 320,
        "input_height": 320,
        "default_precision": "auto",
        "preferred_providers": ["cuda", "cpu"],
        "normalized": True,
        "distance_metric": "cosine",
        "metadata": {"supports_text": True, "test_only": True},
    }
    spec = config_from_record(record)
    restored = config_from_record(spec.as_dict())
    assert restored is spec
    assert restored.fingerprint == spec.fingerprint
    assert restored.embedding_dim == 1024
    assert restored.input_resolution == (320, 320)
    assert restored.provider_preferences == ("cuda", "cpu")
    assert restored.index_kind == "exact"


def test_profile_aliases_resolve_to_canonical_models():
    assert model_registry.resolve("openclip-vit-l-14").model_key == OPENCLIP_VIT_L14_KEY
    assert (
        model_registry.resolve("openclip-siglip-so400m-384").model_key
        == OPENCLIP_SIGLIP_KEY
    )
    assert model_registry.resolve("siglip2-so400m-384").model_key == HF_SIGLIP2_KEY
    assert model_registry.resolve("hf-siglip2-so400m-384").model_key == HF_SIGLIP2_KEY


if __name__ == "__main__":
    functions = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for function in functions:
        function()
        print(f"ok  {function.__name__}")
    print(f"\n{len(functions)} passed")
