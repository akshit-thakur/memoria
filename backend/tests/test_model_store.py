"""Framework- and database-free tests for Phase 2 persistence safety."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_store
from ml.base import ModelSpec
from ml.scene import builtin_profiles


def _expect_value_error(fn) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_descriptor_fingerprint_is_recomputed_and_tampering_is_rejected():
    descriptor = builtin_profiles()[0].as_dict()
    normalized = model_store.normalize_model_descriptor(descriptor)
    assert normalized["fingerprint"] == descriptor["fingerprint"]

    tampered = dict(descriptor)
    tampered["embedding_dim"] = int(descriptor["embedding_dim"]) + 1
    _expect_value_error(lambda: model_store.normalize_model_descriptor(tampered))


def test_builtin_and_custom_index_plans_are_dimension_safe():
    builtin = model_store.normalize_model_descriptor(builtin_profiles()[0])
    builtin_plan = model_store.scene_knn_plan(builtin)
    assert builtin_plan.dimension == 768
    assert builtin_plan.uses_hnsw is True

    custom = model_store.normalize_model_descriptor(
        ModelSpec(
            model_key="scene/test/custom-1024",
            task="scene",
            driver="hf",
            model_id="test/custom",
            input_resolution=(320, 320),
            embedding_dim=1024,
            provider_preferences=("cpu",),
            index_kind="exact",
        )
    )
    custom_plan = model_store.scene_knn_plan(custom)
    assert custom_plan.dimension == 1024
    assert custom_plan.uses_hnsw is False


class _FakeAsyncConnection:
    def __init__(self, model: dict) -> None:
        self.model = model
        self.query = ""
        self.params: tuple = ()

    async def fetchrow(self, query: str, *params):
        assert "FROM ml_models" in query
        assert params == (self.model["model_key"],)
        return self.model

    async def fetch(self, query: str, *params):
        self.query = query
        self.params = params
        return []


def test_custom_model_key_remains_bound_and_uses_exact_query_shape():
    malicious_key = "scene/custom/x' OR TRUE --"
    model = model_store.normalize_model_descriptor(
        ModelSpec(
            model_key=malicious_key,
            task="scene",
            driver="hf",
            model_id="test/custom-three",
            input_resolution=(32, 32),
            embedding_dim=3,
            provider_preferences=("cpu",),
            index_kind="exact",
        )
    )
    conn = _FakeAsyncConnection(model)
    rows = asyncio.run(
        model_store.query_scene_neighbors_async(
            conn,
            [1.0, 0.0, 0.0],
            model_key=malicious_key,
            exclude_media_id=9,
            limit=7,
        )
    )
    assert rows == []
    assert malicious_key not in conn.query
    assert conn.params == (malicious_key, "[1,0,0]", 9, 7)
    assert "embedding::vector(3)" in conn.query


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
