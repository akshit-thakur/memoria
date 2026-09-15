"""Shared contracts for pluggable Memoria model engines.

The contracts in this module deliberately have no torch, ONNX, or transformers
imports.  Model metadata can therefore be inspected by migrations, health
checks, and API code without loading a framework or allocating accelerator
memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


_PRECISIONS = frozenset({"auto", "fp32", "fp16", "bf16"})
_PROVIDERS = frozenset({"auto", "tensorrt", "onnx-cuda", "cuda", "mps", "cpu"})
_DISTANCES = frozenset({"cosine", "inner_product", "l2"})
_INDEX_KINDS = frozenset({"hnsw", "exact", "none"})


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable, self-describing identity for one model configuration.

    ``model_key`` is the stable database/API identity. ``fingerprint`` is a
    deterministic digest of every semantic field and is useful for detecting a
    configuration change that accidentally reused a key.
    """

    model_key: str
    task: str
    driver: str
    model_id: str
    input_resolution: tuple[int, int]
    embedding_dim: int | None = None
    weights: str | None = None
    revision: str | None = None
    default_precision: str = "auto"
    provider_preferences: tuple[str, ...] = ("cuda", "mps", "cpu")
    distance_metric: str = "cosine"
    normalized: bool = True
    index_kind: str = "hnsw"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("model_key", "task", "driver", "model_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())

        if (
            not isinstance(self.input_resolution, tuple)
            or len(self.input_resolution) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in self.input_resolution)
        ):
            raise ValueError("input_resolution must be a (positive_width, positive_height) tuple")
        if self.embedding_dim is not None and (
            not isinstance(self.embedding_dim, int)
            or isinstance(self.embedding_dim, bool)
            or not 1 <= self.embedding_dim <= 16_000
        ):
            raise ValueError("embedding_dim must be between 1 and 16000")

        precision = self.default_precision.strip().lower()
        if precision not in _PRECISIONS:
            raise ValueError(f"unsupported default_precision: {self.default_precision!r}")
        object.__setattr__(self, "default_precision", precision)

        providers = tuple(str(p).strip().lower() for p in self.provider_preferences)
        if not providers or any(p not in _PROVIDERS - {"auto"} for p in providers):
            raise ValueError(f"invalid provider_preferences: {self.provider_preferences!r}")
        # Preserve preference order while removing duplicates.
        providers = tuple(dict.fromkeys(providers))
        object.__setattr__(self, "provider_preferences", providers)

        distance = self.distance_metric.strip().lower()
        if distance not in _DISTANCES:
            raise ValueError(f"unsupported distance_metric: {self.distance_metric!r}")
        object.__setattr__(self, "distance_metric", distance)

        index_kind = self.index_kind.strip().lower()
        if index_kind not in _INDEX_KINDS:
            raise ValueError(f"unsupported index_kind: {self.index_kind!r}")
        object.__setattr__(self, "index_kind", index_kind)

        # Freeze a shallow copy so callers cannot mutate a frozen spec through
        # the original dict. Metadata values must be JSON-compatible because
        # they participate in the deterministic fingerprint.
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def as_dict(self, execution: Any | None = None) -> dict[str, Any]:
        """Return a JSON-safe description, optionally enriched at runtime."""

        data = {
            "model_key": self.model_key,
            "kind": self.task,
            "driver": self.driver,
            "model_name": self.model_id,
            "embedding_dim": self.embedding_dim,
            "weights": self.weights,
            "revision": self.revision,
            "input_width": self.input_resolution[0],
            "input_height": self.input_resolution[1],
            "default_precision": self.default_precision,
            "preferred_providers": list(self.provider_preferences),
            "distance_metric": self.distance_metric,
            "normalized": self.normalized,
            "index_kind": self.index_kind,
            "metadata": dict(self.metadata),
            "fingerprint": self.fingerprint,
            # Backward-compatible field spellings accepted by early Phase 2
            # callers. New persistence code should use the names above.
            "task": self.task,
            "model_id": self.model_id,
            "input_resolution": list(self.input_resolution),
            "provider_preferences": list(self.provider_preferences),
        }
        if execution is not None:
            runtime = execution.as_dict() if hasattr(execution, "as_dict") else dict(execution)
            data["actual_provider"] = runtime.get("actual_provider")
            data["actual_precision"] = runtime.get("actual_precision")
        return data

    @property
    def kind(self) -> str:
        return self.task

    @property
    def model_name(self) -> str:
        return self.model_id

    @property
    def input_width(self) -> int:
        return self.input_resolution[0]

    @property
    def input_height(self) -> int:
        return self.input_resolution[1]

    @property
    def preferred_providers(self) -> tuple[str, ...]:
        return self.provider_preferences

    @property
    def fingerprint(self) -> str:
        """SHA-256 identity over the canonical semantic description."""

        payload = {
            "model_key": self.model_key,
            "task": self.task,
            "driver": self.driver,
            "model_id": self.model_id,
            "weights": self.weights,
            "revision": self.revision,
            "input_resolution": list(self.input_resolution),
            "embedding_dim": self.embedding_dim,
            "default_precision": self.default_precision,
            "provider_preferences": list(self.provider_preferences),
            "distance_metric": self.distance_metric,
            "normalized": self.normalized,
            "index_kind": self.index_kind,
            "metadata": dict(self.metadata),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_embedding(values: Iterable[float], expected_dim: int) -> list[float]:
    """Return a plain finite float list, rejecting wrong-dimensional output."""

    if not isinstance(expected_dim, int) or isinstance(expected_dim, bool) or expected_dim <= 0:
        raise ValueError("expected_dim must be a positive integer")
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding contains a non-numeric value") from exc
    if len(result) != expected_dim:
        raise ValueError(f"model returned {len(result)} dimensions; expected {expected_dim}")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("embedding contains NaN or infinity")
    return result


def validate_embeddings(
    vectors: Iterable[Iterable[float]], expected_dim: int, expected_count: int | None = None
) -> list[list[float]]:
    """Validate a batch and optionally ensure one result per input."""

    result = [validate_embedding(vector, expected_dim) for vector in vectors]
    if expected_count is not None and len(result) != expected_count:
        raise ValueError(f"model returned {len(result)} embeddings; expected {expected_count}")
    return result


class BaseModelEngine:
    """Small lifecycle helper shared by the loaded model implementations."""

    def __init__(self, spec: ModelSpec, execution: Any) -> None:
        self.spec = spec
        self.execution = execution
        self._inference_lock = threading.RLock()
        self._closed = False

    @contextmanager
    def inference_guard(self):
        """Serialize inference and reject calls after explicit shutdown."""

        with self._inference_lock:
            if self._closed:
                raise RuntimeError(f"model engine {self.spec.model_key!r} is closed")
            yield

    @property
    def runtime_info(self) -> dict[str, Any]:
        execution = self.execution.as_dict() if hasattr(self.execution, "as_dict") else {}
        model = self.spec.as_dict(self.execution)
        return {
            **model,
            "actual_provider": execution.get("actual_provider"),
            "actual_precision": execution.get("actual_precision"),
            "model": model,
            "execution": execution,
            "loaded": self.loaded,
        }

    @property
    def loaded(self) -> bool:
        """Whether heavyweight model state has been initialized."""
        raise NotImplementedError

    def close(self) -> None:
        """Release model references and framework caches."""
        raise NotImplementedError


class SceneEmbeddingEngine(BaseModelEngine):
    """Shared image/text embedding interface for semantic search models."""

    def _dimension(self) -> int:
        if self.spec.embedding_dim is None:
            raise RuntimeError(f"scene model {self.spec.model_key!r} has no embedding dimension")
        return self.spec.embedding_dim

    def validate(self, vector: Iterable[float]) -> list[float]:
        return validate_embedding(vector, self._dimension())

    def validate_batch(
        self, vectors: Iterable[Iterable[float]], expected_count: int
    ) -> list[list[float]]:
        return validate_embeddings(vectors, self._dimension(), expected_count)

    def encode_image(self, image_or_path: Any) -> list[float]:
        raise NotImplementedError

    def encode_images(self, images: Sequence[Any]) -> list[list[float]]:
        raise NotImplementedError

    def encode_text(self, text: str) -> list[float]:
        raise NotImplementedError


class FeatureExtractorEngine(BaseModelEngine):
    """Generic contract for registered non-scene feature extractors.

    Drivers such as perceptual hashes, blur scorers, object descriptors, or
    future dedup models can expose structured features through the same lazy
    lifecycle and self-describing :class:`ModelSpec` registry without being
    forced into the image/text embedding interface.
    """

    def extract(self, item: Any) -> Any:
        raise NotImplementedError

    def extract_batch(self, items: Sequence[Any]) -> list[Any]:
        """Correct default for CPU extractors; accelerated drivers may override."""
        return [self.extract(item) for item in items]
