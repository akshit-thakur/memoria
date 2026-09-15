"""Contracts and environment configuration for pluggable face analysis.

This module stays framework-free: importing it does not load Torch, OpenCV,
InsightFace, or allocate accelerator memory. Heavy driver construction remains
lazy in :mod:`face_detection.face_pipeline`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .base import BaseModelEngine, ModelSpec, validate_embedding
from .providers import ExecutionPlan, resolve_execution
from .registry import model_registry


FACE_VECTOR_DIMENSION = 512
DEFAULT_FACE_DETECTORS = ("mtcnn", "yunet", "insightface")
_DETECTOR_KEYS = frozenset(DEFAULT_FACE_DETECTORS)
_DETECTOR_ALIASES = {
    "facenet-mtcnn": "mtcnn",
    "opencv-yunet": "yunet",
    "insightface-scrfd": "insightface",
    "scrfd": "insightface",
}
_EMBEDDER_ALIASES = {
    "arcface": "insightface",
    "insightface-arcface": "insightface",
    "facenet-vggface2": "facenet",
    "vggface2": "facenet",
    "disabled": "none",
}
_ENGINE_ALIASES = {
    "ensemble": "ensemble-v1",
    "default": "ensemble-v1",
    "face/ensemble-v1": "ensemble-v1",
}


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = str(env.get(key, default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True, slots=True)
class FaceEngineConfig:
    """Validated, fingerprintable face-engine configuration."""

    engine_key: str = "ensemble-v1"
    detectors: tuple[str, ...] = DEFAULT_FACE_DETECTORS
    embedder: str = "insightface"
    requested_provider: str = "auto"
    requested_precision: str = "auto"
    min_consensus: int = 2
    batch_concurrency: int = 2
    insightface_model: str = "buffalo_l"

    def __post_init__(self) -> None:
        raw_engine = self.engine_key.strip().lower()
        engine = _ENGINE_ALIASES.get(raw_engine, raw_engine)
        if engine != "ensemble-v1":
            raise ValueError(
                f"Unknown MEMORIA_FACE_ENGINE {self.engine_key!r}; "
                "available engine: ensemble-v1"
            )
        object.__setattr__(self, "engine_key", engine)

        detectors = tuple(
            dict.fromkeys(
                _DETECTOR_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
                for value in self.detectors
            )
        )
        unknown = sorted(set(detectors) - _DETECTOR_KEYS)
        if unknown:
            raise ValueError(f"Unknown face detector(s): {', '.join(unknown)}")
        if not detectors:
            raise ValueError("At least one face detector must be configured")
        object.__setattr__(self, "detectors", detectors)

        embedder = self.embedder.strip().lower()
        embedder = _EMBEDDER_ALIASES.get(embedder, embedder)
        if embedder not in {"insightface", "facenet", "none"}:
            raise ValueError(f"Unknown face embedder: {self.embedder!r}")
        if embedder == "insightface" and "insightface" not in detectors:
            raise ValueError(
                "The current InsightFace embedder adapter requires the "
                "insightface detector to be enabled"
            )
        object.__setattr__(self, "embedder", embedder)

        if self.min_consensus < 2:
            raise ValueError("MEMORIA_FACE_MIN_CONSENSUS cannot be lower than 2")
        if self.min_consensus > len(detectors):
            raise ValueError(
                "MEMORIA_FACE_MIN_CONSENSUS exceeds the configured detector count"
            )
        if not 1 <= self.batch_concurrency <= 8:
            raise ValueError("MEMORIA_FACE_BATCH_CONCURRENCY must be between 1 and 8")
        model_name = self.insightface_model.strip()
        if not model_name:
            raise ValueError("MEMORIA_FACE_INSIGHTFACE_MODEL cannot be empty")
        if model_name != "buffalo_l":
            raise ValueError(
                "Unregistered InsightFace model pack: "
                f"{model_name!r}; available pack: 'buffalo_l'"
            )
        object.__setattr__(self, "insightface_model", model_name)
        object.__setattr__(
            self, "requested_provider", self.requested_provider.strip().lower() or "auto"
        )
        object.__setattr__(
            self, "requested_precision", self.requested_precision.strip().lower() or "auto"
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "FaceEngineConfig":
        env = os.environ if environ is None else environ
        detector_raw = str(
            env.get("MEMORIA_FACE_DETECTORS", ",".join(DEFAULT_FACE_DETECTORS))
        )
        detectors = tuple(
            value for value in re.split(r"[,+\s]+", detector_raw.strip().lower()) if value
        )
        return cls(
            engine_key=str(env.get("MEMORIA_FACE_ENGINE", "ensemble-v1")),
            detectors=detectors,
            embedder=str(env.get("MEMORIA_FACE_EMBEDDER", "insightface")),
            requested_provider=str(
                env.get(
                    "MEMORIA_FACE_PROVIDERS",
                    env.get(
                        "MEMORIA_FACE_PROVIDER",
                        env.get("MEMORIA_FACE_DEVICE", "auto"),
                    ),
                )
            ),
            requested_precision=str(env.get("MEMORIA_FACE_PRECISION", "auto")),
            min_consensus=_env_int(env, "MEMORIA_FACE_MIN_CONSENSUS", 2),
            batch_concurrency=_env_int(env, "MEMORIA_FACE_BATCH_CONCURRENCY", 2),
            insightface_model=str(
                env.get("MEMORIA_FACE_INSIGHTFACE_MODEL", "buffalo_l")
            ),
        )

    @property
    def embedding_model_key(self) -> str | None:
        if self.embedder == "insightface":
            model = self.insightface_model.strip().lower().replace("_", "-")
            return f"face/insightface-{model}-arcface"
        if self.embedder == "facenet":
            return "face/facenet-vggface2"
        return None

    @property
    def detection_pipeline_key(self) -> str:
        """Return the registered model key for this exact face pipeline."""

        default_shape = (
            self.detectors == DEFAULT_FACE_DETECTORS
            and self.embedder == "insightface"
            and self.insightface_model == "buffalo_l"
            and self.min_consensus == 2
            and self.batch_concurrency == 2
        )
        model_key = f"face/{self.engine_key}"
        if not default_shape:
            model_key += f"/{self.pipeline_fingerprint[:12]}"
        return model_key

    @property
    def pipeline_fingerprint(self) -> str:
        """Identity of settings that can change persisted face analysis."""

        payload = {
            "engine": self.engine_key,
            "detectors": list(self.detectors),
            "embedder": self.embedder,
            "insightface_model": self.insightface_model,
            "min_consensus": self.min_consensus,
            "batch_concurrency": self.batch_concurrency,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        """Full runtime identity used to reload the lazy singleton safely."""

        payload = {
            "pipeline": self.pipeline_fingerprint,
            "provider": self.requested_provider,
            "precision": self.requested_precision,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_model_spec(self) -> ModelSpec:
        """Describe the complete configured analysis pipeline."""

        if self.embedder == "insightface":
            provider_preferences = ("tensorrt", "onnx-cuda", "cpu")
        elif self.embedder == "facenet":
            provider_preferences = ("cuda", "cpu")
        else:
            provider_preferences = ("onnx-cuda", "cuda", "cpu")
        return ModelSpec(
            model_key=self.detection_pipeline_key,
            task="face",
            driver="face-ensemble",
            model_id=self.engine_key,
            input_resolution=(640, 640),
            embedding_dim=(
                FACE_VECTOR_DIMENSION if self.embedding_model_key is not None else None
            ),
            default_precision="auto",
            provider_preferences=provider_preferences,
            distance_metric="cosine",
            normalized=True,
            metadata={
                "detectors": list(self.detectors),
                "embedder": self.embedder,
                "embedding_model_key": self.embedding_model_key,
                "detection_pipeline_key": self.detection_pipeline_key,
                "minimum_consensus": self.min_consensus,
                "batch_concurrency": self.batch_concurrency,
                "fixed_storage_dimension": FACE_VECTOR_DIMENSION,
                "driver_provider_support": {
                    "mtcnn": ["cuda", "cpu"],
                    "yunet": ["cpu"],
                    "insightface": ["tensorrt", "onnx-cuda", "cpu"],
                },
            },
        )


def resolve_face_execution(config: FaceEngineConfig, spec: ModelSpec) -> ExecutionPlan:
    requested_provider = config.requested_provider
    if config.embedder == "insightface":
        # InsightFace acceleration is provided by ONNX Runtime, not the
        # PyTorch CUDA device probe. TensorRT remains an explicit opt-in
        # because first use compiles large engine caches.
        if requested_provider == "auto":
            trt_opt_in = os.getenv(
                "MEMORIA_INSIGHTFACE_USE_TRT", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            requested_provider = (
                "tensorrt,onnx-cuda,cpu" if trt_opt_in else "onnx-cuda,cpu"
            )
        else:
            # The generic provider abstraction accepts an ordered list. Map
            # the PyTorch-oriented ``cuda`` alias per entry because
            # InsightFace acceleration is supplied by ONNX Runtime.
            providers = requested_provider.replace(";", ",").split(",")
            requested_provider = ",".join(
                "onnx-cuda" if value.strip() == "cuda" else value.strip()
                for value in providers
                if value.strip()
            )
    plan = resolve_execution(
        spec,
        requested_provider=requested_provider,
        requested_precision=config.requested_precision,
    )
    # The current adapters only turn on reduced precision for TensorRT. Torch
    # MTCNN, OpenCV YuNet, and the plain InsightFace CUDA EP remain fp32, so do
    # not advertise a precision the composite engine does not actually apply.
    # TensorRT is configured with ``trt_fp16_enable`` in the face pipeline;
    # there is no active TensorRT bf16 flag. Report the precision that is
    # actually executed, even when a caller requested bf16.
    actual_precision = plan.actual_precision
    reason = plan.fallback_reason
    if plan.actual_provider == "tensorrt":
        actual_precision = (
            "fp32" if plan.requested_precision == "fp32" else "fp16"
        )
        if plan.requested_precision == "bf16":
            bf16_reason = (
                "TensorRT face pipeline enables FP16 only; requested bf16 "
                "was executed as fp16"
            )
            reason = f"{reason}; {bf16_reason}" if reason else bf16_reason
    elif actual_precision != "fp32":
        reason = "face adapters apply reduced precision only with TensorRT"
        if plan.fallback_reason:
            reason = f"{plan.fallback_reason}; {reason}"
        actual_precision = "fp32"

    if actual_precision != plan.actual_precision or reason != plan.fallback_reason:
        plan = ExecutionPlan(
            requested_provider=plan.requested_provider,
            actual_provider=plan.actual_provider,
            device=plan.device,
            requested_precision=plan.requested_precision,
            actual_precision=actual_precision,
            available_providers=plan.available_providers,
            requested_providers=plan.requested_providers,
            fallback_reason=reason,
        )
    return plan


def reconcile_face_onnx_execution(
    plan: ExecutionPlan,
    active_session_providers: Sequence[str],
    *,
    tensorrt_fp16_enabled: bool,
) -> ExecutionPlan:
    """Reconcile a planned runtime with the loaded recognition session.

    Provider discovery only proves an execution provider is installed. This
    second pass records the provider chain selected by the concrete ONNX
    Runtime recognition session after InsightFace has prepared its models.
    """
    active = tuple(dict.fromkeys(
        str(provider).strip()
        for provider in active_session_providers
        if str(provider).strip()
    ))
    if "TensorrtExecutionProvider" in active:
        actual_provider = "tensorrt"
        actual_precision = "fp16" if tensorrt_fp16_enabled else "fp32"
        device = "cuda"
    elif "CUDAExecutionProvider" in active:
        actual_provider, actual_precision, device = "onnx-cuda", "fp32", "cuda"
    elif "CPUExecutionProvider" in active:
        actual_provider, actual_precision, device = "cpu", "fp32", "cpu"
    else:
        return plan

    reason = plan.fallback_reason
    if actual_provider != plan.actual_provider:
        detail = (
            "loaded InsightFace recognition session selected "
            f"{actual_provider} instead of planned {plan.actual_provider}"
        )
        reason = f"{reason}; {detail}" if reason else detail
    if (
        actual_provider == plan.actual_provider
        and actual_precision == plan.actual_precision
        and device == plan.device
        and reason == plan.fallback_reason
    ):
        return plan
    return ExecutionPlan(
        requested_provider=plan.requested_provider,
        actual_provider=actual_provider,
        device=device,
        requested_precision=plan.requested_precision,
        actual_precision=actual_precision,
        available_providers=plan.available_providers,
        requested_providers=plan.requested_providers,
        fallback_reason=reason,
    )


class FaceDetectionEngine(BaseModelEngine):
    """Contract for a face detector, independent of embedding generation."""

    def detect(self, image: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    def detect_batch(
        self, images: Sequence[Any], max_workers: int = 2
    ) -> list[list[dict[str, Any]]]:
        raise NotImplementedError


class FaceEmbeddingEngine(BaseModelEngine):
    """Contract for drivers that encode normalized, cropped face images."""

    def validate_face_embedding(self, values: Sequence[float]) -> list[float]:
        dimension = self.spec.embedding_dim
        if dimension != FACE_VECTOR_DIMENSION:
            raise RuntimeError(
                "Face storage is fixed at 512 dimensions; "
                f"configured model declares {dimension!r}"
            )
        return validate_embedding(values, FACE_VECTOR_DIMENSION)

    def embed(self, face: Any) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, faces: Sequence[Any]) -> list[list[float]]:
        """Correct default for drivers without native batched inference."""

        return [self.embed(face) for face in faces]


class FaceAnalysisEngine(FaceDetectionEngine):
    """Compatibility contract for composite detector/embedder pipelines."""

    def validate_face_embedding(self, values: Sequence[float]) -> list[float]:
        dimension = self.spec.embedding_dim
        if dimension != FACE_VECTOR_DIMENSION:
            raise RuntimeError(
                "Face storage is fixed at 512 dimensions; "
                f"configured model declares {dimension!r}"
            )
        return validate_embedding(values, FACE_VECTOR_DIMENSION)

    def reset(self) -> None:
        self.close()


# Register the compatibility-preserving default pipeline as a discoverable
# runtime profile. Its heavyweight factory remains owned by face_engine.py so
# importing the catalog never imports OpenCV/Torch/InsightFace.
DEFAULT_FACE_ENGINE_SPEC = model_registry.register(
    FaceEngineConfig().to_model_spec(),
    aliases=("face", "face/ensemble", "face-default"),
)


# Catalog the vector spaces independently from the configurable detector
# pipeline. Equal dimensions never imply that these two spaces are compatible.
INSIGHTFACE_ARCFACE_SPEC = model_registry.register(
    ModelSpec(
        model_key="face/insightface-buffalo-l-arcface",
        task="face-embedding",
        driver="insightface",
        model_id="buffalo_l/arcface",
        input_resolution=(112, 112),
        embedding_dim=FACE_VECTOR_DIMENSION,
        default_precision="fp32",
        provider_preferences=("tensorrt", "onnx-cuda", "cpu"),
        distance_metric="cosine",
        normalized=True,
    ),
    aliases=("arcface", "insightface-arcface"),
)

FACENET_VGGFACE2_SPEC = model_registry.register(
    ModelSpec(
        model_key="face/facenet-vggface2",
        task="face-embedding",
        driver="facenet-pytorch",
        model_id="inception-resnet-v1/vggface2",
        input_resolution=(160, 160),
        embedding_dim=FACE_VECTOR_DIMENSION,
        default_precision="fp32",
        provider_preferences=("cuda", "cpu"),
        distance_metric="cosine",
        normalized=True,
    ),
    aliases=("facenet",),
)


__all__ = [
    "DEFAULT_FACE_DETECTORS",
    "DEFAULT_FACE_ENGINE_SPEC",
    "FACE_VECTOR_DIMENSION",
    "FACENET_VGGFACE2_SPEC",
    "FaceAnalysisEngine",
    "FaceDetectionEngine",
    "FaceEmbeddingEngine",
    "FaceEngineConfig",
    "INSIGHTFACE_ARCFACE_SPEC",
    "reconcile_face_onnx_execution",
    "resolve_face_execution",
]
