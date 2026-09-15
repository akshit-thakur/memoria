"""Public contracts for Memoria's pluggable ML engines."""

from .base import (
    BaseModelEngine,
    FeatureExtractorEngine,
    ModelSpec,
    SceneEmbeddingEngine,
    validate_embedding,
    validate_embeddings,
)
from .providers import ExecutionPlan, detect_available_providers, resolve_execution
from .registry import ModelRegistry, model_registry
from .face import (
    DEFAULT_FACE_DETECTORS,
    DEFAULT_FACE_ENGINE_SPEC,
    FACE_VECTOR_DIMENSION,
    FACENET_VGGFACE2_SPEC,
    INSIGHTFACE_ARCFACE_SPEC,
    FaceAnalysisEngine,
    FaceDetectionEngine,
    FaceEmbeddingEngine,
    FaceEngineConfig,
    resolve_face_execution,
)
from .scene import (
    BUILTIN_SCENE_SPECS,
    builtin_profiles,
    catalog,
    config_from_record,
    configured_model_spec,
    get_engine,
    model_spec,
    reset_engine,
)

__all__ = [
    "BaseModelEngine",
    "BUILTIN_SCENE_SPECS",
    "DEFAULT_FACE_DETECTORS",
    "DEFAULT_FACE_ENGINE_SPEC",
    "ExecutionPlan",
    "FeatureExtractorEngine",
    "FACENET_VGGFACE2_SPEC",
    "FACE_VECTOR_DIMENSION",
    "FaceAnalysisEngine",
    "FaceDetectionEngine",
    "FaceEmbeddingEngine",
    "FaceEngineConfig",
    "INSIGHTFACE_ARCFACE_SPEC",
    "ModelRegistry",
    "ModelSpec",
    "SceneEmbeddingEngine",
    "builtin_profiles",
    "catalog",
    "config_from_record",
    "configured_model_spec",
    "detect_available_providers",
    "get_engine",
    "model_spec",
    "model_registry",
    "reset_engine",
    "resolve_face_execution",
    "resolve_execution",
    "validate_embedding",
    "validate_embeddings",
]
