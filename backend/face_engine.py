"""Lazy, environment-configured face-analysis engine facade.

Workers retain the historical :func:`get_detector` call while Phase 2 callers
can use :func:`get_face_engine`, inspect its model descriptor, and explicitly
reset or close heavyweight model state.
"""

from __future__ import annotations

import atexit
import logging
import threading
from typing import Any

from ml.base import ModelSpec
from ml.face import FaceAnalysisEngine, FaceEngineConfig, resolve_face_execution


__all__ = [
    "close",
    "close_face_engine",
    "get_detector",
    "get_face_engine",
    "model_spec",
    "reset",
    "reset_face_engine",
    "run_clustering",
    "runtime_info",
]

logger = logging.getLogger("memoria.ml.face")

_engine: FaceAnalysisEngine | None = None
_config_fingerprint: str | None = None
_engine_lock = threading.RLock()


def _build_engine(config: FaceEngineConfig) -> FaceAnalysisEngine:
    # Keep metadata inspection framework-free. Importing FaceDetector pulls in
    # OpenCV and NumPy, so defer it until an actual runtime engine is requested.
    from face_detection import FaceDetector

    spec = config.to_model_spec()
    if spec.embedding_dim not in {None, 512}:
        raise RuntimeError(
            "The current face_embeddings schema accepts only 512-dimensional "
            f"vectors; {spec.model_key!r} declares {spec.embedding_dim}"
        )
    execution = resolve_face_execution(config, spec)
    logger.info(
        "Initializing face engine=%s detectors=%s embedder=%s provider=%s "
        "precision=%s",
        config.engine_key,
        config.detectors,
        config.embedder,
        execution.actual_provider,
        execution.actual_precision,
    )
    return FaceDetector(
        hw_mode=execution.device,
        models=list(config.detectors),
        min_consensus=config.min_consensus,
        use_embeddings=config.embedder != "none",
        embedder=config.embedder,
        provider=execution.actual_provider,
        precision=execution.actual_precision,
        engine_key=config.engine_key,
        pipeline_key=config.detection_pipeline_key,
        embedding_model_key=config.embedding_model_key,
        insightface_model=config.insightface_model,
        batch_concurrency=config.batch_concurrency,
        model_spec=spec,
        execution=execution,
    )


def get_face_engine() -> FaceAnalysisEngine:
    """Return the one lazily constructed engine for the current environment.

    Environment changes are detected before inference. The old singleton is
    released before a differently configured engine is created, preventing two
    full face stacks from occupying accelerator memory at once.
    """

    global _engine, _config_fingerprint
    config = FaceEngineConfig.from_env()
    fingerprint = config.fingerprint
    with _engine_lock:
        if _engine is not None and fingerprint != _config_fingerprint:
            logger.info("Face configuration changed; releasing the previous engine")
            _engine.close()
            _engine = None
            _config_fingerprint = None
        if _engine is None:
            _engine = _build_engine(config)
            _config_fingerprint = fingerprint
        return _engine


def get_detector() -> FaceAnalysisEngine:
    """Compatibility alias used by the existing ingestion workers."""

    return get_face_engine()


def model_spec() -> ModelSpec:
    """Describe configured face analysis without constructing an engine."""

    return FaceEngineConfig.from_env().to_model_spec()


def runtime_info() -> dict:
    """Return JSON-safe model, provider, precision, and load-state details."""

    return get_face_engine().runtime_info


def close_face_engine() -> None:
    """Release the singleton and all heavyweight model references."""

    global _engine, _config_fingerprint
    with _engine_lock:
        engine = _engine
        _engine = None
        _config_fingerprint = None
        if engine is not None:
            engine.close()


def reset_face_engine() -> None:
    """Discard the singleton; the next access rebuilds it from environment."""

    close_face_engine()


def close() -> None:
    """Short lifecycle alias for integration code and tests."""

    close_face_engine()


def reset() -> None:
    """Short lifecycle alias for integration code and tests."""

    reset_face_engine()


def run_clustering(*args: Any, **kwargs: Any):
    """Compatibility wrapper that imports clustering only when invoked."""

    from face_detection import run_clustering as _run_clustering

    return _run_clustering(*args, **kwargs)


atexit.register(close_face_engine)
