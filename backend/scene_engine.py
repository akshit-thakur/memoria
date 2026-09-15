"""Backward-compatible facade for the pluggable scene model registry.

Existing worker and API callers intentionally keep the small function API from
Phase D. The implementation, model selection, validation, and lifecycle now
live in :mod:`ml.scene`.
"""

from __future__ import annotations

from typing import Any, Sequence

from ml.scene import (
    builtin_profiles,
    catalog,
    config_from_record,
    configured_model_spec,
    get_engine,
    get_scene_engine,
    list_scene_models,
    model_spec,
    reset_engine,
    reset_scene_engine,
    scene_runtime_info,
)


def encode_image(pil_or_path: Any) -> list[float]:
    """Encode one image with the configured scene engine."""

    return get_scene_engine().encode_image(pil_or_path)


def encode_images_batch(pil_images: Sequence[Any]) -> list[list[float]]:
    """Encode an image batch, preserving input order."""

    return get_scene_engine().encode_images(pil_images)


def encode_text(query: str) -> list[float]:
    """Encode text in the same normalized vector space as images."""

    return get_scene_engine().encode_text(query)


def embedding_dim() -> int:
    """Return the configured output dimension without loading weights."""

    dimension = configured_model_spec().embedding_dim
    if dimension is None:
        raise RuntimeError("configured scene model does not declare an embedding dimension")
    return dimension


def runtime_info() -> dict:
    """Return selected/actual provider details and lazy-load state."""

    return scene_runtime_info()


def close() -> None:
    """Release the active engine and framework accelerator caches."""

    reset_scene_engine()


__all__ = [
    "close",
    "builtin_profiles",
    "catalog",
    "config_from_record",
    "configured_model_spec",
    "embedding_dim",
    "encode_image",
    "encode_images_batch",
    "encode_text",
    "get_engine",
    "get_scene_engine",
    "list_scene_models",
    "model_spec",
    "reset_engine",
    "runtime_info",
]
