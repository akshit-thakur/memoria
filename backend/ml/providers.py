"""Execution-provider and precision selection for ML drivers.

Selection is capability based: requesting TensorRT cannot make a PyTorch-only
driver use TensorRT, and an unavailable accelerator always falls through to a
supported CPU plan instead of crashing model initialization.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

from .base import ModelSpec


logger = logging.getLogger("memoria.ml.providers")

_PROVIDER_ALIASES = {
    "gpu": "cuda",
    "torch-cuda": "cuda",
    "cudaexecutionprovider": "onnx-cuda",
    "tensorrtexecutionprovider": "tensorrt",
    "cpuexecutionprovider": "cpu",
}
_PRECISION_ALIASES = {
    "float": "fp32",
    "float32": "fp32",
    "half": "fp16",
    "float16": "fp16",
    "bfloat16": "bf16",
}


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    requested_provider: str
    actual_provider: str
    device: str
    requested_precision: str
    actual_precision: str
    available_providers: tuple[str, ...]
    requested_providers: tuple[str, ...] = ()
    fallback_reason: str | None = None

    def as_dict(self) -> dict:
        data = asdict(self)
        requested = self.requested_providers or _provider_list(self.requested_provider)
        data["requested_providers"] = list(requested)
        data["available_providers"] = list(self.available_providers)
        return data


def _canonical_provider(value: str | None) -> str:
    raw = (value or "auto").strip().lower()
    return _PROVIDER_ALIASES.get(raw, raw)


def _provider_list(value: str | None) -> tuple[str, ...]:
    raw_values = str(value or "auto").replace(";", ",").split(",")
    providers = tuple(
        _canonical_provider(raw) for raw in raw_values if str(raw).strip()
    ) or ("auto",)
    allowed = {"auto", "tensorrt", "onnx-cuda", "cuda", "mps", "cpu"}
    unknown = [provider for provider in providers if provider not in allowed]
    if unknown:
        raise ValueError(f"unsupported execution provider: {unknown[0]!r}")
    if "auto" in providers and len(providers) > 1:
        raise ValueError("auto cannot be combined with explicit execution providers")
    return tuple(dict.fromkeys(providers))


def _canonical_precision(value: str | None) -> str:
    raw = (value or "auto").strip().lower()
    return _PRECISION_ALIASES.get(raw, raw)


def detect_available_providers() -> tuple[str, ...]:
    """Return detected providers in descending acceleration order."""

    available: list[str] = []
    try:
        import onnxruntime as ort

        ort_providers = set(ort.get_available_providers())
        if "TensorrtExecutionProvider" in ort_providers:
            available.append("tensorrt")
        if "CUDAExecutionProvider" in ort_providers:
            available.append("onnx-cuda")
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            available.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            available.append("mps")
    except Exception:
        pass

    available.append("cpu")
    return tuple(dict.fromkeys(available))


def resolve_execution(
    spec: ModelSpec,
    requested_provider: str | None = None,
    requested_precision: str | None = None,
    *,
    supported_providers: tuple[str, ...] | None = None,
) -> ExecutionPlan:
    """Resolve requested settings to a provider supported on this runtime.

    Environment defaults are ``MEMORIA_<TASK>_PROVIDER`` / ``_DEVICE`` and
    ``MEMORIA_<TASK>_PRECISION``. For scene models these become the documented
    ``MEMORIA_SCENE_*`` settings.
    """

    prefix = "MEMORIA_" + spec.task.upper().replace("-", "_")
    requested_providers = _provider_list(
        requested_provider
        or os.getenv(f"{prefix}_PROVIDERS")
        or os.getenv(f"{prefix}_PROVIDER")
        or os.getenv(f"{prefix}_DEVICE")
        or "auto"
    )
    requested_provider = ",".join(requested_providers)

    requested_precision = _canonical_precision(
        requested_precision or os.getenv(f"{prefix}_PRECISION") or spec.default_precision
    )
    if requested_precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError(f"unsupported precision: {requested_precision!r}")

    available = detect_available_providers()
    supported = tuple(supported_providers or spec.provider_preferences)
    if requested_providers == ("auto",):
        candidates = list(spec.provider_preferences)
    else:
        candidates = list(requested_providers)
        candidates.extend(p for p in spec.provider_preferences if p not in candidates)
    if "cpu" not in candidates:
        candidates.append("cpu")

    actual = next((p for p in candidates if p in supported and p in available), None)
    if actual is None:
        # A conforming engine must support CPU. Keep this explicit so adding a
        # GPU-only driver fails during registration/testing rather than later.
        if "cpu" not in supported:
            raise RuntimeError(f"model driver {spec.driver!r} has no available provider")
        actual = "cpu"

    fallback_reason = None
    if requested_providers != ("auto",) and actual not in requested_providers:
        fallback_reason = f"requested {requested_provider} is unavailable or unsupported"
        logger.warning(
            "Model %s: %s; falling back to %s",
            spec.model_key,
            fallback_reason,
            actual,
        )

    if actual in {"cuda", "onnx-cuda", "tensorrt"}:
        device = "cuda"
    elif actual == "mps":
        device = "mps"
    else:
        device = "cpu"

    if requested_precision == "auto":
        precision = "fp16" if device == "cuda" else "fp32"
    else:
        precision = requested_precision
    if device == "cpu" and precision in {"fp16", "bf16"}:
        fallback_reason = (fallback_reason + "; " if fallback_reason else "") + (
            f"{precision} CPU inference is not a portable path; using fp32"
        )
        precision = "fp32"
    if device == "mps" and precision == "bf16":
        fallback_reason = (fallback_reason + "; " if fallback_reason else "") + (
            "bf16 MPS inference is unsupported; using fp32"
        )
        precision = "fp32"
    if device == "cuda" and precision == "bf16":
        try:
            import torch

            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False
        if not bf16_supported:
            fallback_reason = (fallback_reason + "; " if fallback_reason else "") + (
                "bf16 is unsupported by this CUDA device; using fp16"
            )
            precision = "fp16"

    return ExecutionPlan(
        requested_provider=requested_provider,
        requested_providers=requested_providers,
        actual_provider=actual,
        device=device,
        requested_precision=requested_precision,
        actual_precision=precision,
        available_providers=available,
        fallback_reason=fallback_reason,
    )
