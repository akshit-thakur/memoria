"""Built-in semantic image/text engines and the active-engine manager."""

from __future__ import annotations

import atexit
import gc
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from typing import Any, Sequence

from .base import ModelSpec, SceneEmbeddingEngine
from .providers import ExecutionPlan, resolve_execution
from .registry import model_registry


logger = logging.getLogger("memoria.ml.scene")

OPENCLIP_VIT_L14_KEY = "scene/open_clip/ViT-L-14/laion2b_s32b_b82k"
OPENCLIP_SIGLIP_KEY = "scene/open_clip/ViT-SO400M-14-SigLIP-384/webli"
HF_SIGLIP2_KEY = "scene/hf/google/siglip2-so400m-patch14-384"
DEFAULT_SCENE_MODEL_KEY = OPENCLIP_VIT_L14_KEY


BUILTIN_SCENE_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_key=OPENCLIP_VIT_L14_KEY,
        task="scene",
        driver="open_clip",
        model_id="ViT-L-14",
        weights="laion2b_s32b_b82k",
        input_resolution=(224, 224),
        embedding_dim=768,
        default_precision="auto",
        provider_preferences=("cuda", "mps", "cpu"),
        metadata={"family": "clip", "supports_text": True, "profile": "openclip-vit-l-14"},
    ),
    ModelSpec(
        model_key=OPENCLIP_SIGLIP_KEY,
        task="scene",
        driver="open_clip",
        model_id="ViT-SO400M-14-SigLIP-384",
        weights="webli",
        input_resolution=(384, 384),
        embedding_dim=1152,
        default_precision="auto",
        provider_preferences=("cuda", "mps", "cpu"),
        metadata={
            "family": "siglip",
            "supports_text": True,
            "profile": "openclip-siglip-so400m-384",
        },
    ),
    ModelSpec(
        model_key=HF_SIGLIP2_KEY,
        task="scene",
        driver="hf",
        model_id="google/siglip2-so400m-patch14-384",
        input_resolution=(384, 384),
        embedding_dim=1152,
        default_precision="auto",
        provider_preferences=("cuda", "mps", "cpu"),
        metadata={
            "family": "siglip2",
            "supports_text": True,
            "profile": "hf-siglip2-so400m-384",
        },
    ),
)

_BUILTIN_ALIASES: dict[str, tuple[str, ...]] = {
    OPENCLIP_VIT_L14_KEY: (
        "default",
        "openclip-vit-l-14",
        "ViT-L-14",
        "ViT-L-14/laion2b_s32b_b82k",
    ),
    OPENCLIP_SIGLIP_KEY: (
        "openclip-siglip-so400m-384",
        "ViT-SO400M-14-SigLIP-384",
        "ViT-SO400M-14-SigLIP-384/webli",
    ),
    HF_SIGLIP2_KEY: (
        "siglip2-so400m-384",
        "hf-siglip2-so400m-384",
        "google/siglip2-so400m-patch14-384",
    ),
}


def _torch_dtype(torch, precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _clear_framework_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()


def _is_accelerator_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: memory allocation" in text


def _normalise_features(torch, features):
    if not torch.is_tensor(features):
        if getattr(features, "pooler_output", None) is not None:
            features = features.pooler_output
        elif getattr(features, "last_hidden_state", None) is not None:
            features = features.last_hidden_state.mean(dim=1)
        elif isinstance(features, (tuple, list)) and features:
            features = features[0]
        else:
            raise TypeError(f"model returned unsupported feature type: {type(features).__name__}")
    if features.ndim == 1:
        features = features.unsqueeze(0)
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-9)


class _LazyTorchSceneEngine(SceneEmbeddingEngine):
    """Common lifecycle and output handling for the two PyTorch drivers."""

    def __init__(self, spec: ModelSpec, execution: ExecutionPlan) -> None:
        super().__init__(spec, execution)
        self._state: dict[str, Any] | None = None
        # Re-entrant because load-time OOM recovery switches execution plans
        # while _ensure_loaded already owns this lock.
        self._load_lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._state is not None

    def _ensure_loaded(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError(f"model engine {self.spec.model_key!r} is closed")
        if self._state is None:
            with self._load_lock:
                if self._state is None:
                    logger.info(
                        "Loading %s with %s on %s (%s)",
                        self.spec.model_key,
                        self.spec.driver,
                        self.execution.device,
                        self.execution.actual_precision,
                    )
                    try:
                        self._state = self._load_state()
                    except RuntimeError as exc:
                        if self.execution.device == "cpu" or not _is_accelerator_oom(exc):
                            raise
                        # Drop traceback frames that may still retain a partly
                        # constructed accelerator model before emptying its allocator.
                        exc.__traceback__ = None
                        self._switch_to_cpu("model loading")
                        self._state = self._load_state()
        return self._state

    def _switch_to_cpu(self, operation: str) -> bool:
        """Release accelerator state and select portable CPU/fp32 execution."""

        if self.execution.device == "cpu":
            return False
        with self._load_lock:
            previous_provider = self.execution.actual_provider
            self._state = None
            self.execution = resolve_execution(
                self.spec,
                requested_provider="cpu",
                requested_precision="fp32",
                supported_providers=("cuda", "mps", "cpu"),
            )
        _clear_framework_memory()
        logger.warning(
            "%s exhausted %s memory during %s; retrying on CPU",
            self.spec.model_key,
            previous_provider,
            operation,
        )
        return True

    def _load_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def _vectors(self, features, expected_count: int) -> list[list[float]]:
        import torch

        features = _normalise_features(torch, features)
        values = features.detach().to(device="cpu", dtype=torch.float32).tolist()
        return self.validate_batch(values, expected_count)

    def close(self) -> None:
        # Acquiring the inference lock first ensures no request is still using
        # state when its references are dropped.
        with self._inference_lock:
            with self._load_lock:
                self._closed = True
                self._state = None
        _clear_framework_memory()


class OpenClipSceneEngine(_LazyTorchSceneEngine):
    """OpenCLIP driver, including OpenCLIP's SigLIP profiles."""

    def _load_state(self) -> dict[str, Any]:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.spec.model_id,
            pretrained=self.spec.weights,
            force_image_size=self.spec.input_resolution,
        )
        dtype = _torch_dtype(torch, self.execution.actual_precision)
        model = model.to(device=self.execution.device, dtype=dtype).eval()
        tokenizer = open_clip.get_tokenizer(self.spec.model_id)

        reported_dim = getattr(getattr(model, "visual", None), "output_dim", None)
        if reported_dim is None:
            projection = getattr(model, "text_projection", None)
            if projection is not None and getattr(projection, "shape", None):
                reported_dim = int(projection.shape[-1])
        if reported_dim is not None and int(reported_dim) != self.spec.embedding_dim:
            raise RuntimeError(
                f"{self.spec.model_key} reports dimension {reported_dim}; "
                f"registry declares {self.spec.embedding_dim}"
            )
        return {
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "torch": torch,
            "dtype": dtype,
        }

    @staticmethod
    def _prepare_one(preprocess, image_or_path):
        from PIL import Image

        owned = False
        if isinstance(image_or_path, (str, bytes, os.PathLike)):
            image = Image.open(image_or_path)
            owned = True
        else:
            image = image_or_path
        try:
            if not hasattr(image, "mode"):
                raise TypeError("expected a PIL image or filesystem path")
            rgb = image if image.mode == "RGB" else image.convert("RGB")
            try:
                return preprocess(rgb)
            finally:
                if rgb is not image:
                    rgb.close()
        finally:
            if owned:
                image.close()

    def encode_image(self, image_or_path: Any) -> list[float]:
        return self.encode_images([image_or_path])[0]

    def encode_images(self, images: Sequence[Any]) -> list[list[float]]:
        if not images:
            return []
        with self.inference_guard():
            state = self._ensure_loaded()
            workers = min(len(images), 8)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                tensors = list(
                    pool.map(lambda image: self._prepare_one(state["preprocess"], image), images)
                )
            batch = state["torch"].stack(tensors).to(
                device=self.execution.device,
                dtype=state["dtype"],
                non_blocking=self.execution.device == "cuda",
            )
            try:
                with state["torch"].inference_mode():
                    features = state["model"].encode_image(batch)
            except RuntimeError as exc:
                if not _is_accelerator_oom(exc):
                    raise
                del batch, tensors
                exc.__traceback__ = None
                if len(images) <= 1:
                    del state
                    if self._switch_to_cpu("image inference"):
                        return self.encode_images(images)
                    raise
                logger.warning(
                    "Scene batch of %d exhausted accelerator memory; splitting",
                    len(images),
                )
                _clear_framework_memory()
                midpoint = len(images) // 2
                return self.encode_images(images[:midpoint]) + self.encode_images(
                    images[midpoint:]
                )
            return self._vectors(features, len(images))

    def encode_text(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text query must be a string")
        with self.inference_guard():
            state = self._ensure_loaded()
            tokens = state["tokenizer"]([text]).to(self.execution.device)
            try:
                with state["torch"].inference_mode():
                    features = state["model"].encode_text(tokens)
            except RuntimeError as exc:
                if not _is_accelerator_oom(exc):
                    raise
                del tokens, state
                exc.__traceback__ = None
                if self._switch_to_cpu("text inference"):
                    return self.encode_text(text)
                raise
            return self._vectors(features, 1)[0]


class HuggingFaceSiglipSceneEngine(_LazyTorchSceneEngine):
    """Transformers driver for Google's SigLIP/SigLIP2 checkpoints."""

    def _load_state(self) -> dict[str, Any]:
        import torch
        from transformers import AutoModel, AutoProcessor

        kwargs = {"revision": self.spec.revision} if self.spec.revision else {}
        processor = AutoProcessor.from_pretrained(self.spec.model_id, **kwargs)
        model = AutoModel.from_pretrained(self.spec.model_id, **kwargs)
        dtype = _torch_dtype(torch, self.execution.actual_precision)
        model = model.to(device=self.execution.device, dtype=dtype).eval()
        reported_dim = getattr(model.config, "projection_dim", None)
        if reported_dim is None:
            text_config = getattr(model.config, "text_config", None)
            reported_dim = getattr(text_config, "projection_size", None)
        if reported_dim is not None and int(reported_dim) != self.spec.embedding_dim:
            raise RuntimeError(
                f"{self.spec.model_key} reports dimension {reported_dim}; "
                f"registry declares {self.spec.embedding_dim}"
            )
        return {"model": model, "processor": processor, "torch": torch, "dtype": dtype}

    def _move_inputs(self, state: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        torch = state["torch"]
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                moved[key] = value
            elif value.is_floating_point():
                moved[key] = value.to(
                    device=self.execution.device,
                    dtype=state["dtype"],
                    non_blocking=self.execution.device == "cuda",
                )
            else:
                moved[key] = value.to(self.execution.device)
        return moved

    @staticmethod
    def _coerce_images(images: Sequence[Any]):
        from PIL import Image

        result = []
        owned = []
        try:
            for item in images:
                item_owned = isinstance(item, (str, bytes, os.PathLike))
                image = Image.open(item) if item_owned else item
                if not hasattr(image, "mode"):
                    raise TypeError("expected PIL images or filesystem paths")
                if image.mode != "RGB":
                    rgb = image.convert("RGB")
                    if item_owned:
                        image.close()
                    image = rgb
                    item_owned = True
                if item_owned:
                    owned.append(image)
                result.append(image)
        except Exception:
            for opened in owned:
                opened.close()
            raise
        return result, owned

    def encode_image(self, image_or_path: Any) -> list[float]:
        return self.encode_images([image_or_path])[0]

    def encode_images(self, images: Sequence[Any]) -> list[list[float]]:
        if not images:
            return []
        with self.inference_guard():
            state = self._ensure_loaded()
            coerced, owned = self._coerce_images(images)
            try:
                inputs = state["processor"](
                    images=coerced,
                    return_tensors="pt",
                )
                inputs = self._move_inputs(state, dict(inputs))
                try:
                    with state["torch"].inference_mode():
                        features = state["model"].get_image_features(**inputs)
                except RuntimeError as exc:
                    if not _is_accelerator_oom(exc):
                        raise
                    del inputs
                    for image in owned:
                        image.close()
                    owned = []
                    exc.__traceback__ = None
                    if len(images) <= 1:
                        del state
                        if self._switch_to_cpu("image inference"):
                            return self.encode_images(images)
                        raise
                    logger.warning(
                        "Scene batch of %d exhausted accelerator memory; splitting",
                        len(images),
                    )
                    _clear_framework_memory()
                    midpoint = len(images) // 2
                    return self.encode_images(images[:midpoint]) + self.encode_images(
                        images[midpoint:]
                    )
                return self._vectors(features, len(images))
            finally:
                for image in owned:
                    image.close()

    def encode_text(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text query must be a string")
        with self.inference_guard():
            state = self._ensure_loaded()
            inputs = state["processor"](
                text=[text],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = self._move_inputs(state, dict(inputs))
            try:
                with state["torch"].inference_mode():
                    features = state["model"].get_text_features(**inputs)
            except RuntimeError as exc:
                if not _is_accelerator_oom(exc):
                    raise
                del inputs, state
                exc.__traceback__ = None
                if self._switch_to_cpu("text inference"):
                    return self.encode_text(text)
                raise
            return self._vectors(features, 1)[0]


for _spec in BUILTIN_SCENE_SPECS:
    _factory = (
        OpenClipSceneEngine
        if _spec.driver == "open_clip"
        else HuggingFaceSiglipSceneEngine
    )
    model_registry.register(_spec, _factory, aliases=_BUILTIN_ALIASES[_spec.model_key])


def _parse_resolution(raw: str | None, default: int) -> tuple[int, int]:
    if not raw:
        return (default, default)
    parts = raw.lower().replace("x", " ").split()
    try:
        if len(parts) == 1:
            value = int(parts[0])
            return (value, value)
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except ValueError:
        pass
    raise ValueError(f"invalid MEMORIA_SCENE_INPUT_RESOLUTION: {raw!r}")


def _normalise_driver(raw: str | None) -> str:
    driver = (raw or "open_clip").strip().lower().replace("-", "_")
    aliases = {"openclip": "open_clip", "huggingface": "hf", "transformers": "hf"}
    driver = aliases.get(driver, driver)
    if driver not in {"open_clip", "hf"}:
        raise ValueError(f"unsupported MEMORIA_SCENE_DRIVER: {raw!r}")
    return driver


def _register_custom_spec(
    *,
    driver: str,
    model_id: str,
    weights: str | None,
    revision: str | None,
    dimension: int,
    resolution: tuple[int, int],
) -> ModelSpec:
    shape_suffix = f"@{dimension}d-{resolution[0]}x{resolution[1]}"
    if driver == "open_clip":
        model_key = f"scene/open_clip/{model_id}/{weights or 'untrained'}{shape_suffix}"
        if revision:
            model_key += f"@{revision}"
        factory = OpenClipSceneEngine
    else:
        model_key = f"scene/hf/{model_id}"
        if revision:
            model_key += f"@{revision}"
        model_key += shape_suffix
        factory = HuggingFaceSiglipSceneEngine
    spec = ModelSpec(
        model_key=model_key,
        task="scene",
        driver=driver,
        model_id=model_id,
        weights=weights,
        revision=revision,
        input_resolution=resolution,
        embedding_dim=dimension,
        default_precision="auto",
        provider_preferences=("cuda", "mps", "cpu"),
        index_kind="exact",
        metadata={"family": "custom", "supports_text": True, "custom": True},
    )
    return model_registry.register(spec, factory)


def _record_value(record: Mapping[str, Any], *names: str, default=None):
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    return default


def _record_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def config_from_record(record: Mapping[str, Any]) -> ModelSpec:
    """Normalize a model-store record into an immutable :class:`ModelSpec`."""

    if not isinstance(record, Mapping) and not hasattr(record, "get"):
        raise TypeError("model record must be a mapping")
    model_key = _record_value(record, "model_key", "key")
    if model_key:
        try:
            return model_registry.resolve(str(model_key))
        except KeyError:
            pass

    task = str(_record_value(record, "kind", "task", default="scene"))
    if task != "scene":
        raise ValueError(f"expected a scene model record, got kind={task!r}")
    driver = _normalise_driver(str(_record_value(record, "driver", default="open_clip")))
    model_id = _record_value(record, "model_name", "model_id", "name")
    if not model_id:
        raise ValueError("model record is missing model_name")
    model_id = str(model_id)
    weights = _record_value(record, "weights", "pretrained")
    weights = str(weights) if weights is not None else None
    revision = _record_value(record, "revision")
    revision = str(revision) if revision is not None else None

    dimension = _record_value(
        record, "embedding_dim", "embedding_dimension", "output_dimension", "dimension"
    )
    if dimension is None:
        raise ValueError("model record is missing embedding_dim")
    dimension = int(dimension)

    width = _record_value(record, "input_width")
    height = _record_value(record, "input_height")
    size = _record_value(record, "input_resolution", "input_size")
    if width is not None or height is not None:
        width = int(width if width is not None else height)
        height = int(height if height is not None else width)
        resolution = (width, height)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        resolution = (int(size[0]), int(size[1]))
    elif size is not None:
        resolution = _parse_resolution(str(size), 224)
    else:
        resolution = (384, 384) if "384" in model_id else (224, 224)

    providers = _record_value(record, "preferred_providers", "provider_preferences")
    if isinstance(providers, str):
        if providers.lstrip().startswith("["):
            providers = tuple(str(value) for value in json.loads(providers))
        else:
            providers = tuple(value.strip() for value in providers.split(",") if value.strip())
    elif providers is not None:
        providers = tuple(str(value) for value in providers)
    else:
        providers = ("cuda", "mps", "cpu")

    metadata = _record_value(record, "metadata", default={})
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, Mapping):
        raise ValueError("model record metadata must be an object")

    if not model_key:
        if driver == "open_clip":
            model_key = (
                f"scene/open_clip/{model_id}/{weights or 'untrained'}"
                f"@{dimension}d-{resolution[0]}x{resolution[1]}"
            )
        else:
            model_key = f"scene/hf/{model_id}" + (f"@{revision}" if revision else "")
            model_key += f"@{dimension}d-{resolution[0]}x{resolution[1]}"
    spec = ModelSpec(
        model_key=str(model_key),
        task="scene",
        driver=driver,
        model_id=model_id,
        weights=weights,
        revision=revision,
        input_resolution=resolution,
        embedding_dim=dimension,
        default_precision=str(_record_value(record, "default_precision", default="auto")),
        provider_preferences=providers,
        normalized=_record_bool(record.get("normalized"), True),
        distance_metric=str(_record_value(record, "distance_metric", default="cosine")),
        index_kind=str(_record_value(record, "index_kind", default="exact")),
        metadata=dict(metadata),
    )
    factory = OpenClipSceneEngine if driver == "open_clip" else HuggingFaceSiglipSceneEngine
    return model_registry.register(spec, factory)


def resolve_configured_scene_spec() -> ModelSpec:
    """Resolve new scene settings, then the legacy MEMORIA_CLIP_MODEL setting."""

    selector = os.getenv("MEMORIA_SCENE_MODEL") or os.getenv("MEMORIA_SCENE_PROFILE")
    explicit_driver = os.getenv("MEMORIA_SCENE_DRIVER")
    weights_override = os.getenv("MEMORIA_SCENE_WEIGHTS")
    revision = os.getenv("MEMORIA_SCENE_REVISION") or None

    if selector and not any((
        explicit_driver,
        weights_override,
        revision,
        os.getenv("MEMORIA_SCENE_DIM"),
        os.getenv("MEMORIA_SCENE_INPUT_RESOLUTION"),
    )):
        try:
            return model_registry.resolve(selector)
        except KeyError:
            pass

    if not selector and not explicit_driver and not weights_override and not revision:
        legacy = os.getenv("MEMORIA_CLIP_MODEL", "ViT-L-14/laion2b_s32b_b82k").strip()
        try:
            return model_registry.resolve(legacy)
        except KeyError:
            if "/" in legacy:
                model_id, weights = legacy.split("/", 1)
            else:
                model_id, weights = legacy, "laion2b_s32b_b82k"
            dimension = int(os.getenv("MEMORIA_SCENE_DIM", "768"))
            resolution = _parse_resolution(os.getenv("MEMORIA_SCENE_INPUT_RESOLUTION"), 224)
            return _register_custom_spec(
                driver="open_clip",
                model_id=model_id,
                weights=weights,
                revision=None,
                dimension=dimension,
                resolution=resolution,
            )

    if not selector:
        driver = _normalise_driver(explicit_driver)
        selector = "google/siglip2-so400m-patch14-384" if driver == "hf" else "ViT-L-14"
    else:
        try:
            selected_profile = model_registry.resolve(selector)
        except KeyError:
            selected_profile = None
        driver = (
            _normalise_driver(explicit_driver)
            if explicit_driver
            else selected_profile.driver if selected_profile is not None else "open_clip"
        )

    try:
        selected_profile = model_registry.resolve(selector)
    except KeyError:
        selected_profile = None

    # A built-in profile remains self-describing unless an explicit override
    # asks for a custom identity.
    if selected_profile is not None and selected_profile.driver == driver:
        if not any((
            weights_override,
            revision,
            os.getenv("MEMORIA_SCENE_DIM"),
            os.getenv("MEMORIA_SCENE_INPUT_RESOLUTION"),
        )):
            return selected_profile
        dimension = int(
            os.getenv("MEMORIA_SCENE_DIM", str(selected_profile.embedding_dim))
        )
        resolution = (
            _parse_resolution(
                os.getenv("MEMORIA_SCENE_INPUT_RESOLUTION"),
                selected_profile.input_resolution[0],
            )
            if os.getenv("MEMORIA_SCENE_INPUT_RESOLUTION")
            else selected_profile.input_resolution
        )
        return _register_custom_spec(
            driver=driver,
            model_id=selected_profile.model_id,
            weights=weights_override or selected_profile.weights,
            revision=revision or selected_profile.revision,
            dimension=dimension,
            resolution=resolution,
        )

    if driver == "open_clip":
        model_id = selector
        weights = weights_override
        if weights is None and "/" in selector:
            model_id, weights = selector.split("/", 1)
        weights = weights or "laion2b_s32b_b82k"
    else:
        model_id = selector
        weights = None

    raw_dim = os.getenv("MEMORIA_SCENE_DIM")
    if raw_dim is None:
        raise ValueError(
            "MEMORIA_SCENE_DIM is required for a scene model that is not a built-in profile"
        )
    raw_resolution = os.getenv("MEMORIA_SCENE_INPUT_RESOLUTION")
    if raw_resolution is None:
        raise ValueError(
            "MEMORIA_SCENE_INPUT_RESOLUTION is required for a scene model that is "
            "not a built-in profile"
        )
    dimension = int(raw_dim)
    default_resolution = 384 if "384" in model_id else 224
    resolution = _parse_resolution(raw_resolution, default_resolution)
    return _register_custom_spec(
        driver=driver,
        model_id=model_id,
        weights=weights,
        revision=revision,
        dimension=dimension,
        resolution=resolution,
    )


_manager_lock = threading.RLock()
_active_engine: SceneEmbeddingEngine | None = None
_active_requested_execution: ExecutionPlan | None = None


def configured_model_spec() -> ModelSpec:
    """Return the environment-selected model description without loading it."""

    return resolve_configured_scene_spec()


def model_spec(config: ModelSpec | Mapping[str, Any] | str | None = None) -> ModelSpec:
    """Resolve a spec, registry key, or persisted model record."""

    if config is None:
        return configured_model_spec()
    if isinstance(config, ModelSpec):
        return config
    if isinstance(config, str):
        return model_registry.resolve(config)
    return config_from_record(config)


def get_engine(
    config: ModelSpec | Mapping[str, Any] | str | None = None,
    *,
    device: str | None = None,
) -> SceneEmbeddingEngine:
    """Return the one active lazy engine, optionally for an explicit spec."""

    global _active_engine, _active_requested_execution
    spec = model_spec(config)
    execution = resolve_execution(
        spec,
        requested_provider=device,
        supported_providers=("cuda", "mps", "cpu"),
    )
    with _manager_lock:
        if _active_engine is not None and (
            _active_engine.spec.fingerprint != spec.fingerprint
            or _active_requested_execution != execution
        ):
            logger.info(
                "Scene model changed from %s to %s; releasing old engine",
                _active_engine.spec.model_key,
                spec.model_key,
            )
            _active_engine.close()
            _active_engine = None
            _active_requested_execution = None
        if _active_engine is None:
            engine = model_registry.create(spec.model_key, execution)
            if not isinstance(engine, SceneEmbeddingEngine):
                raise TypeError(f"factory for {spec.model_key!r} did not return a scene engine")
            _active_engine = engine
            _active_requested_execution = execution
        return _active_engine


def get_scene_engine() -> SceneEmbeddingEngine:
    """Backward-compatible name for the environment-configured engine."""

    return get_engine()


def reset_scene_engine() -> None:
    """Close and forget the active engine, releasing framework caches."""

    global _active_engine, _active_requested_execution
    with _manager_lock:
        engine, _active_engine = _active_engine, None
        _active_requested_execution = None
        if engine is not None:
            engine.close()


def reset_engine() -> None:
    reset_scene_engine()


def scene_runtime_info() -> dict[str, Any]:
    return get_scene_engine().runtime_info


def list_scene_models() -> list[dict[str, Any]]:
    return [spec.as_dict() for spec in model_registry.list_specs(task="scene")]


def catalog() -> list[dict[str, Any]]:
    return list_scene_models()


def builtin_profiles() -> tuple[ModelSpec, ...]:
    return BUILTIN_SCENE_SPECS


atexit.register(reset_scene_engine)


__all__ = [
    "BUILTIN_SCENE_SPECS",
    "DEFAULT_SCENE_MODEL_KEY",
    "HF_SIGLIP2_KEY",
    "HuggingFaceSiglipSceneEngine",
    "OPENCLIP_SIGLIP_KEY",
    "OPENCLIP_VIT_L14_KEY",
    "OpenClipSceneEngine",
    "builtin_profiles",
    "catalog",
    "config_from_record",
    "configured_model_spec",
    "get_engine",
    "get_scene_engine",
    "list_scene_models",
    "model_spec",
    "reset_engine",
    "reset_scene_engine",
    "resolve_configured_scene_spec",
    "scene_runtime_info",
]
