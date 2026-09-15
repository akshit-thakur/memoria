"""Thread-safe registry of model specifications and engine factories."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any

from .base import BaseModelEngine, ModelSpec


EngineFactory = Callable[[ModelSpec, Any], BaseModelEngine]


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._factories: dict[str, EngineFactory] = {}
        self._aliases: dict[str, str] = {}
        self._lock = threading.RLock()

    def register(
        self,
        spec: ModelSpec,
        factory: EngineFactory | None = None,
        aliases: Iterable[str] = (),
    ) -> ModelSpec:
        """Register idempotently, rejecting identity collisions."""

        with self._lock:
            existing = self._specs.get(spec.model_key)
            if existing is not None and existing.fingerprint != spec.fingerprint:
                raise ValueError(f"model_key collision: {spec.model_key!r}")
            self._specs[spec.model_key] = spec
            if factory is not None:
                old_factory = self._factories.get(spec.model_key)
                if old_factory is not None and old_factory is not factory:
                    raise ValueError(f"factory collision: {spec.model_key!r}")
                self._factories[spec.model_key] = factory

            for alias in aliases:
                alias = str(alias).strip()
                if not alias:
                    continue
                target = self._aliases.get(alias)
                if target is not None and target != spec.model_key:
                    raise ValueError(f"model alias collision: {alias!r}")
                self._aliases[alias] = spec.model_key
        return spec

    def resolve(self, key_or_alias: str) -> ModelSpec:
        key_or_alias = str(key_or_alias).strip()
        with self._lock:
            key = self._aliases.get(key_or_alias, key_or_alias)
            try:
                return self._specs[key]
            except KeyError as exc:
                raise KeyError(f"unknown model: {key_or_alias!r}") from exc

    def create(self, key_or_alias: str, execution: Any) -> BaseModelEngine:
        spec = self.resolve(key_or_alias)
        with self._lock:
            factory = self._factories.get(spec.model_key)
        if factory is None:
            raise KeyError(f"no engine factory registered for {spec.model_key!r}")
        return factory(spec, execution)

    def list_specs(self, task: str | None = None) -> tuple[ModelSpec, ...]:
        with self._lock:
            specs = tuple(self._specs.values())
        if task is not None:
            specs = tuple(spec for spec in specs if spec.task == task)
        return tuple(sorted(specs, key=lambda spec: spec.model_key))


model_registry = ModelRegistry()

