"""name->Scorer registry (mirror of adapters.registry)."""
from __future__ import annotations

from typing import Callable, Type

_REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        if name in _REGISTRY:
            raise ValueError(f"scorer '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return deco


def get_scorer(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"unknown scorer '{name}'. registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
