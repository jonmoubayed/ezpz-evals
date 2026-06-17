"""Tiny name->DocumentSource registry. Sources register via @register('name').

Deliberately identical in shape to ``ezpz.adapters.registry`` — a document source is the
input-side sibling of an adapter, and the same plugin/entry-point machinery extends both.
"""
from __future__ import annotations

from typing import Callable, Type

_REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        if name in _REGISTRY:
            raise ValueError(f"document source '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return deco


def get_source(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"unknown document source '{name}'. registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
