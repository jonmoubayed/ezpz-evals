"""Plugin discovery — let OTHER packages add adapters/scorers without changing ezpz.

A consuming project registers a custom adapter (or scorer) by:
  1. defining a `Pipeline` subclass decorated with
     ``@ezpz.adapters.registry.register("myapp")``, and
  2. advertising the module via an entry point in its own pyproject.toml:

        [project.entry-points."ezpz.adapters"]
        myapp = "myapp.ezpz_adapter"

On startup ezpz imports each advertised module, so its ``@register`` side effects run and
``adapter: myapp`` becomes usable in an experiment. Custom scorers use the ``ezpz.scorers`` group.

This is purely additive — the built-in adapters/scorers are always available. For an ad-hoc local
module (not yet packaged with an entry point), an experiment can instead list it under ``plugins:``.
"""
from __future__ import annotations

import importlib
import os
import sys
from importlib.metadata import entry_points

_GROUPS = ("ezpz.adapters", "ezpz.scorers", "ezpz.sources")
_loaded = False


def load_plugins(force: bool = False) -> list[str]:
    """Import all modules advertised under the ezpz entry-point groups (idempotent).

    Returns the names loaded. A broken plugin is reported to stderr but does not abort the others.
    """
    global _loaded
    if _loaded and not force:
        return []
    _loaded = True
    loaded: list[str] = []
    for group in _GROUPS:
        for ep in entry_points(group=group):
            try:
                ep.load()  # imports the module -> @register side effects run
                loaded.append(f"{group}:{ep.name}")
            except Exception as e:  # a broken plugin shouldn't break the whole CLI
                print(f"[ezpz] failed to load plugin '{ep.name}' ({group}): {e}", file=sys.stderr)
    return loaded


def import_modules(modules: list[str]) -> None:
    """Import explicit adapter/scorer modules from an experiment's ``plugins:`` list.

    The current working directory is put on sys.path so a loose local module (e.g. a
    ``my_adapter.py`` in the project root) is importable, not only installed packages.
    """
    if not modules:
        return
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    for name in modules:
        importlib.import_module(name)
