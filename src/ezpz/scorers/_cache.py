"""Tiny content-addressed cache for expensive scorer calls (embeddings, llm_judge).

Keyed on a stable hash of inputs + config so re-scoring costs zero extra API calls — the same
"derive on read" guarantee as the extraction cache, for the second (cheap-but-not-free) layer.
Location is ``$EZPZ_HOME/score_cache`` (defaults to ``.ezpz``).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


def _root() -> Path:
    return Path(os.environ.get("EZPZ_HOME", ".ezpz")) / "score_cache"


def cache_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def cached(key: str, compute: Callable[[], dict]) -> dict:
    path = _root() / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value
