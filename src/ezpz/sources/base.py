"""The ``DocumentSource`` contract: where a cohort's documents come from.

A source's single job is to turn a :class:`DatasetSpec` into a :class:`Dataset` of
content-hashed :class:`Document`s whose ``source_path`` points at locally-readable bytes —
so everything downstream (adapters' ``ingest``, the cache key, the store) is identical no
matter whether the bytes originated on local disk, in S3, in a Langfuse dataset, or in Extend.

Contract notes that keep the invariants intact:
  - **Identity is still content-addressed.** A remote source must materialize bytes locally and
    hash them, so ``doc_id`` (and therefore the extraction cache) is stable across sources.
  - **Ground truth keeps its three states.** A source either attaches a ``ground_truth_path``
    (loaded later by ``GroundTruth.load``) or leaves documents unlabeled — it never invents GT.
  - **No network in ``__init__``.** Construct cheaply; do I/O in ``load`` so ``ezpz validate``
    and registry listing work without credentials or the optional SDK installed.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ezpz.core.document import Dataset, DatasetSpec


class DocumentSource(ABC):
    """Base class for document sources. Subclasses implement :meth:`load`."""

    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def load(self, spec: DatasetSpec, *, root: str, cache_dir: Path) -> Dataset:
        """Enumerate (and, for remote sources, download) the cohort.

        ``root`` is the experiment project root (for resolving local paths / relative layout).
        ``cache_dir`` is a writable directory a remote source should materialize bytes into;
        re-materializing the same remote object should be idempotent (content-addressed).
        """
        raise NotImplementedError

    # -- helpers shared by remote sources -------------------------------------------------

    @staticmethod
    def _cache_path(cache_dir: Path, key: str, data: bytes, suffix: str = "") -> Path:
        """Write ``data`` under ``cache_dir`` at a content-addressed path; return it.

        Idempotent: the path is derived from the bytes, so an unchanged remote object maps to the
        same file and is not rewritten. ``suffix`` preserves a file extension for mime guessing.
        """
        digest = hashlib.sha256(data).hexdigest()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{digest}{suffix}"
        if not path.exists():
            path.write_bytes(data)
        return path
