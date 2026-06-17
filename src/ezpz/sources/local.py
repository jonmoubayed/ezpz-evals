"""``local`` — the default source: a manifest + files on local disk.

This is exactly the original behavior (``<root>/datasets/<name>/manifest.jsonl`` with co-located
``docs/`` and ``ground_truth/``), now expressed as a :class:`DocumentSource`. Remote sources
(``s3``, ``langfuse``, ``extend``) download into a local directory and then reuse this loader, so
manifest parsing and ground-truth wiring live in one place.
"""
from __future__ import annotations

from pathlib import Path

from ezpz.core.document import Dataset, DatasetSpec
from ezpz.sources.base import DocumentSource
from ezpz.sources.registry import register


@register("local")
class LocalSource(DocumentSource):
    name = "local"

    def load(self, spec: DatasetSpec, *, root: str, cache_dir: Path) -> Dataset:
        # `dir` overrides the conventional layout (used by remote sources that staged a download).
        dataset_dir = self.config.get("dir") or str(Path(root) / "datasets" / spec.name)
        return Dataset.load_from_manifest(dataset_dir, spec.name, spec.version)
