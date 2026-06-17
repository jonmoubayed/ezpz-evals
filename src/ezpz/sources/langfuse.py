"""``langfuse`` — documents drawn from a Langfuse dataset.

Each Langfuse dataset item becomes one document: its ``input`` is materialized as the document's
bytes and its ``expected_output`` (when present) becomes ground truth, so a Langfuse-curated cohort
scores exactly like a local one. Unlabeled items simply carry no ground truth (the "not-labeled"
state), never a fabricated one.

Auth uses the Langfuse SDK's standard environment variables
(``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST``); ``config.dataset`` selects
the dataset name (defaults to the experiment's dataset name).

Install: ``pip install -e ".[langfuse]"``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ezpz.core.document import ABSENT, Dataset, DatasetSpec, Document
from ezpz.sources.base import DocumentSource
from ezpz.sources.registry import register


def _expected_to_fields(expected: Any) -> dict[str, Any] | None:
    """Coerce an item's ``expected_output`` into a GroundTruth ``fields`` mapping.

    Accepts either the canonical ``{"fields": {...}}`` envelope or a bare ``{field: value}`` map.
    Anything else (a scalar / list) can't be addressed by field name, so the item is left unlabeled.
    """
    if expected is None:
        return None
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except json.JSONDecodeError:
            return None
    if isinstance(expected, dict):
        fields = expected.get("fields") if "fields" in expected else expected
        if isinstance(fields, dict):
            return {k: (ABSENT if v is None else v) for k, v in fields.items()}
    return None


def _input_bytes(value: Any) -> tuple[bytes, str]:
    """Render an item's ``input`` to bytes + a file suffix (text stays text; structured -> json)."""
    if isinstance(value, str):
        return value.encode("utf-8"), ".txt"
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"), ".json"


@register("langfuse")
class LangfuseSource(DocumentSource):
    name = "langfuse"

    def _client(self):  # type: ignore[no-untyped-def]
        try:
            from langfuse import get_client  # lazy: keep registry usable without the extra
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "the 'langfuse' source needs the langfuse SDK — install with: pip install ezpz-evals[langfuse]"
            ) from e
        return get_client()

    def load(self, spec: DatasetSpec, *, root: str, cache_dir: Path) -> Dataset:
        dataset_name = self.config.get("dataset", spec.name)
        client = self._client()
        remote = client.get_dataset(dataset_name)

        staged = cache_dir / spec.name
        gt_dir = staged / "ground_truth"
        documents: list[Document] = []
        for i, item in enumerate(remote.items):
            slug = getattr(item, "id", None) or f"item-{i}"
            data, suffix = _input_bytes(getattr(item, "input", None))
            doc_path = self._cache_path(staged / "docs", slug, data, suffix)

            gt_rel: str | None = None
            fields = _expected_to_fields(getattr(item, "expected_output", None))
            if fields is not None:
                gt_dir.mkdir(parents=True, exist_ok=True)
                gt_file = gt_dir / f"{slug}.json"
                gt_file.write_text(json.dumps({"fields": fields}, ensure_ascii=False, indent=2))
                gt_rel = str(gt_file.relative_to(staged))

            documents.append(
                Document(
                    doc_id=hashlib.sha256(data).hexdigest(),
                    slug=slug,
                    path=str(doc_path.relative_to(staged)),
                    mime="text/plain" if suffix == ".txt" else "application/json",
                    tags=list((getattr(item, "metadata", None) or {}).get("tags", [])),
                    ground_truth_path=gt_rel,
                    source_path=str(doc_path.resolve()),
                )
            )
        return Dataset(name=spec.name, version=spec.version, root=str(staged), documents=documents)
