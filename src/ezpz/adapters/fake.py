"""Deterministic, network-free adapter — the backbone of fast tests and the M1 vertical slice.
It lets the whole engine (cache, scorers, store, matrix) run with no provider, SDK, or cost."""
from __future__ import annotations

from typing import Any

from ezpz.adapters.base import Capabilities, Pipeline
from ezpz.adapters.registry import register
from ezpz.core.document import Document
from ezpz.core.result import Cost, FieldValue, Provenance
from ezpz.core.task import Task


@register("fake")
class FakePipeline(Pipeline):
    """Echoes pre-seeded field values. Seed predictions globally via
    ``config={"fake_fields": {name: value}}`` or per document via
    ``config={"by_slug": {slug: {name: value}}}``. Unseeded fields fall back to None.
    """

    capabilities = Capabilities(confidence=True, provenance=False, is_async=False)

    def __init__(self, config):
        super().__init__(config)
        self.invocations = 0  # spied on by tests to prove cache hits skip invoke()

    def compile(self, task: Task) -> Any:
        return [field.name for field in task.fields]

    def ingest(self, document: Document) -> Any:
        return document

    def invoke(self, prepared: Any, ingested: Any) -> tuple[Any, Cost]:
        self.invocations += 1
        by_slug = self.config.config.get("by_slug", {})
        canned = by_slug.get(
            getattr(ingested, "slug", None), self.config.config.get("fake_fields", {})
        )
        raw = {name: canned.get(name) for name in prepared}
        if "_confidence" in canned:  # let a fake tier simulate a confidence signal (for cascades)
            raw["__confidence__"] = canned["_confidence"]
        usd = self.config.config.get("cost_usd", 0.0)  # let a fake tier simulate a per-call cost
        cost = Cost(input_tokens=0, output_tokens=0, usd=usd, raw={"fake": True})
        return raw, cost

    def map(self, raw: Any, task: Task) -> dict[str, FieldValue]:
        # Structural map only; value-normalization happens centrally, later, in the engine.
        confidence = raw.get("__confidence__", 1.0)
        # Optionally emit text-span provenance (config emit_provenance) so the viewer's source
        # tracer is demonstrable with no real provenance-emitting tool. text_span = the raw value,
        # which (for correct predictions) appears verbatim in the source document.
        emit_prov = self.config.config.get("emit_provenance", False)

        def prov(value: Any):
            if not emit_prov or value in (None, ""):
                return None
            return Provenance(page=1, text_span=str(value))

        return {
            name: FieldValue(value=value, confidence=confidence, provenance=prov(value))
            for name, value in raw.items()
            if name != "__confidence__"
        }
