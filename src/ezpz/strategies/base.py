"""Composite extraction strategies — meta-adapters that orchestrate INNER pipelines.

A strategy (cascade / ensemble / verify) is itself one `Pipeline` (one comparable strategy under
test). It overrides `extract()` to run its inner pipelines and combine their results; the other
stages are unused except `map()`, which re-derives FieldValues from the cached envelope (so a cache
hit is free). Inner pipelines are specified in config as ``{adapter, config}`` and can be ANY
registered adapter — `fake` for no-key demos, or real ones (Anthropic/Gemini/...) in production.

`extract()` returns a JSON-serializable envelope ``{"fields": {name: {value, confidence}},
"extras": {...}}`` as `raw`; the engine restores `extras` on a cache hit too.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any

from ezpz.adapters.base import Pipeline
from ezpz.adapters.registry import get_adapter
from ezpz.core.document import Document
from ezpz.core.result import Cost, ExtractionResult, FieldValue, ResultStatus, Timing
from ezpz.core.run import PipelineConfig
from ezpz.core.task import Task


class CompositePipeline(Pipeline):
    def _inner(self, spec: dict) -> Pipeline:
        return get_adapter(spec["adapter"])(
            PipelineConfig(adapter=spec["adapter"], config=spec.get("config", {}))
        )

    @staticmethod
    def _usd(result: ExtractionResult) -> float:
        return (result.cost.usd or 0.0) if result.cost else 0.0

    def _build(self, document, task, fields_env: dict, extras: dict, usd: float, t0: float):
        envelope = {"fields": fields_env, "extras": extras}  # 'extras' is restored on a cache hit
        result = ExtractionResult(
            doc_id=document.doc_id, pipeline_id=self.pipeline_id, status=ResultStatus.OK,
            fields=self.map(envelope, task), cost=Cost(usd=round(usd, 6)), extras=extras,
            timing=Timing(latency_ms=(time.perf_counter() - t0) * 1000.0),
        )
        return result, envelope

    # ---- unused stage stubs (extract is the whole flow) ----
    def compile(self, task: Task) -> Any:
        return None

    def ingest(self, document: Document) -> Any:
        return document

    def invoke(self, prepared: Any, ingested: Any) -> tuple[Any, Cost]:
        raise NotImplementedError("composite strategies override extract()")

    def map(self, raw: Any, task: Task) -> dict[str, FieldValue]:
        cells = raw.get("fields", {}) if isinstance(raw, dict) else {}
        return {
            f.name: FieldValue(
                value=(cells.get(f.name) or {}).get("value"),
                confidence=(cells.get(f.name) or {}).get("confidence"),
            )
            for f in task.fields
        }


def confidence_of(result: ExtractionResult) -> float:
    """Scalar confidence for a result = min over fields that emit one (1.0 if none do)."""
    confs = [fv.confidence for fv in result.fields.values() if fv.confidence is not None]
    return min(confs) if confs else 1.0


def cell(value: Any, confidence: Any) -> dict:
    return {"value": value, "confidence": confidence}


def majority(values: list) -> tuple[Any, float]:
    """Most common value (by JSON identity, so dicts/lists are votable) + agreement fraction."""
    if not values:
        return None, 0.0
    keyed = [(json.dumps(v, sort_keys=True, default=str), v) for v in values]
    top_key, n = Counter(k for k, _ in keyed).most_common(1)[0]
    value = next(v for k, v in keyed if k == top_key)
    return value, round(n / len(values), 4)
