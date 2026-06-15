"""Self-consistency / ensemble voting: run several members (same model sampled N times, or several
models), then majority-vote each field. Per-field confidence = the agreement fraction, so a unanimous
field is high-confidence and a split field is low — which feeds calibration nicely.

config: { members: [ {adapter, config}, ... ] }
"""
from __future__ import annotations

import time
from typing import Any

from ezpz.adapters.base import Capabilities
from ezpz.adapters.registry import register
from ezpz.core.document import Document
from ezpz.core.result import ExtractionResult
from ezpz.core.task import Task
from ezpz.strategies.base import CompositePipeline, cell, majority


@register("ensemble")
class EnsemblePipeline(CompositePipeline):
    capabilities = Capabilities(confidence=True)

    def extract(self, document: Document, task: Task) -> tuple[ExtractionResult, Any]:
        t0 = time.perf_counter()
        members = self.config.config["members"]
        results, usd = [], 0.0
        for spec in members:
            res, _raw = self._inner(spec).extract(document, task)
            results.append(res)
            usd += self._usd(res)

        fields_env = {}
        min_agreement = 1.0
        for f in task.fields:
            votes = [r.fields[f.name].value for r in results if f.name in r.fields]
            value, agreement = majority(votes)
            fields_env[f.name] = cell(value, agreement)
            min_agreement = min(min_agreement, agreement)
        extras = {"members": len(members), "unanimous": min_agreement == 1.0}
        return self._build(document, task, fields_env, extras, usd, t0)
