"""Cascade / model routing: run a cheap tier first; escalate to the next tier while the tier's
confidence is below `threshold`. Gates on the inner pipeline's EMITTED confidence — pair it with an
adapter that emits one (Extend natively, or an LLM adapter that self-rates). Records which tier was
used + whether it escalated in `extras` (so `ezpz analyze` can report the escalation rate).

config: { threshold: 0.8, tiers: [ {adapter, config}, ... ] }  # cheapest -> most capable
"""
from __future__ import annotations

import time
from typing import Any

from ezpz.adapters.base import Capabilities
from ezpz.adapters.registry import register
from ezpz.core.document import Document
from ezpz.core.result import ExtractionResult, ResultStatus
from ezpz.core.task import Task
from ezpz.strategies.base import CompositePipeline, cell, confidence_of


@register("cascade")
class CascadePipeline(CompositePipeline):
    capabilities = Capabilities(confidence=True)

    def extract(self, document: Document, task: Task) -> tuple[ExtractionResult, Any]:
        t0 = time.perf_counter()
        cfg = self.config.config
        tiers = cfg["tiers"]
        if not tiers:
            raise ValueError("cascade requires at least one tier")
        threshold = float(cfg.get("threshold", 0.8))
        usd, chosen, used, conf = 0.0, None, 0, 0.0
        for i, spec in enumerate(tiers):
            res, _raw = self._inner(spec).extract(document, task)
            usd += self._usd(res)
            chosen, used = res, i
            conf = confidence_of(res) if res.status is ResultStatus.OK else -1.0
            if conf >= threshold:
                break
        assert chosen is not None  # the non-empty check above guarantees a tier ran
        fields_env = {n: cell(fv.value, fv.confidence) for n, fv in chosen.fields.items()}
        extras = {
            "tier_used": used,
            "tier_label": tiers[used].get("config", {}).get("label", str(used)),
            "escalated": used > 0,
            "confidence": conf,
        }
        return self._build(document, task, fields_env, extras, usd, t0)
