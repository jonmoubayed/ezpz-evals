"""Extract-then-verify (critic / self-check): an extractor proposes values, then a verifier
re-extracts the same fields; where they disagree, the verifier's value wins and the field is marked
low-confidence. Records how many fields were corrected in `extras`.

config: { extractor: {adapter, config}, verifier: {adapter, config} }
"""
from __future__ import annotations

import time
from typing import Any

from ezpz.adapters.base import Capabilities
from ezpz.adapters.registry import register
from ezpz.core.document import Document
from ezpz.core.result import ExtractionResult
from ezpz.core.task import Task
from ezpz.strategies.base import CompositePipeline, cell


@register("verify")
class VerifyPipeline(CompositePipeline):
    capabilities = Capabilities(confidence=True)

    def extract(self, document: Document, task: Task) -> tuple[ExtractionResult, Any]:
        t0 = time.perf_counter()
        cfg = self.config.config
        extractor, _ = self._inner(cfg["extractor"]).extract(document, task)
        verifier, _ = self._inner(cfg["verifier"]).extract(document, task)
        usd = self._usd(extractor) + self._usd(verifier)

        fields_env, corrections = {}, 0
        for f in task.fields:
            ev = extractor.fields.get(f.name)
            vv = verifier.fields.get(f.name)
            e_val = ev.value if ev else None
            v_val = vv.value if vv else None
            agree = e_val == v_val
            if not agree:
                corrections += 1
            fields_env[f.name] = cell(v_val if vv is not None else e_val, 1.0 if agree else 0.5)
        extras = {"corrections": corrections, "verified": corrections == 0}
        return self._build(document, task, fields_env, extras, usd, t0)
