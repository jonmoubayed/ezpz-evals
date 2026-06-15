"""Cost estimation + budget guard. Prevents the 'oops, $300' moment on a local tool."""
from __future__ import annotations

from typing import Optional

from ezpz.engine.cache import cache_key

# Rough per-(doc, pipeline) USD priors by adapter for the PRE-run estimate; override per pipeline
# with config `cost_prior_usd`. Verify against real spend (PLAN §13).
_PRIORS = {
    "fake": 0.0, "gemini": 0.01, "openai": 0.01, "anthropic": 0.03,
    "extend": 0.05, "llamaindex": 0.02,
}


class BudgetGuard:
    """Enforces RunOptions.budget_usd: accumulate actuals, stop-early once over the cap."""

    def __init__(self, cap_usd: Optional[float]):
        self.cap_usd = cap_usd
        self.spent = 0.0

    def add(self, usd: Optional[float]) -> None:
        self.spent += usd or 0.0

    @property
    def over(self) -> bool:
        return self.cap_usd is not None and self.spent > self.cap_usd


def estimate_cost(run, dataset, cache, schema_hash: str) -> dict:
    """Estimate USD = (uncached docs x pipelines) x per-pipeline prior; cached cells cost nothing."""
    uncached = cached = 0
    estimate = 0.0
    for doc in dataset.documents:
        for pc in run.pipelines:
            if cache.get(cache_key(doc.doc_id, pc.config_hash, schema_hash)) is not None:
                cached += 1
            else:
                uncached += 1
                estimate += pc.config.get("cost_prior_usd", _PRIORS.get(pc.adapter, 0.02))
    return {"estimate_usd": round(estimate, 4), "uncached_cells": uncached, "cached_cells": cached}
