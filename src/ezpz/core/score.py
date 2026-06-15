"""Score records. Scorers emit FieldScore over the canonical result; aggregation lives in
ezpz.engine.aggregate so raw per-field results can always be re-aggregated."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class FieldScore(BaseModel):
    doc_id: str
    pipeline_id: str
    field: str
    scorer: str
    value: float                 # 0..1 (or a metric-native scalar)
    passed: Optional[bool] = None
    detail: dict[str, Any] = Field(default_factory=dict)  # e.g. judge rationale, diff


class MetricSummary(BaseModel):
    metric: str
    value: float
    n: int
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    slice: Optional[str] = None
