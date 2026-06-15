"""The canonical result shape — the output contract. Adapters map raw responses into this;
nothing downstream ever reads a raw response for scoring."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ResultStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"   # some fields parsed, some failed (e.g. bad JSON repaired partially)
    ERROR = "error"       # extraction failed entirely


class Provenance(BaseModel):
    page: Optional[int] = None
    bbox: Optional[list[float]] = None     # [x0, y0, x1, y1]
    text_span: Optional[str] = None


class FieldValue(BaseModel):
    value: Any = None                       # canonical representation (post-normalization)
    confidence: Optional[float] = None      # None if the tool does not emit confidence
    provenance: Optional[Provenance] = None


class Timing(BaseModel):
    latency_ms: float
    stage_ms: Optional[dict[str, float]] = None  # compile/ingest/invoke/map breakdown


class Cost(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    usd: Optional[float] = None             # normalized cost where computable
    units: Optional[float] = None           # native billing unit (e.g. pages) where token-less
    raw: dict[str, Any] = Field(default_factory=dict)  # provider-reported usage, verbatim


class ErrorInfo(BaseModel):
    error_class: str                        # transport | parse | refusal | timeout | unknown
    message: Optional[str] = None


class ExtractionResult(BaseModel):
    doc_id: str
    pipeline_id: str                        # = PipelineConfig.config_hash
    status: ResultStatus
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    raw_response_ref: Optional[str] = None  # blob hash; raw response kept for debugging only
    timing: Optional[Timing] = None
    cost: Optional[Cost] = None
    error: Optional[ErrorInfo] = None
    extras: dict[str, Any] = Field(default_factory=dict)  # adapter-specific; scoring never reads
