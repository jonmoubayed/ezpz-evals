"""Pipeline = the adapter base. Subclasses implement five stages; `run` sequences them,
times them, normalizes, and maps errors into the canonical result. No provider logic here."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from ezpz.core.document import Document
from ezpz.core.result import (Cost, ErrorInfo, ExtractionResult, FieldValue,
                              ResultStatus, Timing)
from ezpz.core.run import PipelineConfig
from ezpz.core.task import Task


class Capabilities:
    """What a tool can provide, so UI/scoring degrade gracefully."""
    def __init__(self, confidence: bool = False, provenance: bool = False, is_async: bool = False):
        self.confidence = confidence
        self.provenance = provenance
        self.is_async = is_async


class Pipeline(ABC):
    capabilities: Capabilities = Capabilities()

    def __init__(self, config: PipelineConfig):
        self.config = config

    @property
    def pipeline_id(self) -> str:
        return self.config.config_hash

    # ---- the five stages (subclasses implement) ----
    @abstractmethod
    def compile(self, task: Task) -> Any:
        """task schema -> backend-native schema/prompt. Cacheable per (task, config)."""

    @abstractmethod
    def ingest(self, document: Document) -> Any:
        """Get the document into a backend-acceptable form (bytes, upload handle, parsed text)."""

    @abstractmethod
    def invoke(self, prepared: Any, ingested: Any) -> tuple[Any, Cost]:
        """Call the backend. Handle retries/timeouts/polling. Return (raw_response, cost).

        ``raw_response`` MUST be JSON-serializable: the engine caches it and re-runs ``map`` over
        it on a cache hit, so it has to round-trip through JSON unchanged.
        """

    @abstractmethod
    def map(self, raw: Any, task: Task) -> dict[str, FieldValue]:
        """Structural map: raw response -> canonical FieldValue dict (pre value-normalization)."""

    def classify_error(self, exc: Exception) -> str:
        """Map an exception raised during a stage to transport|parse|refusal|timeout|unknown.

        Provider-specific exception knowledge lives in the adapter (SDK churn stays contained);
        the base defaults to 'unknown'.
        """
        return "unknown"

    # ---- orchestration (shared) ----
    def extract(self, document: Document, task: Task) -> tuple[ExtractionResult, Any]:
        """Run the stages, timed and error-wrapped. Returns (result, raw_response) so the
        engine can cache the RAWEST artifact and derive normalized fields on read."""
        stage_ms: dict[str, float] = {}
        raw: Any = None
        t0 = time.perf_counter()
        try:
            s = time.perf_counter()
            prepared = self.compile(task)
            stage_ms["compile"] = _ms(s)
            s = time.perf_counter()
            ingested = self.ingest(document)
            stage_ms["ingest"] = _ms(s)
            s = time.perf_counter()
            raw, cost = self.invoke(prepared, ingested)
            stage_ms["invoke"] = _ms(s)
            s = time.perf_counter()
            fields = self.map(raw, task)
            stage_ms["map"] = _ms(s)
            # Value-normalization is applied centrally by the engine AFTER map (never here),
            # so it is identical across adapters and applied to ground truth too.
            result = ExtractionResult(
                doc_id=document.doc_id, pipeline_id=self.pipeline_id,
                status=ResultStatus.OK, fields=fields, cost=cost,
                timing=Timing(latency_ms=_ms(t0), stage_ms=stage_ms),
            )
            return result, raw
        except Exception as e:
            result = ExtractionResult(
                doc_id=document.doc_id, pipeline_id=self.pipeline_id,
                status=ResultStatus.ERROR,
                error=ErrorInfo(error_class=self.classify_error(e), message=str(e)),
                timing=Timing(latency_ms=_ms(t0), stage_ms=stage_ms),
            )
            return result, raw

    def run(self, document: Document, task: Task) -> ExtractionResult:
        """Convenience: full extraction, discarding the raw response."""
        result, _raw = self.extract(document, task)
        return result


def _ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000.0
