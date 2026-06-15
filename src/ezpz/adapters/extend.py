"""Extend.ai adapter. Managed extraction: give it a schema, get structured fields back with
confidences and (usually) bounding-box provenance.

Job-based: submit -> poll -> fetch, hidden behind the sync stage interface. Bills per page ->
Cost.units. This is the first adapter with native confidence + provenance, so it stresses the
abstraction: confidence -> FieldValue.confidence, bbox -> FieldValue.provenance.

VERIFY the REST endpoints / response shape / per-page pricing against current Extend docs before
pinning (PLAN §13). The HTTP is isolated in `_extract`; tests override that seam.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from ezpz.adapters.base import Capabilities, Pipeline
from ezpz.adapters.llm_base import load_document, task_to_json_schema
from ezpz.adapters.registry import register
from ezpz.core.document import Document
from ezpz.core.result import Cost, FieldValue, Provenance
from ezpz.core.task import Task
from ezpz.core.values import ValueType

# canonical ValueType -> Extend field type. compile() fails loudly on anything unmapped.
_EXTEND_TYPES = {
    ValueType.STRING: "text", ValueType.INTEGER: "number", ValueType.NUMBER: "number",
    ValueType.BOOLEAN: "boolean", ValueType.DATE: "date", ValueType.DATETIME: "date",
    ValueType.ENUM: "enum", ValueType.CURRENCY: "currency", ValueType.LIST: "array",
    ValueType.OBJECT: "object",
}
_TERMINAL = {"PROCESSED", "COMPLETED", "DONE", "FAILED", "ERRORED"}


def _provenance(refs: Any) -> Optional[Provenance]:
    first = refs[0] if isinstance(refs, list) and refs else refs
    if not isinstance(first, dict):
        return None
    bbox = first.get("bbox")
    if isinstance(bbox, dict):
        bbox = [bbox.get("x0"), bbox.get("y0"), bbox.get("x1"), bbox.get("y1")]
    return Provenance(page=first.get("page"), bbox=bbox)


def _to_field_value(node: Any) -> FieldValue:
    if isinstance(node, dict) and "value" in node:  # Extend's field-keyed {value, confidence, references}
        return FieldValue(
            value=node["value"],
            confidence=node.get("confidence"),
            provenance=_provenance(node.get("references") or node.get("provenance")),
        )
    return FieldValue(value=node)


@register("extend")
class ExtendPipeline(Pipeline):
    capabilities = Capabilities(confidence=True, provenance=True, is_async=True)

    def compile(self, task: Task) -> Any:
        fields = []
        for f in task.fields:
            if f.type not in _EXTEND_TYPES:
                raise NotImplementedError(f"Extend can't express field '{f.name}' of type {f.type}")
            fields.append({"name": f.name, "type": _EXTEND_TYPES[f.type], "description": f.description})
        return {
            "schema": task_to_json_schema(task),
            "fields": fields,
            "processor_id": self.config.config.get("processor_id"),
        }

    def ingest(self, document: Document) -> Any:
        return load_document(document)

    def invoke(self, prepared: Any, ingested: Any) -> tuple[Any, Cost]:
        raw = self._extract(prepared, ingested)  # JSON-serializable Extend response (cache contract)
        pages = (raw.get("metadata") or {}).get("pageCount", raw.get("pageCount"))
        price = self.config.config.get("price_per_page")
        usd = round(pages * price, 6) if (pages is not None and price is not None) else None
        cost = Cost(units=float(pages) if pages is not None else None, usd=usd,
                    raw={"pageCount": pages})
        return raw, cost

    def map(self, raw: Any, task: Task) -> dict[str, FieldValue]:
        output = raw.get("output") or raw.get("fields") or {}
        return {f.name: _to_field_value(output.get(f.name)) for f in task.fields}

    def classify_error(self, exc: Exception) -> str:
        try:
            import httpx

            if isinstance(exc, httpx.TimeoutException):
                return "timeout"
            if isinstance(exc, httpx.HTTPError):
                return "transport"
        except Exception:
            pass
        if isinstance(exc, TimeoutError):
            return "timeout"
        if isinstance(exc, json.JSONDecodeError):
            return "parse"
        return "unknown"

    # ---- network seam (override in tests) ----
    def _extract(self, prepared: Any, ingested: Any) -> dict:
        """submit -> poll -> fetch. Endpoints/shape per current Extend docs (PLAN §13)."""
        import time

        import httpx

        cfg = self.config.config
        key = os.environ.get(cfg.get("api_key_env", "EXTEND_API_KEY"))
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        deadline = time.time() + cfg.get("timeout_s", 120)
        with httpx.Client(base_url=cfg.get("base_url", "https://api.extend.ai"),
                          headers=headers, timeout=30) as client:
            run_id = client.post("/v1/processor_runs", json={
                "processorId": prepared.get("processor_id"),
                "schema": prepared["schema"],
                "file": {"contents": ingested.get("data_b64") or ingested.get("text"),
                         "mimeType": ingested["mime"]},
            }).raise_for_status().json()["id"]
            while time.time() < deadline:
                resp = client.get(f"/v1/processor_runs/{run_id}").raise_for_status().json()
                if resp.get("status") in _TERMINAL:
                    if resp["status"] in ("FAILED", "ERRORED"):
                        raise RuntimeError(f"extend run {run_id} failed")
                    return resp
                time.sleep(cfg.get("poll_interval_s", 2))
        raise TimeoutError(f"extend run {run_id} timed out")
