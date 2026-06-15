"""Shared base for LLM extraction adapters (Gemini / Anthropic / OpenAI).

All three do the same thing: compile the task schema into a JSON Schema + a prompt, call the
provider with structured-output mode, parse the JSON response, and derive cost from token usage.
Only the provider call (`_generate`) and exception mapping (`classify_error`) differ — those live
in the per-provider subclasses, so SDK churn stays contained there.

Invariants honored:
  - The task schema is COMPILED here, never invented by a provider.
  - `map` is structural only; value-normalization happens centrally in the engine, after `map`.
  - `raw` returned by `invoke` is JSON-serializable (cache contract), so a cache hit re-maps it.
  - No native confidence/provenance from these tools -> FieldValue.confidence stays None.
"""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from ezpz.adapters.base import Capabilities, Pipeline
from ezpz.core.document import Document
from ezpz.core.result import Cost, FieldValue
from ezpz.core.run import PipelineConfig
from ezpz.core.task import FieldSpec, Task
from ezpz.core.values import ValueType

DEFAULT_SYSTEM = (
    "You are a precise document-extraction engine. Extract exactly the requested fields and "
    "return JSON matching the provided schema. If a field is not present in the document, return "
    "null for it — never guess or fabricate a value."
)

# JSON Schema 'type' for each scalar ValueType (null is unioned in to allow 'absent' -> null).
_SCALAR_TYPES = {
    ValueType.STRING: "string",
    ValueType.DATE: "string",
    ValueType.DATETIME: "string",
    ValueType.ENUM: "string",
    ValueType.INTEGER: "integer",
    ValueType.NUMBER: "number",
    ValueType.BOOLEAN: "boolean",
}


class RefusalError(Exception):
    """Raised when a provider declines to answer (maps to error_class='refusal')."""


def _field_schema(spec: FieldSpec) -> dict:
    """JSON Schema for one field. Everything is nullable so 'absent' is expressible as null."""
    if spec.type == ValueType.CURRENCY:
        node: dict = {
            "type": ["object", "null"],
            "properties": {
                "amount": {"type": ["number", "null"]},
                "currency": {"type": ["string", "null"]},
            },
            "required": ["amount", "currency"],
            "additionalProperties": False,
        }
    elif spec.type == ValueType.LIST:
        item = spec.item or FieldSpec(name="item", type=ValueType.STRING)
        node = {"type": ["array", "null"], "items": _field_schema(item)}
    elif spec.type == ValueType.OBJECT:
        fields = spec.fields or []
        node = {
            "type": ["object", "null"],
            "properties": {f.name: _field_schema(f) for f in fields},
            "required": [f.name for f in fields],
            "additionalProperties": False,
        }
    else:
        node = {"type": [_SCALAR_TYPES.get(spec.type, "string"), "null"]}
        if spec.type == ValueType.ENUM and spec.enum_values:
            node["enum"] = [*spec.enum_values, None]
    if spec.description:
        node["description"] = spec.description
    return node


def task_to_json_schema(task: Task) -> dict:
    """Compile the task into a strict JSON Schema (all fields required + nullable)."""
    return {
        "type": "object",
        "properties": {f.name: _field_schema(f) for f in task.fields},
        "required": [f.name for f in task.fields],
        "additionalProperties": False,
    }


def build_prompt(task: Task) -> str:
    """Field names + descriptions as guidance. Does NOT restate the JSON schema (redundant)."""
    lines = [task.instructions or "Extract the following fields from the document."]
    lines.append("")
    lines.append("Fields to extract:")
    for f in task.fields:
        desc = f" — {f.description}" if f.description else ""
        lines.append(f"- {f.name} ({f.type.value}){desc}")
    lines.append("")
    lines.append("Return null for any field that is not present in the document.")
    return "\n".join(lines)


def load_document(document: Document) -> dict:
    """Read the document for the provider. Text files inline as text; everything else as base64."""
    path = Path(document.source_path or document.path)
    data = path.read_bytes()
    mime = document.mime or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    if mime.startswith("text/") or mime in ("application/json", "text/markdown"):
        return {"kind": "text", "text": data.decode("utf-8", errors="replace"), "mime": mime}
    return {"kind": "binary", "data_b64": base64.b64encode(data).decode(), "mime": mime}


def _as_float(x: Any) -> "float | None":
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


class LLMExtractionPipeline(Pipeline):
    """Template for prompt->call->parse LLM extraction. Subclasses implement `_generate`.

    Provider-agnostic options (any LLM adapter): ``model``, ``system_prompt``, ``prices``, and
    ``self_rate: true`` — which makes the model emit a per-field confidence (so it can drive a
    cascade gate). The technique is the same across providers; only `_generate` differs.
    """

    default_model: str = ""
    # {model_id: (input_usd_per_1M, output_usd_per_1M)} — VERIFY against current pricing (PLAN §13).
    default_prices: dict[str, tuple[float, float]] = {}

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        # confidence is only ever the model's own self-rating — advertised only when it's on.
        self.capabilities = Capabilities(confidence=bool(config.config.get("self_rate")))

    @property
    def model(self) -> str:
        return self.config.config.get("model") or self.default_model

    def compile(self, task: Task) -> Any:
        schema = task_to_json_schema(task)
        prompt = build_prompt(task)
        if self.config.config.get("self_rate"):  # provider-neutral self-rated confidence
            schema["properties"]["_confidence"] = {
                "type": "object",
                "properties": {f.name: {"type": ["number", "null"]} for f in task.fields},
                "required": [f.name for f in task.fields],
                "additionalProperties": False,
            }
            schema["required"] = [*schema["required"], "_confidence"]
            prompt += (
                "\n\nAlso return `_confidence`: your confidence from 0.0 to 1.0 for EACH field above "
                "(1.0 = certain it is correct, 0.0 = guessing)."
            )
        return {
            "schema": schema, "prompt": prompt,
            "system": self.config.config.get("system_prompt") or DEFAULT_SYSTEM,
            "model": self.model,
        }

    def ingest(self, document: Document) -> Any:
        return load_document(document)

    def invoke(self, prepared: Any, ingested: Any) -> tuple[Any, Cost]:
        text, usage, meta = self._generate(prepared, ingested)
        parsed = self._parse_json(text)
        raw = {"fields": parsed, "_meta": {**meta, "model": prepared["model"], "usage": usage}}
        return raw, self._cost(usage)

    def map(self, raw: Any, task: Task) -> dict[str, FieldValue]:
        fields = raw.get("fields", {}) if isinstance(raw, dict) else {}
        # Native provenance isn't available; confidence is only the model's self-rating (`self_rate`).
        conf = fields.get("_confidence") if self.config.config.get("self_rate") else None
        conf = conf if isinstance(conf, dict) else {}
        return {
            f.name: FieldValue(value=fields.get(f.name), confidence=_as_float(conf.get(f.name)))
            for f in task.fields
        }

    # ---- subclass hooks ----
    def _generate(self, prepared: Any, ingested: Any) -> tuple[Any, dict, dict]:
        """Call the provider with structured output. Return (json_text_or_obj, usage, meta)."""
        raise NotImplementedError

    def classify_error(self, exc: Exception) -> str:
        return self._classify_common(exc)

    # ---- shared helpers ----
    @staticmethod
    def _parse_json(text: Any) -> dict:
        if isinstance(text, dict):
            return text
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            raise json.JSONDecodeError(f"provider returned non-JSON: {e}", str(text or ""), 0)
        if not isinstance(obj, dict):
            raise json.JSONDecodeError("provider returned non-object JSON", str(text), 0)
        return obj

    def _cost(self, usage: dict) -> Cost:
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        prices = self.config.config.get("prices") or self.default_prices.get(self.model)
        usd = None
        if prices is not None and in_tok is not None and out_tok is not None:
            in_p, out_p = (
                (prices["input_per_1m"], prices["output_per_1m"])
                if isinstance(prices, dict)
                else prices
            )
            usd = round(in_tok / 1_000_000 * in_p + out_tok / 1_000_000 * out_p, 6)
        return Cost(input_tokens=in_tok, output_tokens=out_tok, usd=usd, raw=dict(usage))

    @staticmethod
    def _classify_common(exc: Exception) -> str:
        if isinstance(exc, RefusalError):
            return "refusal"
        if isinstance(exc, json.JSONDecodeError):
            return "parse"
        if isinstance(exc, TimeoutError):
            return "timeout"
        if isinstance(exc, (ConnectionError, OSError)):
            return "transport"
        return "unknown"

    def _api_key(self) -> "str | None":
        """Resolve the API key from the env var NAMED in config (never inline). See conventions."""
        import os

        env_name = self.config.config.get("api_key_env")
        return os.environ.get(env_name) if env_name else None
