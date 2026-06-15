"""LlamaIndex adapter: a parser (LlamaParse or a simple reader) then a structured-extraction
program over the parsed text. Unlike Gemini/Anthropic/OpenAI (end-to-end), this is parse+extract,
so parse quality is a DISTINCT failure mode you're evaluating — encode ALL sub-choices (parser,
backend model, chunking) in `config` so the config_hash is meaningful.

Reuses the shared LLM template (compile/map/cost); overrides `ingest` (parse) and `_generate`
(the LlamaIndex call), and folds the parse cost into the total. VERIFY the LlamaIndex / LlamaParse
API + pricing against current docs (PLAN §13). SDK imports are lazy; `_generate` is the test seam.
"""
from __future__ import annotations

import json
from typing import Any

from ezpz.adapters.llm_base import LLMExtractionPipeline, load_document
from ezpz.adapters.registry import register
from ezpz.core.document import Document
from ezpz.core.result import Cost


def _usage_from(resp: Any) -> dict:
    raw = getattr(resp, "raw", None) or {}
    usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
    if usage is None:
        return {"input_tokens": None, "output_tokens": None}

    def g(k):
        return usage.get(k) if isinstance(usage, dict) else getattr(usage, k, None)

    return {
        "input_tokens": g("prompt_tokens") or g("input_tokens"),
        "output_tokens": g("completion_tokens") or g("output_tokens"),
    }


@register("llamaindex")
class LlamaIndexPipeline(LLMExtractionPipeline):
    default_model = "gpt-4o-mini"
    default_prices: dict[str, tuple[float, float]] = {}

    @property
    def model(self) -> str:
        cfg = self.config.config
        return cfg.get("backend_model") or cfg.get("model") or self.default_model

    def ingest(self, document: Document) -> Any:
        parser = self.config.config.get("parser", "simple")
        if parser == "llamaparse":
            text = self._llamaparse(document)
        else:
            doc = load_document(document)
            text = doc.get("text") or f"(binary {doc.get('mime')} document)"
        return {"kind": "text", "text": text, "parser": parser}

    def _generate(self, prepared: Any, ingested: Any) -> tuple[Any, dict, dict]:
        prompt = (
            f"{prepared['prompt']}\n\nReturn ONLY a JSON object matching this schema (no prose):\n"
            f"{json.dumps(prepared['schema'])}\n\nDocument:\n{ingested['text']}"
        )
        resp = self._llm().complete(prompt)
        return resp.text, _usage_from(resp), {"parser": ingested["parser"]}

    def _cost(self, usage: dict) -> Cost:
        cost = super()._cost(usage)  # backend-LLM token cost
        parse_usd = self.config.config.get("parse_cost_usd")
        if parse_usd is not None:
            cost.usd = round((cost.usd or 0.0) + parse_usd, 6)
            cost.raw["parse_usd"] = parse_usd
        return cost

    def _llm(self):
        import os

        from llama_index.llms.openai import OpenAI as LlamaOpenAI

        api_key = os.environ.get(self.config.config.get("api_key_env", "OPENAI_API_KEY"))
        return LlamaOpenAI(
            model=self.model, api_key=api_key,
            temperature=self.config.config.get("temperature", 0),
        )

    def _llamaparse(self, document: Document) -> str:
        import os

        from llama_parse import LlamaParse

        api_key = os.environ.get(
            self.config.config.get("llama_cloud_api_key_env", "LLAMA_CLOUD_API_KEY")
        )
        docs = LlamaParse(api_key=api_key, result_type="markdown").load_data(
            document.source_path or document.path
        )
        return "\n\n".join(d.text for d in docs)
