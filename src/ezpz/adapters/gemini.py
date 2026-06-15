"""Gemini-direct adapter (Google Gen AI SDK, package `google-genai`).

Structured output via `response_mime_type=application/json` + a JSON Schema; cost from token
usage. The prompt template is part of `config` (versioned), so prompt A vs prompt B are two
separate, comparable pipelines. No native confidence/provenance.

Verify the SDK/model ids/pricing against current docs before pinning (PLAN §13).
"""
from __future__ import annotations

from typing import Any

from ezpz.adapters.llm_base import LLMExtractionPipeline
from ezpz.adapters.registry import register


@register("gemini")
class GeminiPipeline(LLMExtractionPipeline):
    default_model = "gemini-2.5-flash"
    # Prices vary by model/tier — supply {"prices": {"input_per_1m", "output_per_1m"}} in config.
    default_prices: dict[str, tuple[float, float]] = {}

    def _client(self):
        from google import genai

        key = self._api_key()
        return genai.Client(api_key=key) if key else genai.Client()

    def _generate(self, prepared: Any, ingested: Any) -> tuple[Any, dict, dict]:
        from google.genai import types

        if ingested["kind"] == "text":
            contents: Any = f"{prepared['prompt']}\n\nDocument:\n{ingested['text']}"
        else:
            import base64

            part = types.Part.from_bytes(
                data=base64.b64decode(ingested["data_b64"]), mime_type=ingested["mime"]
            )
            contents = [prepared["prompt"], part]

        config: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_json_schema": prepared["schema"],
            "system_instruction": prepared["system"],
        }
        temperature = self.config.config.get("temperature")
        if temperature is not None:
            config["temperature"] = temperature

        resp = self._client().models.generate_content(
            model=prepared["model"], contents=contents, config=config
        )
        um = getattr(resp, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(um, "prompt_token_count", None),
            "output_tokens": getattr(um, "candidates_token_count", None),
        }
        return resp.text, usage, {"finish_reason": getattr(resp, "finish_reason", None)}

    def classify_error(self, exc: Exception) -> str:
        try:
            from google.genai import errors

            if isinstance(exc, errors.ServerError):
                return "transport"
            if isinstance(exc, errors.ClientError):
                return "unknown"  # 4xx -> usually a config/schema problem, not transient
        except Exception:
            pass
        return self._classify_common(exc)
