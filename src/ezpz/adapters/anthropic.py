"""Anthropic Claude adapter (official `anthropic` SDK).

Structured output via Messages `output_config.format` (json_schema); cost from token usage.
Pricing for current Claude models is authoritative (per the claude-api reference). Note: the
Opus 4.7/4.8 and Fable families reject `temperature` (400), so it is only sent when configured
AND the model accepts it.
"""
from __future__ import annotations

from typing import Any

from ezpz.adapters.llm_base import LLMExtractionPipeline, RefusalError
from ezpz.adapters.registry import register

_NO_TEMPERATURE_PREFIXES = ("claude-opus-4-8", "claude-opus-4-7", "claude-fable", "claude-mythos")


@register("anthropic")
class AnthropicPipeline(LLMExtractionPipeline):
    default_model = "claude-opus-4-8"
    # USD per 1M tokens (input, output) — current Claude models.
    default_prices: dict[str, tuple[float, float]] = {
        "claude-opus-4-8": (5.0, 25.0),
        "claude-opus-4-7": (5.0, 25.0),
        "claude-opus-4-6": (5.0, 25.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-haiku-4-5": (1.0, 5.0),
        "claude-fable-5": (10.0, 50.0),
    }

    def _client(self):
        import anthropic

        key = self._api_key()
        return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    def _content(self, prepared: Any, ingested: Any) -> Any:
        if ingested["kind"] == "text":
            return f"{prepared['prompt']}\n\nDocument:\n{ingested['text']}"
        block_type = "image" if ingested["mime"].startswith("image/") else "document"
        return [
            {"type": "text", "text": prepared["prompt"]},
            {
                "type": block_type,
                "source": {
                    "type": "base64",
                    "media_type": ingested["mime"],
                    "data": ingested["data_b64"],
                },
            },
        ]

    def _generate(self, prepared: Any, ingested: Any) -> tuple[Any, dict, dict]:
        kwargs: dict[str, Any] = {
            "model": prepared["model"],
            "max_tokens": self.config.config.get("max_tokens", 4096),
            "system": prepared["system"],
            "messages": [{"role": "user", "content": self._content(prepared, ingested)}],
            "output_config": {"format": {"type": "json_schema", "schema": prepared["schema"]}},
        }
        temperature = self.config.config.get("temperature")
        if temperature is not None and not prepared["model"].startswith(_NO_TEMPERATURE_PREFIXES):
            kwargs["temperature"] = temperature

        resp = self._client().messages.create(**kwargs)
        if getattr(resp, "stop_reason", None) == "refusal":
            raise RefusalError(f"{prepared['model']} refused the request")
        text = next(
            (getattr(b, "text", None) for b in resp.content if getattr(b, "type", None) == "text"),
            None,
        )
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return text, usage, {"stop_reason": getattr(resp, "stop_reason", None)}

    def classify_error(self, exc: Exception) -> str:
        try:
            import anthropic

            if isinstance(exc, anthropic.APITimeoutError):
                return "timeout"
            if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError,
                                anthropic.InternalServerError)):
                return "transport"
            if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
                return "transport"
        except Exception:
            pass
        return self._classify_common(exc)
