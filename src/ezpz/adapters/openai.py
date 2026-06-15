"""OpenAI adapter (official `openai` SDK).

Structured output via Chat Completions `response_format={"type": "json_schema", strict: true}`;
cost from token usage. No native confidence/provenance.

Verify model ids / pricing against current docs before pinning (PLAN §13).
"""
from __future__ import annotations

from typing import Any

from ezpz.adapters.llm_base import LLMExtractionPipeline, RefusalError
from ezpz.adapters.registry import register


@register("openai")
class OpenAIPipeline(LLMExtractionPipeline):
    default_model = "gpt-4o-mini"
    # Prices vary by model — supply {"prices": {"input_per_1m", "output_per_1m"}} in config.
    default_prices: dict[str, tuple[float, float]] = {}

    def _client(self):
        from openai import OpenAI

        key = self._api_key()
        base_url = self.config.config.get("base_url")  # None -> SDK default (api.openai.com)
        return OpenAI(api_key=key, base_url=base_url) if key else OpenAI(base_url=base_url)

    def _user_content(self, prepared: Any, ingested: Any) -> Any:
        if ingested["kind"] == "text":
            return f"{prepared['prompt']}\n\nDocument:\n{ingested['text']}"
        if ingested["mime"].startswith("image/"):
            url = f"data:{ingested['mime']};base64,{ingested['data_b64']}"
            return [
                {"type": "text", "text": prepared["prompt"]},
                {"type": "image_url", "image_url": {"url": url}},
            ]
        # Non-image binary (e.g. PDF) isn't inline-able via chat content; send the prompt only.
        return f"{prepared['prompt']}\n\n(Document is a {ingested['mime']} file.)"

    def _generate(self, prepared: Any, ingested: Any) -> tuple[Any, dict, dict]:
        resp = self._client().chat.completions.create(
            model=prepared["model"],
            messages=[
                {"role": "system", "content": prepared["system"]},
                {"role": "user", "content": self._user_content(prepared, ingested)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": prepared["schema"], "strict": True},
            },
            temperature=self.config.config.get("temperature", 0),
        )
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise RefusalError(choice.message.refusal)
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }
        return choice.message.content, usage, {"finish_reason": choice.finish_reason}

    def classify_error(self, exc: Exception) -> str:
        try:
            import openai

            if isinstance(exc, openai.APITimeoutError):
                return "timeout"
            if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError)):
                return "transport"
            if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
                return "transport"
        except Exception:
            pass
        return self._classify_common(exc)


@register("openai_compatible")
class OpenAICompatiblePipeline(OpenAIPipeline):
    """Any OpenAI-compatible chat-completions endpoint — Ollama, vLLM, LM Studio, Together, Groq,
    Fireworks, OpenRouter, DeepSeek, Azure OpenAI, ... — via the official `openai` SDK pointed at a
    custom `base_url`. Identical extraction template as `openai`; only the client construction
    differs. Config: `base_url` (required, the endpoint URL), `model` (endpoint-specific, no
    default), optional `api_key_env`, and optional `prices` ({input_per_1m, output_per_1m}; omit
    for self-hosted to leave cost unknown).
    """

    default_model = ""  # endpoint-specific; there's no sensible default model id to assume

    def _client(self):
        from openai import OpenAI

        base_url = self.config.config.get("base_url")
        if not base_url:
            raise ValueError("openai_compatible requires `base_url` (the endpoint URL).")
        # Local servers (Ollama, vLLM, LM Studio) ignore the key, but the SDK requires a non-empty
        # one — send a harmless placeholder when no `api_key_env` is configured.
        return OpenAI(api_key=self._api_key() or "not-needed", base_url=base_url)
