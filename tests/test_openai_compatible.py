"""`openai_compatible` points the OpenAI SDK at any OpenAI-compatible chat endpoint (Ollama,
vLLM, LM Studio, Together, Groq, OpenRouter, ...). It reuses the whole OpenAI extraction
template — only how the client is constructed differs (a `base_url`, and a placeholder key for
keyless local servers). No SDK/network: we inject a fake `openai` module that records kwargs."""
import sys
import types

import pytest

from ezpz.adapters.registry import get_adapter
from ezpz.core.run import PipelineConfig


def _fake_openai(monkeypatch) -> dict:
    """Install a fake `openai` module whose `OpenAI(...)` records its constructor kwargs."""
    captured: dict = {}
    mod = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, **kwargs):
            captured.clear()
            captured.update(kwargs)

    mod.OpenAI = OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mod)
    return captured


def _pipe(**cfg):
    return get_adapter("openai_compatible")(PipelineConfig(adapter="openai_compatible", config=cfg))


def test_registered_and_inherits_the_openai_template():
    pipe = _pipe(base_url="http://localhost:11434/v1", model="llama3.1")
    assert pipe.model == "llama3.1"                 # model id is endpoint-specific (no default)
    assert pipe.capabilities.confidence is False    # plain LLM extraction unless self_rate is set


def test_threads_base_url_and_uses_placeholder_key_when_none_configured(monkeypatch):
    captured = _fake_openai(monkeypatch)
    _pipe(base_url="http://localhost:11434/v1", model="llama3.1")._client()
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key"] == "not-needed"      # local servers ignore it; SDK requires non-empty


def test_uses_configured_api_key_when_present(monkeypatch):
    captured = _fake_openai(monkeypatch)
    monkeypatch.setenv("LOCAL_KEY", "sk-real")
    _pipe(base_url="https://api.together.xyz/v1", api_key_env="LOCAL_KEY")._client()
    assert captured["api_key"] == "sk-real"
    assert captured["base_url"] == "https://api.together.xyz/v1"


def test_requires_base_url(monkeypatch):
    _fake_openai(monkeypatch)
    with pytest.raises(ValueError, match="base_url"):
        _pipe(model="llama3.1")._client()


def test_plain_openai_adapter_can_also_forward_base_url(monkeypatch):
    captured = _fake_openai(monkeypatch)
    get_adapter("openai")(
        PipelineConfig(adapter="openai", config={"base_url": "http://proxy:8080/v1"})
    )._client()
    assert captured["base_url"] == "http://proxy:8080/v1"
