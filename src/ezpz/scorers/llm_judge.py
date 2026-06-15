"""LLM-as-judge for open-ended fields / holistic correctness.

Reliability discipline (config pins all of this):
  - Fix judge model + prompt + rubric (part of the task's identity).
  - Calibrate against a human-labeled subset; watch verbosity/position bias.
  - Treat scores as noisier than deterministic ones; CACHE judge calls (they cost money),
    keyed on hash(prediction + ground_truth + judge_config). Default backend = Anthropic (lazy);
    override `_judge` for other providers / in tests.
"""
from __future__ import annotations

from ezpz.scorers._cache import cache_key, cached
from ezpz.scorers.base import Scorer, resolve_empty_states
from ezpz.scorers.registry import register

_DEFAULT_RUBRIC = "Rate how well the PREDICTION matches the GROUND TRUTH for this field, 0.0–1.0."
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "rationale": {"type": "string"}},
    "required": ["score", "rationale"],
    "additionalProperties": False,
}


@register("llm_judge")
class LLMJudge(Scorer):
    name = "llm_judge"

    def score(self, prediction, ground_truth, ctx):
        empty = resolve_empty_states(prediction, ground_truth)
        if empty is not None:
            value, passed, detail = empty
            return self._result(ctx, value, passed, **detail)

        cfg = ctx.config
        threshold = float(cfg.get("threshold", 0.5))
        key = cache_key(
            "judge", cfg.get("model"), cfg.get("prompt"), cfg.get("rubric"),
            ctx.spec.name, str(prediction), str(ground_truth),
        )
        out = cached(key, lambda: self._judge(str(prediction), str(ground_truth), ctx))
        score = float(out.get("score", 0.0))
        return self._result(ctx, round(score, 4), score >= threshold, rationale=out.get("rationale", ""))

    def _judge(self, prediction: str, ground_truth: str, ctx) -> dict:
        """Default backend: Anthropic Claude judging against the rubric. Returns {score, rationale}."""
        import json
        import os

        import anthropic

        cfg = ctx.config
        model = cfg.get("model", "claude-haiku-4-5")
        rubric = cfg.get("rubric") or _DEFAULT_RUBRIC
        api_key = os.environ.get(cfg.get("api_key_env", "ANTHROPIC_API_KEY"))
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            system="You are a strict evaluation judge. Return only JSON {score, rationale}.",
            messages=[{
                "role": "user",
                "content": (
                    f"Field: {ctx.spec.name}\nRubric: {rubric}\n"
                    f"PREDICTION: {prediction}\nGROUND TRUTH: {ground_truth}"
                ),
            }],
            output_config={"format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
        )
        text = next(
            (getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"),
            "{}",
        )
        return json.loads(text)
