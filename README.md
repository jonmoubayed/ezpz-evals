# ezpz evals

[![CI](https://github.com/jonmoubayed/ezpz-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/jonmoubayed/ezpz-evals/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

A **local-first** harness for evaluating unstructured-document pipelines (extraction today,
retrieval/QA later) and comparing tools/models side by side on the same cohort.

> Status: **implemented end-to-end (M0–M7)** — extraction, central normalization, pluggable
> scoring, raw-response caching, SQLite + content-addressed blobs, a Streamlit viewer, adapters
> (Gemini · Anthropic · OpenAI · any OpenAI-compatible/local endpoint · Extend · LlamaIndex · a
> no-network `fake`), and M7 hardening (per-provider concurrency, retries, circuit breaker, budget
> guard, resumability, machine-readable output + CI gating).

## The one idea everything hangs off

Every tool is wrapped in an **adapter** that honors two contracts:

1. **Input contract** — the task schema is authored once and each adapter *compiles* it into
   its own native format (Extend field config, Gemini response schema, LlamaIndex Pydantic class).
2. **Output contract** — each adapter maps its raw response into one **canonical result shape**.

Scoring, storage, and reporting touch **only** the canonical shape. Raw responses are kept for
debugging but never scored. A separate, **central** value-normalization step (type-aware) runs
before scoring so `"Jan 3 2024"` and `"2024-01-03"` compare equal across every tool.

## Layers

- `core/`        — the contracts: values, Document/Dataset, Task/schema, Result, Score, Run.
- `normalize/`   — central type-aware value canonicalization (NOT per-adapter, on purpose).
- `adapters/`    — the swappable part. `base.Pipeline` + one module per tool.
- `scorers/`     — pluggable metrics over (normalized prediction, ground truth).
- `engine/`      — executor (per-provider concurrency), cache, cost/budget, aggregation.
- `store/`       — SQLite for results, content-addressed blob store for raw artifacts.
- `config/`      — declarative experiment config (the backbone; version it).
- `cli/`         — `init / validate / run / score / compare / view`.
- `ui/`          — local viewer (leaderboard, per-doc drill-down, run diff).

## Install (adapter deps are optional extras — install per tool)

```
pip install -e .                  # core only
pip install -e ".[gemini]"        # + Gemini adapter
pip install -e ".[anthropic]"     # + Anthropic (Claude) adapter
pip install -e ".[openai]"        # + OpenAI adapter (also powers `openai_compatible`, see below)
pip install -e ".[extend]"        # + Extend adapter
pip install -e ".[llamaindex]"    # + LlamaIndex adapter
pip install -e ".[ui,dev]"        # viewer + dev tooling
```

## Quickstart

```
# fully local — no API keys (uses the deterministic `fake` adapter):
ezpz validate examples/experiments/fake_invoice_smoke.yaml
ezpz run      examples/experiments/fake_invoice_smoke.yaml
ezpz score    <run_id>                      # re-score cached extractions (fast, free)
ezpz compare  <run_id_a> <run_id_b>
ezpz view                                   # Streamlit viewer  (pip install -e ".[ui]")

# real 3-way LLM comparison (needs ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY):
ezpz run      examples/experiments/claude_vs_gemini_vs_openai.yaml --budget-usd 5
```

## Local & OpenAI-compatible models (`openai_compatible`)

The `openai_compatible` adapter points the OpenAI SDK at **any** OpenAI-compatible chat endpoint —
**Ollama, vLLM, LM Studio, Together, Groq, Fireworks, OpenRouter, DeepSeek, Azure OpenAI**, … — so
you can eval **local models with no cloud spend**, or any hosted gateway, on the same cohort as the
first-party adapters. It uses the same `[openai]` extra and the same extraction template; only
`base_url` (required) and `model` differ:

```yaml
- adapter: openai_compatible        # local, keyless (Ollama) — runs offline
  config: { base_url: http://localhost:11434/v1, model: llama3.1,
            prices: { input_per_1m: 0, output_per_1m: 0 } }
- adapter: openai_compatible        # hosted gateway — set api_key_env + real prices
  config: { base_url: https://api.together.xyz/v1, model: meta-llama/Llama-3.3-70B-Instruct-Turbo,
            api_key_env: TOGETHER_API_KEY }
```

Local servers ignore the API key (the adapter sends a placeholder); set `api_key_env` only for
hosted endpoints. See `examples/experiments/local_ollama.yaml`.

## Composite strategies (cascade · ensemble · verify)

Strategies are **meta-adapters** that orchestrate one or more inner pipelines — and they're
generic over the inner adapter, so any tier can be any registered adapter/model. Specify a tier as
`{adapter, config}`; mix providers freely.

- **`cascade`** — run a cheap tier first; escalate to a stronger tier only when confidence is low.
- **`ensemble`** — run N times and majority-vote (self-consistency).
- **`verify`** — extract, then have a second pass verify/correct.

```yaml
- adapter: cascade
  config:
    threshold: 0.8                 # escalate when the cheap tier's min per-field confidence < 0.8
    tiers:
      - { adapter: gemini,    config: { model: gemini-2.5-flash, self_rate: true, api_key_env: GEMINI_API_KEY } }
      - { adapter: anthropic, config: { model: claude-sonnet-4-6,                  api_key_env: ANTHROPIC_API_KEY } }
```

### `self_rate` — provider-neutral self-rated confidence

`self_rate: true` is a config flag on **any** LLM adapter (Gemini · Anthropic · OpenAI ·
LlamaIndex). It asks the model to emit a per-field confidence in `[0, 1]` alongside the values;
that confidence lands on `FieldValue.confidence` and is what a `cascade` gates on. The flag lives
in the shared LLM base, so it behaves identically across providers — swap the model or the vendor
and nothing else changes.

The score is the model's **own estimate**, a heuristic — not ground truth. Calibrate it before
trusting the gate: `ezpz analyze <run_id>` reports the escalation rate and accuracy-at-automation,
so you can pick a `threshold` that matches the accuracy you need.

```
ezpz run     examples/experiments/earnings_cascade.yaml   # haiku-only vs sonnet-only vs haiku→sonnet cascade
ezpz analyze <run_id>                                      # escalation rate + confidence calibration
```

## CI regression gating

`run` prints a pre-run cost estimate, emits machine-readable output, and returns meaningful exit
codes, so a holdout experiment can gate a build:

```
ezpz validate experiments/holdout.yaml                          # lint before spending (exit 1 on error)
ezpz run      experiments/holdout.yaml --budget-usd 5 --json --fail-under 0.85
#   exit 0: every pipeline ≥ 85% accuracy   ·   exit 2: a pipeline regressed below the gate
#   --budget-usd refuses to start if the estimate exceeds the cap, and stops early if actual
#   spend does.  Unchanged cells are cache hits (free), so re-runs cost nothing.
```

## Suggested build order

1. Lock the contracts (`core/` + `normalize/`).
2. One adapter end-to-end (Gemini) + deterministic scorers + SQLite + cache.
3. Second adapter (Extend) to stress the abstraction; fix leaks while cheap.
4. CLI run/score/compare + leaderboard view.
5. Third adapter (LlamaIndex), list/table scoring, LLM-judge, slices, CIs, budget guard.
6. CI/regression gating; QA task type if/when needed.

## Contributing

Issues and PRs welcome. Dev setup: `pip install -e ".[dev]"`. Before pushing, run `ruff check`,
`mypy src`, and `pytest` — CI runs all three on Python 3.10–3.12. New adapters are a `Pipeline`
subclass plus an `@register` decorator (run `ezpz init` for a template); they're discovered via the
`ezpz.adapters` entry-point group, so they live fine in your own package.

## License

[MIT](LICENSE) © 2026 Jon Moubayed.

The bundled example datasets under `examples/datasets/` are synthetic and authored for this repo —
no third-party documents are redistributed. Bring your own documents for real evaluations.
