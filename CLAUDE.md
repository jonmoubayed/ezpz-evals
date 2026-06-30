# CLAUDE.md — ezpz evals

Local-first harness for evaluating unstructured-document pipelines (structured extraction now;
QA/RAG later) and comparing tools/models on the same cohort.
**Full spec:** `IMPLEMENTATION_PLAN.md` — read it; this file is the always-on distillation.

## How to work in this repo
- Implement **milestone by milestone** (PLAN §10). Do one milestone, satisfy its Definition of
  Done, then stop for review. Don't jump ahead.
- After any change: run `pytest`, then `ruff check`. Before calling a milestone done, run that
  milestone's DoD commands.
- Don't add dependencies casually — adapter SDKs are **optional extras** (see `pyproject.toml`).
- For any vendor's current SDK name, model id, or pricing: **check current docs**. Do not trust
  training data or the placeholders in the plan (PLAN §13 lists what to verify).

## Cardinal invariants — do NOT break these
1. **Two contracts.** The task schema is authored once; each adapter `compile`s it to its native
   format and `map`s its raw response into the canonical `ExtractionResult`. Adapters never invent a schema.
2. **Scoring/storage/reporting read ONLY the canonical shape** — never raw tool output, and never
   branch on *which tool* produced a result. Tempted to special-case a tool in a scorer? The
   abstraction has leaked — fix the contract instead.
3. **Normalization is central + type-aware** (`ezpz.normalize`), applied to predictions AND ground
   truth, after `map` and before scoring. Never normalize inside an adapter.
4. **`PipelineConfig.config` must fully determine behavior** (model, prompt, schema mode, parser,
   temperature…). `config_hash` is the pipeline's identity and the cache key.
5. **Cache the rawest artifact** (the provider response); derive normalized fields + scores on read,
   so re-scoring costs zero API calls.
6. **Three ground-truth states are distinct:** present-value, `ABSENT` (correctly absent),
   not-labeled. Correct null for an `ABSENT` field = correct. A value for an `ABSENT` field =
   hallucination (its own metric). Never conflate them.
7. **Runs are immutable + reproducible**; aggregates are derived and recomputable from raw `FieldScore`s.

## Commands
- Install: `pip install -e ".[dev]"`  — add `.[gemini]` / `.[extend]` / `.[llamaindex]` / `.[ui]` as needed
- Test: `pytest`  ·  Lint/format: `ruff check` / `ruff format`  ·  Types: `mypy src`
- Validate before spending money: `ezpz validate <experiment.yaml>`
- Run: `ezpz run examples/experiments/extend_vs_gemini_vs_llamaindex.yaml --budget-usd 5`
- Re-score a cached run (free): `ezpz score <run_id>`  ·  Compare: `ezpz compare <a> <b>`  ·  View: `ezpz view`

## Repo map (`src/ezpz/`)
- `core/` — contracts (Pydantic): values, document, task, result, score, run. **Load-bearing; change deliberately.**
- `normalize/` — central value canonicalization.
- `adapters/` — the swappable layer: `base.Pipeline` (stages `compile`/`ingest`/`invoke`/`map` + shared `run`)
  plus one module per tool. **All provider SDK calls live here** (contains SDK churn).
- `scorers/` — pluggable metrics over (normalized prediction, GT). `list_table` and `presence` are the tricky ones.
- `engine/` — executor (per-provider concurrency, retries, resumable), cache, cost/budget, aggregate (macro/micro, slices, CIs).
- `store/` — SQLite (`schema.sql`) + content-addressed blob store; **tables hold blob hashes, not bytes**.
- `config/` — `ExperimentConfig` + YAML loader (the backbone; version experiment files).
- `cli/` — init/validate/run/score/compare/view.  ·  `ui/` — local viewer: `server.py` (stdlib HTTP,
  no deps) serves `static/index.html` (the SPA) over `data.py` view-models (read-only) — leaderboard
  (+ per-field heatmap), documents, diff, failures, analyze (+ cost×accuracy Pareto); `/api/export`
  downloads a run, URL-synced view state. The **one** write path is `launch.py` (`POST /api/run`):
  the budget modal re-runs an experiment, budget-gated (estimate → refuse if over cap → background
  thread, cooperatively cancellable via `POST /api/run/cancel`), reconstructed from the stored run.

## Conventions
- Errors classified `transport|parse|refusal|timeout|unknown`; a failed extraction is `status=error`,
  **never silently-empty fields**. Reliability is its own metric.
- Secrets via env-var **name** in config (`api_key_env: GEMINI_API_KEY`), read from env/`.env`; never inline or log keys.
- Temperature 0 by default; record SDK/tool versions in `Run.env`.
- Type-annotate public interfaces. Keep `extras` free-form and never read it downstream.

## Build-order reminder
Build the whole pipeline end-to-end with a `FakePipeline` (no network, no cost) **first** (M1),
then swap in real adapters: **Gemini → Extend → LlamaIndex**. Don't start real adapters until a
fake run flows into SQLite and prints a results matrix.