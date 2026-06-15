# ezpz evals — Implementation Plan

A local-first harness for evaluating unstructured-document pipelines (structured extraction
today; retrieval/QA later) and comparing tools/models side by side on the same cohort.

This document is the source of truth for building ezpz evals. It is written to be executed
**milestone by milestone with Claude Code**.

---

## How to use this document with Claude Code

1. **Pin the invariants.** Copy [§2 Architectural Invariants](#2-architectural-invariants)
   into a `CLAUDE.md` at the repo root so the agent keeps them in context on every task.
   These are the rules that, if broken, silently make the whole tool lie.
2. **Work one milestone at a time.** Each milestone in [§10 Roadmap](#10-implementation-roadmap)
   has a task checklist and a **Definition of Done (DoD)**. Give the agent one milestone,
   let it implement, then verify against the DoD before moving on.
3. **Build the skeleton end-to-end first.** Milestone 1 wires the entire pipeline using a
   `FakePipeline` (no network, no cost). Do not start on real adapters until a fake run flows
   all the way into SQLite and a results table. This de-risks everything cheaply.
4. **Run checks after every milestone:** `pytest`, then `ezpz validate` and `ezpz run` on the
   example experiment. The DoD lists the exact commands.
5. **Verify volatile facts before pinning.** Anything in [§13 Open Decisions](#13-open-decisions--verify-before-pinning)
   (SDK package names, current model IDs, pricing) changes over time — have the agent check
   current vendor docs rather than trusting this file or its training data.
6. **The scaffold already exists.** A repo skeleton (directories + concrete contracts + stubbed
   modules with `NotImplementedError`) has been generated; see [Appendix E](#appendix-e-repository-tree).
   This plan fills it in.

---

## 1. Product Overview

**What we're building.** A CLI + local viewer that runs `Dataset × Pipeline × Task → scored,
normalized Results` and lets you compare pipelines. The motivating question: *should we extract
our documents with Extend.ai, with the Gemini API directly, or with LlamaIndex?* — answered with
real numbers on our own documents.

**Primary users.** Developers/ML engineers running locally. Single-user, local-first. No hosted
service, no auth, no multi-tenant concerns in v1.

**Core capabilities (v1).**
- Define a **cohort** of documents with ground truth.
- Define a **task** (a target extraction schema) once, tool-agnostic.
- Run N documents × M pipelines, where a pipeline is a swappable tool/model + config.
- Score with pluggable metrics; collect operational metrics (latency, cost, errors).
- Compare pipelines in a leaderboard + per-document drill-down + run diff.
- Cache aggressively; never pay twice for the same extraction.

**Non-goals (v1).**
- Hosted/multi-user deployment, RBAC, dashboards-as-a-service.
- Automatic tool integration/marketplace — adapters are written by hand, one per tool.
- Training/fine-tuning. We evaluate, we don't train.
- Labeling UI (ground truth is authored externally for now; model-assisted labeling is a later nicety).
- QA/RAG and classification task types are designed-for but not implemented in v1 (extraction only).

---

## 2. Architectural Invariants

> These are non-negotiable. Every component is shaped to protect them. If a change would
> violate one, the design is wrong — stop and reconsider.

1. **Two contracts, one canonical shape.**
   - *Input contract:* the task schema is authored once; each adapter **compiles** it into its
     own native format. Adapters never invent their own schema.
   - *Output contract:* each adapter **maps** its raw response into one canonical result shape
     (`ExtractionResult`). Adding a tool = writing one adapter; everything downstream is reused.

2. **Scoring, storage, and reporting touch ONLY the canonical shape — never raw tool output.**
   If any code path branches on "which tool produced this" during scoring, the abstraction has
   leaked and comparisons are no longer trustworthy. Raw responses are stored for debugging only.

3. **Value normalization is CENTRAL and type-aware — not per-adapter.**
   `"Jan 3 2024"` and `"2024-01-03"`, `"$1,200.00"` and `1200`, must compare equal regardless of
   which tool produced them. Normalization runs on predictions **and** ground truth, in one place
   (`ezpz.normalize`), after the adapter's structural `map` and before scoring.

4. **A pipeline's config fully determines its behavior.**
   `PipelineConfig.config` must capture everything that affects output (model id, prompt template,
   schema mode, parser, temperature, ...). The `config_hash` derived from it is the cache key and
   the pipeline's identity. Two runs with identical config + identical document must be identical.

5. **Cache the rawest durable artifact (the provider response); derive everything else on read.**
   This makes iterating on normalization and scoring **free** — you re-derive over cached raw
   responses without re-calling any API. You will tweak scorers far more often than extractions.

6. **The three ground-truth states are distinct and must never be conflated:**
   present-with-value, **correctly-absent** (`ABSENT` sentinel), and **not-yet-labeled** (omitted).
   A tool that correctly returns null for an absent field scores *correct*, not penalized. A tool
   that returns a value for an absent field is a **hallucination** (its own metric).

7. **Runs are immutable and reproducible.** A run pins dataset@version, task@version, pipeline
   configs, scorer set, and the environment (SDK versions). Aggregates are derived and always
   recomputable from raw per-field scores.

---

## 3. Glossary / Domain Model

| Term | Meaning |
|---|---|
| **Document** | One file + metadata + optional ground truth. Identity = content hash of bytes. |
| **Dataset** | A versioned collection of Documents, defined by a manifest (`manifest.jsonl`). |
| **Ground Truth (GT)** | The expected answer for a document, conforming to the task schema. |
| **Task** | What we're evaluating: a target schema (extraction) + the scorers that apply. |
| **FieldSpec** | One field in the task schema: name, type, required, description, scorers, etc. |
| **Pipeline / Adapter** | The swappable thing under test: a tool/model + config, wrapped behind one interface. |
| **Scorer** | A pure function comparing a normalized prediction to normalized GT → a score. |
| **Run / Experiment** | One immutable execution: `Dataset × Pipelines × Task × Scorers`. |
| **Slice** | A tag-defined subset of a dataset (e.g., `scanned`, `multi-page`) for per-cohort metrics. |
| **Result** | The canonical `ExtractionResult` for one (document, pipeline). |

---

## 4. Tech Stack & Repository Layout

**Language:** Python ≥ 3.10. The target tools' SDKs, Pydantic, and the eval ecosystem all live here.

**Core dependencies:** `pydantic>=2` (contracts/validation), `typer` + `rich` (CLI), `pyyaml`
(config), `sqlalchemy` or stdlib `sqlite3` (storage).

**Optional, per-tool extras (opt-in — integrations are not bundled):**
`gemini`, `extend`, `llamaindex`, `embeddings`, `ui` (streamlit + pandas), `dev` (pytest/ruff/mypy).

**Layout** (src-layout; see [Appendix E](#appendix-e-repository-tree) for the full tree):

```
src/ezpz/
  core/       # CONTRACTS (concrete): values, document, task, result, score, run
  normalize/  # central type-aware value canonicalization
  adapters/   # the swappable layer: base.Pipeline + registry + one module per tool
  scorers/    # base + registry + one module per metric
  engine/     # executor (per-provider concurrency), cache, cost/budget, aggregate
  store/      # sqlite + schema.sql + content-addressed blob store + migrations
  config/     # ExperimentConfig + YAML loader (the backbone)
  cli/        # init / validate / run / score / compare / view
  ui/         # local viewer (leaderboard / per-doc drill-down / run diff)
examples/     # invoice_extraction task, a 3-tool experiment, a sample dataset + GT
tests/        # contract + normalization + engine tests
```

---

## 5. Data Contracts

These are the load-bearing shapes. Implement them faithfully; everything else depends on them.
(They already exist as Pydantic models in `core/`; this section is the spec they must satisfy.)

### 5.1 Value types and canonical representation

Every `FieldSpec` declares a `ValueType`. The **canonical representation** is what scorers compare;
producing it is the job of `ezpz.normalize` (central), not the adapters.

| ValueType | Canonical representation | Normalization rule (be permissive in, strict out) |
|---|---|---|
| `string` | `str` | Unicode NFC; trim; collapse internal whitespace; `""` → `None`. |
| `integer` | `int` | Strip thousands separators/spaces; parse int; non-integer → unparseable. |
| `number` | `Decimal` | Strip currency symbols/separators; parse `Decimal`; beware locale decimal separators. |
| `boolean` | `bool` | `{true,yes,y,1,✓}`→True / `{false,no,n,0}`→False (case-insensitive). |
| `date` | ISO `YYYY-MM-DD` | Parse common formats; **ambiguous D/M vs M/D resolved by a configured `dayfirst`/locale** (see §13). |
| `datetime` | ISO 8601 | Include timezone if present; else treat as naive (document the choice). |
| `enum` | one of `enum_values` | Case-insensitive + alias map → canonical member; unknown → unparseable. |
| `currency` | `CurrencyValue{amount: Decimal, currency: str?}` | Amount per `number`; map symbol/code → ISO-4217. |
| `list` | `list[<item canonical>]` | Normalize each element by `FieldSpec.item`. |
| `object` | `dict[str, <field canonical>]` | Normalize each field by its `FieldSpec`. |

**Three distinct "empty" values** — keep them separate end to end:
- `None` → field missing / not extracted.
- `ABSENT` (sentinel, `"__ABSENT__"`) → field genuinely not present in the document (a *correct* answer when GT is also ABSENT).
- `UNPARSEABLE` (typed marker) → a value was returned but could not be canonicalized; scorers count these as parse failures rather than silently scoring them wrong.

### 5.2 `FieldValue` and `ExtractionResult` (the output contract)

```
FieldValue:
  value: Any                 # canonical representation (post-normalization)
  confidence: float | None   # None when the tool emits none — do not fabricate
  provenance: { page?, bbox? [x0,y0,x1,y1], text_span? } | None

ExtractionResult:
  doc_id: str
  pipeline_id: str           # == PipelineConfig.config_hash
  status: ok | partial | error
  fields: dict[str, FieldValue]
  raw_response_ref: str | None   # blob hash; raw response, kept for DEBUGGING ONLY
  timing: { latency_ms, stage_ms?: {compile, ingest, invoke, map} }
  cost:   { input_tokens?, output_tokens?, usd?, units?, raw: {...} }
  error:  { error_class: transport|parse|refusal|timeout|unknown, message? } | None
  extras: dict[str, Any]     # adapter-specific; scoring NEVER reads this
```

### 5.3 Ground-truth states × scorer handling

The matrix every value scorer must implement (alongside the `presence` scorer):

| GT state ↓ / Prediction → | has value | `None` (missed) | `UNPARSEABLE` |
|---|---|---|---|
| **present-value** | compare via the field's scorer → correct/wrong | wrong (wrongly-absent) | parse failure (counts wrong) |
| **`ABSENT`** | **hallucination** (separate metric + penalty) | **correct** (correctly-absent) | parse failure |
| **not-labeled** | excluded from scoring (cannot judge — not in denominator) | excluded | excluded |

### 5.4 `PipelineConfig` and identity

```
PipelineConfig:
  adapter: str                 # registered adapter name, e.g. "gemini"
  config: dict[str, Any]       # MUST fully determine behavior
  config_hash -> str           # sha256 over {adapter, config}, key-order independent; pipeline identity & cache key
```

### 5.5 Cache key

```
cache_key = sha256(doc_id + "|" + pipeline_config_hash + "|" + task_schema_hash)
```
A change to the prompt/model/parser changes `config_hash`; a change to the schema changes
`task_schema_hash`; either yields a natural cache miss for exactly what changed.

### 5.6 `ExperimentConfig` (the backbone; version these files)

```
ExperimentConfig:
  dataset: "name@version"
  task: "name@version"
  pipelines: [ PipelineConfig, ... ]      # the tools under test
  scorers: [ {name, config}, ... ]         # task defaults; per-field overrides win
  options: { concurrency, sample?, slices?, budget_usd?, force_refresh, samples_per_doc }
```

### 5.7 Adapter capability matrix

Adapters declare capabilities so the UI/scoring degrade gracefully (don't show a confidence
column for a tool that emits none; don't penalize missing provenance unless provenance is scored).

| Adapter | confidence | provenance | sync/async | cost unit | mapping difficulty |
|---|---|---|---|---|---|
| **gemini** | no (self-rated ≈ heuristic) | no (prompted; unreliable) | sync | tokens → USD | medium — you own JSON parse/repair |
| **extend** | yes | yes (bbox) | async (submit→poll→fetch) | per page → USD | easy — already field-keyed |
| **llamaindex** | no | no | sync | tokens (+ parse cost) | medium — Pydantic instance → fields |

---

## 6. Component Specifications

Each spec follows the same template: **Responsibility / Interface / Key decisions / Watch-outs / Done when.**

### 6.1 `core/` — the contracts

- **Responsibility:** Define the canonical models (§5) with Pydantic. No business logic beyond
  hashing and validation.
- **Interface:** `Document`, `Dataset`, `GroundTruth`, `ABSENT`; `Task`, `FieldSpec`, `ScorerRef`,
  `TaskType`, `ValueType`, `CurrencyValue`; `ExtractionResult`, `FieldValue`, `ResultStatus`,
  `Provenance`, `Timing`, `Cost`, `ErrorInfo`; `FieldScore`, `MetricSummary`; `Run`,
  `PipelineConfig`, `RunOptions`, `RunStatus`.
- **Key decisions:** `config_hash` and `Task.schema_hash` are computed via canonical JSON
  (`sort_keys=True`) so they are stable across key ordering. `FieldSpec` is recursive (`item`,
  `fields`) to support nested lists/objects.
- **Watch-outs:** Do not put normalization here. Keep `extras` free-form and never read it downstream.
- **Done when:** Models import and validate; `config_hash` is deterministic and order-independent
  (covered by `tests/test_contracts.py`).

### 6.2 `normalize/` — central value canonicalization

- **Responsibility:** Turn raw values (from adapters) and GT values into the canonical
  representation per §5.1. Same code path for predictions and GT.
- **Interface:**
  ```
  normalize_value(raw: Any, spec: FieldSpec) -> Any            # dispatch on spec.type
  normalize_fields(raw_fields: dict, specs: list[FieldSpec]) -> dict
  ```
- **Key decisions:** Permissive input, strict output. Recurse for `list`/`object`. Preserve the
  `None` / `ABSENT` / `UNPARSEABLE` distinction exactly.
- **Watch-outs:** Date ambiguity (D/M/Y) and locale decimal separators are the classic bugs —
  drive them from config, never guess silently. Currency symbol→ISO-4217 map must be explicit.
- **Done when:** `tests/test_normalize.py` passes, including: `"$1,200.00"` and `"1200"` →
  equal `CURRENCY`; `"Jan 3 2024"` and `"2024-01-03"` → equal `DATE`; `ABSENT` stays distinct
  from `None`; an ungarbleable value yields `UNPARSEABLE`, not a wrong-but-parsed value.

### 6.3 `adapters/base.py` — the `Pipeline` ABC

- **Responsibility:** Define the adapter interface and orchestrate the five stages; map errors
  into the canonical result. **No provider logic.**
- **Interface:**
  ```
  class Pipeline(ABC):
      capabilities: Capabilities
      compile(task) -> prepared          # task schema -> native schema/prompt (cacheable per (task,config))
      ingest(document) -> ingested       # bytes / upload handle / parsed text
      invoke(prepared, ingested) -> (raw, Cost)   # call backend; retries/timeouts/polling inside
      map(raw, task) -> dict[str, FieldValue]     # structural map (pre value-normalization)
      run(document, task) -> ExtractionResult     # SHARED: sequences stages, times, error-wraps
  ```
- **Key decisions:** `run()` is a template method (already implemented in the scaffold). Value
  normalization happens **after** `map`, in the engine, so it's applied identically to every
  adapter. `pipeline_id == config.config_hash`.
- **Watch-outs:** `run()` must classify exceptions into `transport|parse|refusal|timeout`
  (the scaffold currently tags `unknown` — improve this). A hard failure must produce
  `status=error`, not silently-empty fields.
- **Done when:** A `FakePipeline` subclass (deterministic, no network) runs through `run()` and
  yields a valid `ExtractionResult` with timing populated.

### 6.4 `adapters/gemini.py` — Gemini direct

- **Responsibility:** Own the full prompt → call → parse loop against the Google GenAI API.
- **Key decisions:**
  - `compile`: build a response schema (JSON mode / structured output) from the task schema, plus
    a prompt embedding each field's `description`. The prompt template is part of `config` and is
    versioned — so prompt A vs prompt B are two separate, comparable pipelines.
  - `ingest`: load file bytes; choose inline vs the Files API by size/page count.
  - `invoke`: call the model with retries + timeout; capture reported token usage; compute
    `Cost.usd` from a price table (see §13).
  - `map`: parse the JSON; **repair** common malformations; if unrecoverable, return `partial`
    or `error` with `error_class=parse`.
- **Watch-outs:** No native per-field confidence — leave `confidence=None` (a self-rated field is
  a heuristic; if used, mark it in `extras`, never treat as ground-truthy). Provenance generally
  absent. Results are prompt-sensitive — that's expected and is why prompt is in the config.
- **Done when:** Real extraction on the sample dataset produces a normalized, scored, cached result.

### 6.5 `adapters/extend.py` — Extend.ai

- **Responsibility:** Drive Extend's managed extraction (schema → structured fields + confidence + bbox).
- **Key decisions:**
  - `compile`: translate the task schema into Extend's field/schema config; **fail loudly** if a
    canonical type can't be expressed rather than silently degrading.
  - `ingest`/`invoke`: job-based — submit, **poll to completion**, fetch. Hide the async behind
    the synchronous stage interface. `Cost.units = pages`.
  - `map`: the most direct of the three — rename/type its fields; carry confidence and bbox
    straight into `FieldValue`.
- **Watch-outs:** Polling backoff and job timeouts; per-page billing means `Cost.usd` may be
  derived from a per-page rate, not tokens.
- **Done when:** The same task runs across Gemini + Extend with comparable normalized output; any
  contract leak surfaced here is fixed in `core`/`normalize`, not patched in the adapter.

### 6.6 `adapters/llamaindex.py` — LlamaIndex

- **Responsibility:** Compose a parser + a Pydantic structured-extraction program.
- **Key decisions:**
  - `compile`: build a Pydantic output class + extraction program from the task schema.
  - `ingest`: **parse first** (LlamaParse or a reader) → text/markdown. Parse quality is part of
    what you're evaluating here — note that this is a parse+extract pipeline, vs end-to-end for
    Extend/Gemini.
  - `invoke`: run the program; sum parse cost + backend-LLM cost.
  - `map`: read fields off the Pydantic instance.
- **Watch-outs:** Many sub-choices (parser, chunking, backend model) each move results — encode
  **all** of them in `config` so the `config_hash` is meaningful. When the QA task type lands, the
  retrieval engine is the natural fit and results gain `retrieved_contexts`.
- **Done when:** Three-way comparison (Gemini × Extend × LlamaIndex) runs on the example task.

### 6.7 `scorers/` — pluggable metrics

- **Responsibility:** Compare one normalized prediction to its normalized GT and emit a
  `FieldScore`. Pure functions; type- and config-aware via `ScoreContext(spec, config)`.
- **Interface:**
  ```
  class Scorer(ABC):
      name: str
      score(prediction, ground_truth, ctx: ScoreContext) -> FieldScore   # value in [0,1] + passed
  ```
- **Catalog (one module each):**

  | Scorer | Applies to | Config | Returns | Notes |
  |---|---|---|---|---|
  | `exact` | all | – | 0/1 | baseline; IDs, enums, structured values |
  | `numeric_tolerance` | integer/number/currency | `abs`, `rel` | 0/1 | `|p−g| ≤ abs or rel·|g|` |
  | `date_match` | date/datetime | `tolerance_days` | 0/1 | compare ISO; optional day window |
  | `string_similarity` | string | `threshold`, `method` | sim∈[0,1] + pass | names/addresses |
  | `embedding_similarity` | string | `model`, `threshold` | cosine + pass | free text; **cache embeddings** |
  | `llm_judge` | string/object/any | `model`, `prompt`, `rubric` | score + rationale | **cache**; calibrate; bias-aware |
  | `list_table` | list-of-object | `match_key`/strategy + per-field scorers | P/R/F1 + field scores | see algorithm below |
  | `presence` | all | – | correct-absent / wrongly-absent / hallucination | run alongside the value scorer |

- **`list_table` algorithm (the hard one):**
  1. **Align** predicted rows to GT rows using `FieldSpec.match_key` (exact/fuzzy on the key) or
     an optimal assignment (Hungarian) when no clean key exists.
  2. **Score matched pairs** field-by-field, delegating to each sub-field's scorers.
  3. **Report set precision/recall/F1** for unmatched rows (extra = FP/hallucinated rows;
     missing = FN). Surface *both* "found the right rows" and "the found rows are correct."

- **`llm_judge` discipline:** pin model + prompt + version in config (part of the task's identity);
  calibrate against a human-labeled subset; watch verbosity/position bias; cache on
  `hash(prediction + ground_truth + judge_config)`; treat scores as noisier than deterministic ones.

- **Operational metrics are NOT scorers** — latency, cost, tokens, error rate, %repair,
  %low-confidence come from run/result metadata and need no GT. Compute them in `engine/aggregate`.

- **Done when:** `exact` + `numeric_tolerance` pass unit tests in M1; the full catalog including
  `list_table` and `presence` passes in M4 against the invoice example.

### 6.8 `engine/executor.py`

- **Responsibility:** Run the `docs × pipelines` grid; per cell: cache → (miss) `pipeline.run`
  → cache raw → normalize → score → store.
- **Key decisions:** Bound concurrency **per provider** (each backend has its own limits), not
  globally — asyncio + a per-provider semaphore; sync-only SDKs run in a thread executor (hidden
  behind the adapter). Retries with exponential backoff + jitter on transient errors;
  circuit-break a provider after repeated failures. **Resumable** (completed cells are cached).
  Emit live progress (n/total, cost-so-far, error count).
- **Watch-outs:** Don't let one dead provider stall the whole run. Respect `samples_per_doc > 1`
  for variance measurement (LLMs are nondeterministic even at temp 0).
- **Done when:** A run schedules across pipelines concurrently, resumes from cache on re-invoke,
  and persists results + scores.

### 6.9 `engine/cache.py`

- **Responsibility:** Content-addressed cache of **raw provider responses** keyed by §5.5.
- **Key decisions:** Cache the rawest artifact; derive normalized fields on read so normalization
  fixes are free. Separate, smaller cache for `llm_judge`/embedding scores. `force_refresh`
  bypasses; provide inspect/prune.
- **Done when:** A re-run with unchanged config is a full cache hit (no API calls); changing only
  the prompt re-extracts only that pipeline.

### 6.10 `engine/cost.py`

- **Responsibility:** Pre-run cost estimate + a budget guard.
- **Key decisions:** Estimate = (uncached cells) × per-pipeline per-doc priors; subtract cached
  work via the cache. `BudgetGuard` confirms before start and **stops early** if actuals exceed
  `budget_usd`. Normalize cost to USD where possible but keep native units (pages/tokens) honest.
- **Done when:** `ezpz run --budget-usd N` refuses/aborts past the cap and prints an estimate first.

### 6.11 `engine/aggregate.py`

- **Responsibility:** Aggregate stored `FieldScore`s and operational metrics into summaries.
- **Key decisions:** Per-field accuracy; **macro vs micro** averaging made explicit (they answer
  different questions); per-slice breakdowns; **bootstrap confidence intervals + n**; paired
  comparison between two pipelines on the same docs with a within-noise flag. Optional
  **calibration** metric when a tool emits confidence (accuracy at a given automation rate).
- **Done when:** The leaderboard shows metrics with CIs and slice toggles that match hand-checks.

### 6.12 `store/` — SQLite + blobs

- **Responsibility:** Persist runs/results/scores (SQLite, schema in `schema.sql`); store large
  artifacts (raw responses, parsed text, page images, uploads) in a content-addressed blob dir;
  tables hold blob hashes, not bytes.
- **Key decisions:** Versioned migrations so old run DBs still open. `export(run_id)` →
  JSON/CSV/Parquet. Blobs deduped by content hash.
- **Watch-outs:** Keep the DB lean; never inline blobs. Wrap writes in transactions.
- **Done when:** A run round-trips: persisted, listed, loaded, and re-scored from storage.

### 6.13 `config/experiment.py`

- **Responsibility:** Load + validate `ExperimentConfig` from YAML (§5.6).
- **Done when:** `ExperimentConfig.load_yaml(examples/experiments/...yaml)` validates and round-trips.

### 6.14 `cli/main.py`

- **Responsibility:** `init` (scaffold a project) · `validate` (lint before spending money) ·
  `run` (execute an experiment) · `score` (re-score cached extractions; fast/free) · `compare`
  (diff two runs) · `view` (launch the viewer).
- **Key decisions:** `run` emits a compact pipelines×metrics matrix on completion + machine-readable
  output + sensible exit codes, enabling **CI regression gating** (fail the build if holdout
  accuracy drops > X vs a baseline run).
- **Done when:** All verbs work against the example experiment; `validate` catches a bad schema,
  missing GT, or absent secret before any API call.

### 6.15 `ui/app.py` — local viewer (Streamlit to start)

- **Responsibility:** Read SQLite (never re-run) and present three views: **leaderboard**
  (pipelines×metrics, slice toggles, CI indicators), **per-document drill-down** (columns per
  pipeline of predicted vs GT, color-coded correct/wrong/missing/hallucinated, confidence where
  available, rendered source page alongside), and **run diff** (which fields/docs improved or
  regressed). Plus a failure explorer.
- **Key decisions:** Streamlit/Gradio for speed; graduate to FastAPI + a small frontend only if
  the side-by-side UX justifies it. Because the UI only reads SQLite, the framework can be swapped
  without touching the engine.
- **Done when:** A completed run is explorable: leaderboard, drill into a doc, diff two runs.

---

## 7. End-to-End Execution Flow

**Single (document, pipeline) cell:**
```
compile(task) ─┐
ingest(doc) ───┼─► invoke() ─► raw + cost ─► [cache.put raw] ─► map() ─► fields(FieldValue)
               │                                                          │
            (cache.get hit short-circuits here) ◄──────────────────────  ▼
                                                  normalize_fields(fields, task.fields)
                                                                          │
                                                                          ▼
                                          for each field: applicable scorers → FieldScore
                                                                          │
                                                                          ▼
                                                  store.save_result + store.save_scores
```

**Full run:** load Dataset@version + Task@version + ExperimentConfig → estimate cost + confirm
budget → schedule the `docs × pipelines` grid (per-provider concurrency, retries, circuit-breaker,
resumable) → as cells complete, normalize + score + persist → `aggregate` → print matrix → `view`.

---

## 8. Caching & Cost Model (depth)

- **Why it's the highest-value piece:** local tool + paid APIs. Without caching you re-pay on
  every iteration; with it, your tightest loop (tweaking scorers/normalization) costs nothing.
- **What to cache:** the raw provider response, keyed by `sha256(doc_id|config_hash|schema_hash)`.
  Derive normalized fields and scores on read. Fixing a `map`/`normalize` bug then costs zero API calls.
- **Invalidation falls out of hashing:** change prompt/model/parser → new `config_hash` → miss only
  for that pipeline; change schema → new `schema_hash` → miss only where the schema changed.
- **Second cache layer:** `llm_judge` and embedding calls (cheap but not free) keyed on their inputs.
- **Cost normalization:** Extend bills per page, Gemini/LlamaIndex per token. Compute `Cost.usd`
  via per-provider price tables (see §13) **and** keep native units (`Cost.units`, `Cost.raw`)
  so comparisons stay honest rather than forcing a fake single unit.
- **Budget guard:** estimate before running; stop early past `budget_usd`.

---

## 9. Scoring & Metrics (depth)

- **GT states drive correctness** (see §5.3 matrix). Hallucination (value for an `ABSENT` field)
  is its own metric — for extraction it's a dangerous failure that raw accuracy hides.
- **List/table** is the hardest case — alignment then per-field scoring then set P/R/F1 (§6.7).
- **Aggregation:** be explicit about macro vs micro; always break down by slice; attach n + bootstrap
  CIs so a 3-doc difference isn't over-read; support paired comparison on the same docs.
- **Calibration (advanced):** if a tool emits confidence, measure whether 0.9 means 90% correct.
  This reframes "raw accuracy" into "accuracy at a chosen automation rate" — usually the real
  business question if you plan to auto-accept high-confidence fields and human-review the rest.
- **Operational metrics are often decisive:** a tool 2% more accurate at 5× cost and 3× latency
  usually loses. Always show cost/doc, p50/p95 latency, and error rate next to accuracy.

---

## 10. Implementation Roadmap

Each milestone is a unit of work for Claude Code. Implement, then verify against the DoD before
proceeding. Milestones build strictly on prior ones.

### M0 — Solidify contracts & project setup
- [ ] Finalize `core/` models (§5); ensure `config_hash`/`schema_hash` are deterministic.
- [ ] Implement `config/experiment.py` YAML loader.
- [ ] Implement `store/schema.sql` + `SqliteStore.init_db()` (create tables) and `BlobStore.put/get`.
- [ ] Implement the adapter + scorer **registries** (already concrete in scaffold; verify).
- [ ] Add `FakePipeline` in `tests/` (deterministic, no network) + `tests/test_contracts.py`.
- **DoD:** `pytest tests/test_contracts.py` passes; `SqliteStore.init_db()` creates the DB; a
  `FakePipeline` produces a valid `ExtractionResult`.

### M1 — End-to-end vertical slice (FakePipeline, no network, no cost)
- [ ] Implement `normalize/canonical.py` for at least `string`, `number`, `currency`, `date`,
      preserving `None`/`ABSENT`/`UNPARSEABLE`.
- [ ] Implement `exact` and `numeric_tolerance` scorers (with GT-state handling).
- [ ] Implement a minimal `engine/executor.py`: per-cell cache→run→normalize→score→store (serial is fine here).
- [ ] Implement `engine/cache.py` (`cache_key` + raw-response cache over the blob store).
- [ ] Wire `cli` `run` and `score`; print a compact pipelines×metrics matrix.
- **DoD:** `ezpz run` on a tiny **fake** dataset writes scored results to SQLite and prints a
  matrix; a second `ezpz run` is a full cache hit (no recompute of extractions); `ezpz score
  <run_id>` re-scores from cache without re-extracting; `pytest tests/test_normalize.py` passes.

### M2 — First real adapter: Gemini
- [ ] Implement `adapters/gemini.py` (`compile`/`ingest`/`invoke`/`map`) with JSON-mode + repair.
- [ ] Compute `Cost.usd` from reported usage (price table in config).
- [ ] Complete `normalize/` for all `ValueType`s used by the example task.
- [ ] Error classification in `Pipeline.run` (`transport|parse|refusal|timeout`).
- **DoD:** `ezpz run` extracts the sample invoices via Gemini, normalizes, scores, and caches;
  re-running hits cache; a malformed response yields `partial`/`error`, not silent wrong fields.

### M3 — Second adapter: Extend (stress the abstraction)
- [ ] Implement `adapters/extend.py` incl. submit→poll→fetch and `Cost.units = pages`.
- [ ] Map confidence + bbox provenance into `FieldValue`.
- [ ] Fix any contract leaks discovered (fix in `core`/`normalize`, not in the adapter).
- **DoD:** the same task runs across Gemini + Extend with comparable normalized output; the
  leaderboard compares them on accuracy, cost, and latency.

### M4 — Scoring depth & aggregation
- [ ] Implement `presence`, `string_similarity`, `embedding_similarity` (cached), `date_match`.
- [ ] Implement `list_table` (alignment + per-field scoring + set P/R/F1).
- [ ] Implement `llm_judge` (cached, pinned config) — optional but recommended.
- [ ] Implement `engine/aggregate.py`: macro/micro, slices, bootstrap CIs, paired compare.
- **DoD:** the invoice example (with its `line_items` list and `ABSENT` `po_number`) scores
  sensibly; hallucinations and missed rows are visible; aggregates carry CIs and slice breakdowns.

### M5 — Third adapter: LlamaIndex
- [ ] Implement `adapters/llamaindex.py` (parser + Pydantic extraction program); sum parse+LLM cost.
- **DoD:** a three-way comparison (Gemini × Extend × LlamaIndex) runs on the example task.

### M6 — Viewer
- [ ] Implement `ui/app.py`: leaderboard, per-document drill-down (with rendered source page),
      run diff, failure explorer.
- **DoD:** `ezpz view` opens; you can read the leaderboard, drill into a document side-by-side,
  and diff two runs.

### M7 — Hardening & CI
- [ ] Per-provider concurrency + retries + circuit breaker in the executor.
- [ ] `BudgetGuard` (estimate + stop-early) wired into `run`.
- [ ] Resumability from cache; `store.export`; schema migrations.
- [ ] `validate` lints datasets/tasks/experiments fully; `run` exposes machine-readable output +
      exit codes; document a CI regression-gating recipe.
- **DoD:** a real multi-tool run is safe, resumable, budget-capped, and gateable in CI.

---

## 11. Testing Strategy

- **`FakePipeline`** (deterministic, no network) is the backbone of fast tests — it lets the
  engine, cache, scorers, store, and aggregation be tested end-to-end without cost or SDKs.
- **Normalization tests** are the highest-leverage: comparability is won or lost here. Cover
  currency/date/enum canonicalization and the `None`/`ABSENT`/`UNPARSEABLE` distinction.
- **Scorer unit tests** per scorer, including every GT-state cell from §5.3 and the `list_table`
  alignment edge cases (extra rows, missing rows, reordered rows, fuzzy key matches).
- **Contract tests:** `config_hash`/`schema_hash` determinism and order-independence.
- **Cache tests:** identical config → full hit; changed prompt → miss only for that pipeline.
- **Golden mini-dataset:** a handful of fixtures with known GT, run through `FakePipeline`, with a
  snapshot of expected aggregates — guards against scoring regressions.
- **Adapter tests** hit the network and cost money → mark them and keep them out of the default run.

---

## 12. Conventions

- **Style/tooling:** `ruff` (lint+format), `mypy` (types), `pytest`. Type-annotate public interfaces.
- **Errors:** classify into `transport|parse|refusal|timeout|unknown`; a failed extraction is
  `status=error`, never silently-empty fields. Surface reliability as its own metric.
- **Secrets:** referenced by env var name in config (`api_key_env: GEMINI_API_KEY`); read from the
  environment / `.env`; never inline keys in YAML; never log secrets.
- **Determinism:** temperature 0 by default; expose seeds where supported; `samples_per_doc > 1`
  for variance studies. Record SDK/tool versions in `Run.env`.
- **Provider isolation:** all SDK calls live inside the adapter module — contain SDK churn there.
- **No raw-output leakage:** scoring/reporting must never branch on the producing tool (Invariant #2).
- **Logging:** structured, per attempt; live progress on `run` (n/total, cost-so-far, errors).

---

## 13. Open Decisions / Verify Before Pinning

> Have Claude Code confirm these against **current** vendor docs; do not trust stale values.

- **SDK package names & versions:** Google GenAI SDK (`google-genai`?), Extend (official SDK vs
  REST via `httpx`), `llama-index` / `llama-parse`. Pin only after verifying on PyPI/docs.
- **Current model IDs:** the exact Gemini model identifiers (the example uses placeholders).
- **Price tables** for USD cost normalization (per-token for Gemini/LLM backends; per-page for
  Extend). Centralize in config so they're easy to update.
- **Date locale / `dayfirst`** policy for ambiguous dates; **decimal separator** locale for numbers.
- **Async framework:** `asyncio` vs `anyio`; thread-executor strategy for sync-only SDKs.
- **Embedding model** for `embedding_similarity`; **judge model + prompt** for `llm_judge`.
- **Structured-output mode** specifics for Gemini (response schema vs tool-calling).
- **Storage layer:** raw `sqlite3` vs SQLAlchemy (the scaffold lists SQLAlchemy as a dep; either is fine).

---

## Appendix A — Example task (`examples/tasks/invoice_extraction.yaml`)

```yaml
name: invoice_extraction
version: "1"
type: extraction
instructions: >
  Extract billing fields from the invoice. If a field is not present in the document,
  return null rather than guessing.
scorers:
  - name: exact
fields:
  - { name: invoice_number, type: string, required: true, scorers: [{ name: exact }] }
  - { name: invoice_date, type: date, required: true, scorers: [{ name: date_match, config: { tolerance_days: 0 } }] }
  - { name: vendor_name, type: string, required: true, scorers: [{ name: string_similarity, config: { threshold: 0.9 } }] }
  - { name: total_amount, type: currency, required: true, scorers: [{ name: numeric_tolerance, config: { abs: 0.01 } }] }
  - { name: po_number, type: string, required: false, scorers: [{ name: presence }, { name: exact }] }
  - name: line_items
    type: list
    required: true
    match_key: description
    item:
      name: line_item
      type: object
      fields:
        - { name: description, type: string,   scorers: [{ name: string_similarity, config: { threshold: 0.85 } }] }
        - { name: quantity,    type: number,   scorers: [{ name: numeric_tolerance, config: { abs: 0.0 } }] }
        - { name: unit_price,  type: currency, scorers: [{ name: numeric_tolerance, config: { abs: 0.01 } }] }
    scorers: [{ name: list_table }]
```

## Appendix B — Example experiment (`examples/experiments/extend_vs_gemini_vs_llamaindex.yaml`)

```yaml
dataset: invoices_v1@1
task: invoice_extraction@1
pipelines:
  - { adapter: extend, config: { api_key_env: EXTEND_API_KEY } }
  - adapter: gemini
    config: { model: gemini-2.5-pro, api_key_env: GEMINI_API_KEY, schema_mode: json, prompt_template: default, temperature: 0 }
  - adapter: gemini   # a second config = a separate, comparable pipeline
    config: { model: gemini-2.5-flash, api_key_env: GEMINI_API_KEY, schema_mode: json, prompt_template: default, temperature: 0 }
  - adapter: llamaindex
    config: { parser: llamaparse, backend_model: gpt-4o-mini, api_key_env: LLAMA_CLOUD_API_KEY }
options:
  concurrency: 4
  budget_usd: 5
  samples_per_doc: 1
```

## Appendix C — Example ground truth (one file per doc)

```json
{
  "doc_id": "REPLACE_WITH_CONTENT_HASH",
  "fields": {
    "invoice_number": "INV-10231",
    "invoice_date": "2024-01-15",
    "vendor_name": "Acme Corporation",
    "total_amount": { "amount": "1284.50", "currency": "USD" },
    "po_number": "__ABSENT__",
    "line_items": [
      { "description": "Widget A", "quantity": 10, "unit_price": { "amount": "12.00", "currency": "USD" } },
      { "description": "Setup fee", "quantity": 1,  "unit_price": { "amount": "99.00", "currency": "USD" } }
    ]
  }
}
```

## Appendix D — Canonical result shape (what every adapter must emit, post-normalization)

```json
{
  "doc_id": "ab12...",
  "pipeline_id": "9f3c1a2b4d5e6f70",
  "status": "ok",
  "fields": {
    "invoice_number": { "value": "INV-10231", "confidence": 0.98, "provenance": { "page": 1, "bbox": [0.1,0.1,0.3,0.13] } },
    "total_amount":   { "value": { "amount": "1284.50", "currency": "USD" }, "confidence": null, "provenance": null },
    "po_number":      { "value": null, "confidence": null, "provenance": null }
  },
  "raw_response_ref": "blob:7c9e...",
  "timing": { "latency_ms": 2310.4, "stage_ms": { "compile": 1.2, "ingest": 40.0, "invoke": 2250.1, "map": 19.1 } },
  "cost": { "input_tokens": 5120, "output_tokens": 240, "usd": 0.0123, "units": null, "raw": {} },
  "error": null,
  "extras": {}
}
```

## Appendix E — Repository tree

```
ezpz-evals/
├── README.md
├── pyproject.toml          # adapter deps as OPTIONAL extras: .[gemini]/.[extend]/.[llamaindex]/.[ui]/.[dev]
├── .env.example
├── src/ezpz/
│   ├── core/        values.py document.py task.py result.py score.py run.py
│   ├── normalize/   canonical.py
│   ├── adapters/    base.py registry.py gemini.py extend.py llamaindex.py
│   ├── scorers/     base.py registry.py exact.py numeric_tolerance.py date_match.py
│   │                string_similarity.py embedding_similarity.py llm_judge.py list_table.py presence.py
│   ├── engine/      executor.py cache.py cost.py aggregate.py
│   ├── store/       sqlite.py schema.sql blobs.py migrations/
│   ├── config/      experiment.py
│   ├── cli/         main.py
│   └── ui/          app.py
├── examples/
│   ├── tasks/invoice_extraction.yaml
│   ├── experiments/extend_vs_gemini_vs_llamaindex.yaml
│   └── datasets/invoices_v1/   manifest.jsonl  ground_truth/  docs/
└── tests/           test_contracts.py test_normalize.py conftest.py
```

---

*End of plan. Implement milestone by milestone; keep §2 invariants pinned; verify §13 before pinning versions.*
