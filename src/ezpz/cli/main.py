"""ezpz CLI. `run` executes an experiment; `score` re-scores cached extractions (fast/free);
`validate` lints before spending money; `view` launches the local viewer.

Machine-readable output + exit codes make `run` usable in CI for regression gating
(fail the build if holdout accuracy drops > X vs a baseline run)."""
from __future__ import annotations

import json
import os
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

from ezpz.adapters import registry as adapter_registry
from ezpz.adapters.registry import get_adapter
from ezpz.config.dotenv import load_dotenv
from ezpz.config.experiment import ExperimentConfig, resolve_dataset, resolve_task
from ezpz.core.document import GroundTruth
from ezpz.core.result import ResultStatus
from ezpz.core.run import Run, RunStatus
from ezpz.engine.aggregate import calibration, extras_rate, paired_compare, pipeline_summary
from ezpz.engine.cache import RawResponseCache, cache_key
from ezpz.engine.cost import estimate_cost
from ezpz.engine.executor import Executor, normalize_mapped, score_fields
from ezpz.plugins import import_modules, load_plugins
from ezpz.scorers import registry as scorer_registry
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore

app = typer.Typer(add_completion=False, help="ezpz evals — local document-pipeline eval harness")


@app.callback()
def _bootstrap() -> None:
    """Load secrets (.env) + plugin adapters/scorers (entry points) before any command runs."""
    load_dotenv()
    load_plugins()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store_and_cache() -> tuple[SqliteStore, RawResponseCache]:
    store = SqliteStore()
    store.init_db()
    return store, RawResponseCache(store, BlobStore())


def _print_matrix(run: Run, scores, results) -> None:
    summary = pipeline_summary(scores, results)
    table = Table(title=f"results — {run.dataset_ref}  x  {run.task_ref}")
    table.add_column("pipeline")
    for col in ("accuracy (95% CI)", "macro", "halluc", "missed", "fields", "errors", "cost $", "latency ms"):
        table.add_column(col, justify="right")
    for pc in run.pipelines:
        s = summary.get(pc.config_hash, {})
        label = pc.config.get("label") or pc.adapter
        lo, hi = s.get("ci_low", 0.0), s.get("ci_high", 0.0)
        table.add_row(
            f"{label} [dim]({pc.adapter})[/dim]",
            f"{s.get('accuracy', 0.0):.0%} [dim]\\[{lo:.0%}–{hi:.0%}][/dim]",
            f"{s.get('accuracy_macro', 0.0):.0%}",
            str(s.get("hallucinations", 0)),
            str(s.get("missed", 0)),
            str(s.get("fields_scored", 0)),
            str(s.get("errors", 0)),
            f"{s.get('cost_usd', 0.0):.4f}",
            f"{s.get('latency_ms', 0.0):.1f}",
        )
    rprint(table)


def _print_errors(run: Run, results, store: SqliteStore) -> None:
    errored = [r for r in results if r.status is ResultStatus.ERROR]
    if not errored:
        return
    docs = store.load_documents([r.doc_id for r in errored])
    labels = {pc.config_hash: (pc.config.get("label") or pc.adapter) for pc in run.pipelines}
    rprint(f"[red]{len(errored)} cell(s) errored:[/red]")
    for r in errored:
        doc = docs.get(r.doc_id)
        slug = (doc.slug if doc else None) or r.doc_id[:12]
        label = labels.get(r.pipeline_id, r.pipeline_id[:8])
        ec = r.error.error_class if r.error else "?"
        msg = ((r.error.message if r.error else "") or "").strip().replace("\n", " ")
        rprint(f"  [red]✗[/red] {slug} / {label} — [yellow]{ec}[/yellow]: {msg[:200]}")


@app.command()
def init(path: str = typer.Argument(".", help="directory to scaffold the eval project in")):
    """Scaffold an eval project: experiments/ datasets/ tasks/, a runnable no-key example, and a
    custom-adapter template for evaluating your own app. Never overwrites existing files."""
    root = Path(path)
    created, skipped = [], []
    for rel, content in _SCAFFOLD.items():
        target = root / rel
        if target.exists():
            skipped.append(rel)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        created.append(rel)
    for rel in created:
        rprint(f"[green]+[/green] {rel}")
    for rel in skipped:
        rprint(f"[dim]· exists, skipped {rel}[/dim]")
    prefix = "" if path == "." else f"cd {path} && "
    rprint("\n[bold]Next:[/bold]")
    rprint(f"  {prefix}ezpz run experiments/example.yaml   [dim]# runs with no API key[/dim]")
    rprint("  See experiments/example.yaml for how to plug in a real tool or your own app.")


@app.command()
def validate(experiment: str):
    """Lint a dataset/task/experiment before spending money: refs resolve, adapters + scorers are
    registered, secrets are present, ground-truth files exist."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        exp = ExperimentConfig.load_yaml(experiment)
    except Exception as e:
        rprint(f"[red]✗ cannot load experiment:[/red] {e}")
        raise typer.Exit(1)
    root = str(Path(experiment).resolve().parent.parent)

    try:
        import_modules(exp.plugins)  # so custom adapters/scorers are registered before the checks
    except Exception as e:
        errors.append(f"plugin module import failed: {e}")

    dataset = task = None
    try:
        dataset = resolve_dataset(root, exp.dataset)
    except Exception as e:
        errors.append(f"dataset '{exp.dataset}' did not resolve: {e}")
    try:
        task = resolve_task(root, exp.task)
    except Exception as e:
        errors.append(f"task '{exp.task}' did not resolve: {e}")

    adapters = set(adapter_registry.available())
    for pc in exp.pipelines:
        if pc.adapter not in adapters:
            errors.append(f"unknown adapter '{pc.adapter}' (have: {sorted(adapters)})")
        env = pc.config.get("api_key_env")
        if env and env not in os.environ:
            errors.append(f"pipeline '{pc.adapter}': secret env var '{env}' is not set")

    scorers = set(scorer_registry.available())
    if task is not None:
        for f in task.fields:
            for s in (f.scorers or exp.scorers):
                if s.name not in scorers:
                    errors.append(f"field '{f.name}': unknown scorer '{s.name}'")
    if dataset is not None:
        for doc in dataset.documents:
            if doc.ground_truth_path and not (Path(dataset.root) / doc.ground_truth_path).exists():
                warnings.append(f"doc '{doc.slug or doc.doc_id}': ground-truth file missing")

    for w in warnings:
        rprint(f"[yellow]⚠ {w}[/yellow]")
    if errors:
        for err in errors:
            rprint(f"[red]✗ {err}[/red]")
        raise typer.Exit(1)
    rprint(
        f"[green]✓ valid[/green] — {len(exp.pipelines)} pipelines, "
        f"{len(task.fields) if task else 0} fields, {len(dataset.documents) if dataset else 0} docs"
    )


@app.command()
def export(run_id: str, out: str = typer.Option(None, help="output .json file; default = stdout")):
    """Export a run (run + results + scores) as JSON for offline analysis / archival."""
    store = SqliteStore()
    store.init_db()
    text = json.dumps(store.export(run_id), indent=2)
    if out:
        Path(out).write_text(text)
        rprint(f"[green]exported[/green] {run_id} → {out}")
    else:
        print(text)


@app.command()
def run(
    experiment: str,
    budget_usd: float = typer.Option(None, help="hard cap; refuse if estimate exceeds, stop-early if actuals do"),
    sample: int = typer.Option(None, help="run on N docs for fast iteration"),
    force_refresh: bool = typer.Option(False, help="bypass extraction cache"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable summary to stdout (for CI)"),
    fail_under: float = typer.Option(None, help="exit 2 if any pipeline's accuracy < this (CI gate)"),
):
    """Execute an experiment config (dataset x pipelines x task x scorers)."""
    exp = ExperimentConfig.load_yaml(experiment)
    import_modules(exp.plugins)  # custom adapter/scorer modules declared by the experiment
    root = str(Path(experiment).resolve().parent.parent)
    dataset = resolve_dataset(root, exp.dataset)
    task = resolve_task(root, exp.task)
    if sample:
        dataset = dataset.sample(sample)

    options = exp.options.model_copy(update={
        "force_refresh": force_refresh or exp.options.force_refresh,  # CLI flag enables; never disables YAML
        "budget_usd": budget_usd if budget_usd is not None else exp.options.budget_usd,
    })
    store, cache = _store_and_cache()
    pipelines = [get_adapter(pc.adapter)(pc) for pc in exp.pipelines]
    run_obj = Run(
        run_id=uuid.uuid4().hex[:12], dataset_ref=exp.dataset, task_ref=exp.task,
        pipelines=exp.pipelines, scorers=exp.scorers, options=options,
        env={"python": platform.python_version()}, status=RunStatus.RUNNING, started_at=_now(),
    )

    # Pre-run estimate + budget refusal (don't start a run we can't afford).
    est = estimate_cost(run_obj, dataset, cache, task.schema_hash())
    if not json_out:
        rprint(f"estimate: [bold]${est['estimate_usd']:.4f}[/bold] "
               f"({est['uncached_cells']} uncached, {est['cached_cells']} cached cells)")
    if options.budget_usd is not None and est["estimate_usd"] > options.budget_usd:
        rprint(f"[red]refusing to run:[/red] estimate ${est['estimate_usd']:.4f} "
               f"exceeds budget ${options.budget_usd:.2f}")
        raise typer.Exit(1)

    Executor(store, cache, options.budget_usd).run(run_obj, dataset, task, pipelines, exp.scorers)

    run_loaded = store.load_run(run_obj.run_id)
    scores = store.load_scores(run_obj.run_id)
    results = store.load_results(run_obj.run_id)
    summary = pipeline_summary(scores, results)
    if json_out:
        errored = [
            {"doc_id": r.doc_id, "pipeline_id": r.pipeline_id,
             "error_class": r.error.error_class if r.error else None,
             "message": r.error.message if r.error else None}
            for r in results if r.status is ResultStatus.ERROR
        ]
        print(json.dumps(
            {"run_id": run_obj.run_id, "summary": summary, "errors": errored}, default=str))
    else:
        _print_matrix(run_loaded, scores, results)
        _print_errors(run_loaded, results, store)
        rprint(f"[green]run_id:[/green] {run_obj.run_id}")

    if fail_under is not None:
        low = [pid for pid, s in summary.items() if s.get("accuracy", 0.0) < fail_under]
        if low:
            if not json_out:
                rprint(f"[red]CI gate failed:[/red] {len(low)} pipeline(s) below {fail_under:.0%}")
            raise typer.Exit(2)


@app.command()
def score(run_id: str):
    """Re-score an existing run's CACHED extractions with current scorers. Fast and free."""
    store = SqliteStore()
    store.init_db()
    cache = RawResponseCache(store, BlobStore())

    run_obj = store.load_run(run_id)
    task = store.load_task(run_obj.task_ref)
    schema_hash = task.schema_hash()
    pipelines = {pc.config_hash: get_adapter(pc.adapter)(pc) for pc in run_obj.pipelines}

    new_scores = []
    for r in store.load_results(run_id):
        pipeline = pipelines.get(r.pipeline_id)
        if pipeline is None:
            continue
        cached = cache.get(cache_key(r.doc_id, r.pipeline_id, schema_hash))
        if cached is None:
            continue  # no cached raw -> nothing to re-derive (would need a re-run)
        gt_fields = store.load_ground_truth(run_id, r.doc_id)
        gt = GroundTruth(doc_id=r.doc_id, fields=gt_fields) if gt_fields is not None else None
        norm = normalize_mapped(pipeline.map(cached["raw"], task), task)
        new_scores.extend(score_fields(norm, gt, task, run_obj.scorers, r.doc_id, r.pipeline_id))

    store.replace_scores(run_id, new_scores)
    _print_matrix(run_obj, store.load_scores(run_id), store.load_results(run_id))
    rprint(f"[green]re-scored:[/green] {run_id}  ({len(new_scores)} field-scores, no extraction)")


@app.command()
def compare(run_a: str, run_b: str):
    """Diff two runs: per-pipeline paired accuracy delta on shared (doc, field, scorer) cells."""
    store = SqliteStore()
    store.init_db()
    a_scores, b_scores = store.load_scores(run_a), store.load_scores(run_b)

    table = Table(title=f"compare — {run_a} (A) vs {run_b} (B)")
    table.add_column("pipeline")
    for col in ("Δ accuracy (A−B)", "95% CI", "n", "verdict"):
        table.add_column(col, justify="right")
    for pc in store.load_run(run_a).pipelines:
        pid = pc.config_hash
        label = pc.config.get("label") or pc.adapter
        res = paired_compare(
            [s for s in a_scores if s.pipeline_id == pid],
            [s for s in b_scores if s.pipeline_id == pid],
        )
        verdict = (
            "within noise" if res["within_noise"]
            else ("A better" if res["mean_delta"] > 0 else "B better")
        )
        table.add_row(
            f"{label} [dim]({pc.adapter})[/dim]",
            f"{res['mean_delta']:+.1%}",
            f"[{res['ci_low']:+.2f}, {res['ci_high']:+.2f}]",
            str(res["n"]),
            verdict,
        )
    rprint(table)


@app.command()
def analyze(run_id: str):
    """Strategy diagnostics (reads SQLite; re-runs nothing): confidence calibration — accuracy among
    the most-confident predictions vs overall — and rates of boolean strategy flags (e.g. escalation)."""
    store = SqliteStore()
    store.init_db()
    run = store.load_run(run_id)
    scores = store.load_scores(run_id)
    results = store.load_results(run_id)
    flags = sorted({
        k for r in results if r.extras for k, v in r.extras.items() if isinstance(v, bool)
    })
    table = Table(title=f"analysis — {run_id}")
    table.add_column("pipeline")
    for col in ("overall acc", "acc @ top-50% conf", *flags):
        table.add_column(col, justify="right")
    for pc in run.pipelines:
        pid = pc.config_hash
        label = pc.config.get("label") or pc.adapter
        pr = [r for r in results if r.pipeline_id == pid]
        cal = calibration([s for s in scores if s.pipeline_id == pid], pr)
        row = [f"{label} [dim]({pc.adapter})[/dim]"]
        row += ([f"{cal['overall_accuracy']:.0%}", f"{cal['accuracy_at_auto']:.0%}"]
                if cal.get("n") else ["—", "—"])
        for k in flags:
            rate = extras_rate(pr, k)
            row.append(f"{rate:.0%}" if rate is not None else "—")
        table.add_row(*row)
    rprint(table)


@app.command()
def view(
    db: str = typer.Option(".ezpz/ezpz.sqlite", help="SQLite DB to view"),
    port: int = typer.Option(8501, help="port for the local viewer server"),
    host: str = typer.Option("127.0.0.1", help="bind address"),
    no_browser: bool = typer.Option(False, "--no-browser", help="don't auto-open a browser"),
):
    """Launch the local viewer — a stdlib web server (no extra deps). Serves the SPA + a read-only
    JSON API over SQLite; the budget modal can also launch a budget-gated re-run (datasets/tasks
    resolve from the current directory)."""
    from ezpz.ui.server import serve

    db_path = str(Path(db).resolve())
    if not Path(db_path).exists():
        rprint(f"[yellow]no DB at {db_path}[/yellow] — run [bold]ezpz run <experiment>[/bold] first.")
    serve(db_path, host=host, port=port, open_browser=not no_browser, root=str(Path.cwd()))


_SCAFFOLD: dict[str, str] = {
    "tasks/example_task.yaml": (
        'name: example_task\n'
        'version: "1"\n'
        'type: extraction\n'
        'scorers:\n'
        '  - name: exact\n'
        'fields:\n'
        '  - { name: invoice_number, type: string,   required: true, scorers: [{ name: exact }] }\n'
        '  - { name: total_amount,   type: currency, required: true, scorers: [{ name: numeric_tolerance, config: { abs: 0.01 } }] }\n'
    ),
    "datasets/example/manifest.jsonl": (
        '{"slug": "doc-001", "path": "docs/doc-001.txt", "mime": "text/plain", '
        '"doc_type": "invoice", "tags": ["demo"], "ground_truth_path": "ground_truth/doc-001.json"}\n'
    ),
    "datasets/example/docs/doc-001.txt": "DEMO INVOICE\nInvoice #: INV-001\nTotal: $100.00\n",
    "datasets/example/ground_truth/doc-001.json": (
        '{\n'
        '  "doc_id": "content-hash-resolved-at-load-time",\n'
        '  "fields": {\n'
        '    "invoice_number": "INV-001",\n'
        '    "total_amount": { "amount": "100.00", "currency": "USD" }\n'
        '  }\n'
        '}\n'
    ),
    "experiments/example.yaml": (
        "# Runnable with NO API keys — uses the built-in deterministic `fake` adapter:\n"
        "#     ezpz run experiments/example.yaml\n"
        "#\n"
        "# To evaluate a REAL tool, install its extra and swap the pipeline, e.g.:\n"
        "#     - adapter: anthropic\n"
        "#       config: { model: claude-opus-4-8, api_key_env: ANTHROPIC_API_KEY }\n"
        "#\n"
        "# To evaluate YOUR OWN app, register a custom adapter (see custom_adapter_example.py) and\n"
        "# either advertise it as an entry point in your pyproject.toml:\n"
        '#     [project.entry-points."ezpz.adapters"]\n'
        '#     myapp = "myapp.ezpz_adapter"\n'
        "# or list its module here so ezpz imports it first:\n"
        "#     plugins: [custom_adapter_example]\n"
        "# then reference `adapter: myapp` in a pipeline.\n"
        "dataset: example@1\n"
        "task: example_task@1\n"
        "pipelines:\n"
        "  - adapter: fake\n"
        "    config:\n"
        "      label: perfect\n"
        "      by_slug:\n"
        '        doc-001: { invoice_number: "INV-001", total_amount: "$100.00" }\n'
        "  - adapter: fake\n"
        "    config:\n"
        "      label: typo\n"
        "      by_slug:\n"
        '        doc-001: { invoice_number: "INV-XXX", total_amount: "$100.00" }\n'
        "options:\n"
        "  concurrency: 1\n"
    ),
    "custom_adapter_example.py": (
        '"""Template: wrap YOUR app\'s extraction so ezpz can evaluate it.\n'
        "\n"
        "Make it discoverable by either:\n"
        "  - listing it in an experiment's `plugins:` field:  plugins: [custom_adapter_example]\n"
        "  - or advertising it as an entry point in your pyproject.toml:\n"
        '        [project.entry-points.\"ezpz.adapters\"]\n'
        '        myapp = \"custom_adapter_example\"\n'
        "Then reference `adapter: myapp` in a pipeline.\n"
        '"""\n'
        "from typing import Any\n"
        "\n"
        "from ezpz.adapters.base import Capabilities, Pipeline\n"
        "from ezpz.adapters.registry import register\n"
        "from ezpz.core.document import Document\n"
        "from ezpz.core.result import Cost, FieldValue\n"
        "from ezpz.core.task import Task\n"
        "\n"
        "\n"
        '@register("myapp")\n'
        "class MyAppPipeline(Pipeline):\n"
        "    capabilities = Capabilities(confidence=False, provenance=False)\n"
        "\n"
        "    def compile(self, task: Task) -> Any:\n"
        "        # Translate the task schema into whatever YOUR app needs (a prompt, a field list).\n"
        "        return [f.name for f in task.fields]\n"
        "\n"
        "    def ingest(self, document: Document) -> Any:\n"
        "        # Load the document. `document.source_path` is the absolute file path.\n"
        "        return document.source_path\n"
        "\n"
        "    def invoke(self, prepared: Any, ingested: Any) -> tuple[Any, Cost]:\n"
        "        # Call YOUR app. `raw` MUST be JSON-serializable (it is cached + re-mapped on hits).\n"
        "        raw = {name: None for name in prepared}  # <- replace with your app's real output\n"
        "        return raw, Cost(usd=0.0)\n"
        "\n"
        "    def map(self, raw: Any, task: Task) -> dict[str, FieldValue]:\n"
        "        # Map your app's raw output -> canonical FieldValues (the engine normalizes after).\n"
        "        return {f.name: FieldValue(value=raw.get(f.name)) for f in task.fields}\n"
    ),
    ".env.example": (
        "# Copy to .env (gitignore it). The CLI auto-loads ./.env before every command (real env wins).\n"
        "# Referenced by NAME from adapter configs (api_key_env: ...); never commit real keys.\n"
        "ANTHROPIC_API_KEY=\n"
        "GEMINI_API_KEY=\n"
        "OPENAI_API_KEY=\n"
        "EXTEND_API_KEY=\n"
        "LLAMA_CLOUD_API_KEY=\n"
    ),
}


if __name__ == "__main__":
    app()
