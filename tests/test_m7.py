"""M7 hardening: budget guard, cost estimate, retries, circuit breaker, resumability,
per-provider concurrency, export, and `validate` (machine-readable CI surface)."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ezpz.adapters.base import Capabilities, Pipeline
from ezpz.adapters.fake import FakePipeline
from ezpz.adapters.registry import get_adapter
from ezpz.cli.main import app
from ezpz.config.experiment import ExperimentConfig, resolve_dataset, resolve_task
from ezpz.core.document import Dataset
from ezpz.core.result import Cost, ExtractionResult, FieldValue, ResultStatus
from ezpz.core.run import PipelineConfig, Run, RunOptions, RunStatus
from ezpz.core.task import FieldSpec, ScorerRef, Task
from ezpz.core.values import ValueType
from ezpz.engine.aggregate import pipeline_summary
from ezpz.engine.cache import RawResponseCache
from ezpz.engine.cost import BudgetGuard, estimate_cost
from ezpz.engine.executor import Executor
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore

REPO = Path(__file__).resolve().parents[1]
ROOT = str(REPO / "examples")


def _harness(tmp_path):
    store = SqliteStore(str(tmp_path / "db.sqlite"))
    store.init_db()
    return store, RawResponseCache(store, BlobStore(str(tmp_path / "blobs")))


def _ndoc_dataset(tmp_path, n) -> Dataset:
    ds = tmp_path / "ds"
    (ds / "docs").mkdir(parents=True)
    lines = []
    for i in range(n):
        (ds / "docs" / f"d{i}.txt").write_text(f"doc {i}")
        lines.append(json.dumps({"slug": f"d{i}", "path": f"docs/d{i}.txt", "mime": "text/plain"}))
    (ds / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    return Dataset.load_from_manifest(str(ds), "ds", "1")


def _task() -> Task:
    return Task(name="t", version="1",
                fields=[FieldSpec(name="a", type=ValueType.STRING, scorers=[ScorerRef(name="exact")])])


def _run_obj(pipelines, **opts):
    return Run(run_id="r1", dataset_ref="ds@1", task_ref="t@1", pipelines=pipelines, scorers=[],
               options=RunOptions(**opts), status=RunStatus.RUNNING)


# ---- budget guard + estimate ----

def test_budget_guard_trips_over_cap():
    g = BudgetGuard(0.05)
    g.add(0.03)
    assert not g.over
    g.add(0.03)
    assert g.over
    assert not BudgetGuard(None).over  # no cap -> never over


def test_estimate_counts_uncached_and_sums_priors(tmp_path):
    _, cache = _harness(tmp_path)
    pcs = [PipelineConfig(adapter="fake", config={"label": "a", "cost_prior_usd": 0.02}),
           PipelineConfig(adapter="fake", config={"label": "b", "cost_prior_usd": 0.05})]
    est = estimate_cost(_run_obj(pcs), _ndoc_dataset(tmp_path, 1), cache, _task().schema_hash())
    assert est["uncached_cells"] == 2
    assert est["estimate_usd"] == pytest.approx(0.07)


# ---- executor hardening ----

class _Flaky(Pipeline):
    capabilities = Capabilities()

    def __init__(self, config, fail_times):
        super().__init__(config)
        self.calls = 0
        self.fail_times = fail_times

    def compile(self, task):
        return [f.name for f in task.fields]

    def ingest(self, document):
        return document

    def invoke(self, prepared, ingested):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("flaky")
        return {n: "x" for n in prepared}, Cost(usd=0.0)

    def map(self, raw, task):
        return {n: FieldValue(value=v) for n, v in raw.items()}

    def classify_error(self, exc):
        return "transport" if isinstance(exc, ConnectionError) else "unknown"


def test_executor_retries_transient_errors(tmp_path):
    store, cache = _harness(tmp_path)
    flaky = _Flaky(PipelineConfig(adapter="flaky", config={}), fail_times=2)
    Executor(store, cache, retries=3, backoff=0.0).run(
        _run_obj([flaky.config], concurrency=1), _ndoc_dataset(tmp_path, 1), _task(), [flaky], [])
    assert store.load_results("r1")[0].status is ResultStatus.OK
    assert flaky.calls == 3  # 2 transient failures + 1 success


class _Boom(Pipeline):
    capabilities = Capabilities()

    def __init__(self, config):
        super().__init__(config)
        self.calls = 0

    def compile(self, task):
        return [f.name for f in task.fields]

    def ingest(self, document):
        return document

    def invoke(self, prepared, ingested):
        self.calls += 1
        raise RuntimeError("always")

    def map(self, raw, task):
        return {}


def test_executor_circuit_breaker_short_circuits_dead_provider(tmp_path):
    store, cache = _harness(tmp_path)
    boom = _Boom(PipelineConfig(adapter="boom", config={}))
    Executor(store, cache, retries=0, circuit_threshold=2).run(
        _run_obj([boom.config], concurrency=1), _ndoc_dataset(tmp_path, 4), _task(), [boom], [])
    circuit = [r for r in store.load_results("r1") if r.error and r.error.error_class == "circuit_open"]
    assert len(circuit) == 2  # docs 3 & 4 short-circuited after 2 consecutive failures
    assert boom.calls == 2    # only the first 2 cells were actually invoked


class _Costing(Pipeline):
    capabilities = Capabilities()

    def __init__(self, config):
        super().__init__(config)
        self.calls = 0

    def compile(self, task):
        return [f.name for f in task.fields]

    def ingest(self, document):
        return document

    def invoke(self, prepared, ingested):
        self.calls += 1
        return {n: "x" for n in prepared}, Cost(usd=1.0)

    def map(self, raw, task):
        return {n: FieldValue(value=v) for n, v in raw.items()}


def test_executor_budget_stops_early(tmp_path):
    store, cache = _harness(tmp_path)
    costing = _Costing(PipelineConfig(adapter="costing", config={}))
    Executor(store, cache).run(
        _run_obj([costing.config], concurrency=1, budget_usd=2.5),
        _ndoc_dataset(tmp_path, 5), _task(), [costing], [])
    ok = [r for r in store.load_results("r1") if r.status is ResultStatus.OK]
    assert costing.calls == 3 and len(ok) == 3  # spent 3.0 > 2.5 after the 3rd, then stopped


def test_executor_resumes_completed_cells(tmp_path):
    store, cache = _harness(tmp_path)
    dataset = _ndoc_dataset(tmp_path, 2)
    cfg = PipelineConfig(adapter="fake", config={"fake_fields": {"a": "x"}})
    fake = FakePipeline(cfg)
    run = _run_obj([cfg], concurrency=1)
    store.save_run(run)
    store.save_result("r1", ExtractionResult(  # simulate a prior partial run: doc0 already done
        doc_id=dataset.documents[0].doc_id, pipeline_id=cfg.config_hash, status=ResultStatus.OK, fields={}))
    Executor(store, cache).run(run, dataset, _task(), [fake], [])
    assert fake.invocations == 1  # only the un-done doc was extracted


def test_concurrent_results_match_serial(tmp_path):
    exp = ExperimentConfig.load_yaml(str(REPO / "examples" / "experiments" / "fake_invoice_smoke.yaml"))
    dataset = resolve_dataset(ROOT, exp.dataset)
    task = resolve_task(ROOT, exp.task)

    def summarize(concurrency, sub):
        store, cache = _harness(tmp_path / sub)
        pipelines = [get_adapter(p.adapter)(p) for p in exp.pipelines]
        run = Run(run_id="r1", dataset_ref=exp.dataset.ref, task_ref=exp.task, pipelines=exp.pipelines,
                  scorers=exp.scorers, status=RunStatus.RUNNING,
                  options=exp.options.model_copy(update={"concurrency": concurrency}))
        Executor(store, cache).run(run, dataset, task, pipelines, exp.scorers)
        summary = pipeline_summary(store.load_scores("r1"), store.load_results("r1"))
        return {pid: s["accuracy"] for pid, s in summary.items()}

    assert summarize(1, "serial") == summarize(3, "concurrent")


def test_export_serializes_run_results_scores(tmp_path):
    store, cache = _harness(tmp_path)
    cfg = PipelineConfig(adapter="fake", config={"fake_fields": {"a": "x"}})
    Executor(store, cache).run(_run_obj([cfg]), _ndoc_dataset(tmp_path, 1), _task(),
                               [FakePipeline(cfg)], [ScorerRef(name="exact")])
    data = store.export("r1")
    assert data["run"]["run_id"] == "r1"
    assert len(data["results"]) == 1
    assert isinstance(data["scores"], list)


# ---- validate (CI surface) ----

def test_validate_passes_on_good_experiment():
    result = CliRunner().invoke(
        app, ["validate", str(REPO / "examples" / "experiments" / "fake_invoice_smoke.yaml")])
    assert result.exit_code == 0
    assert "valid" in result.stdout


def test_validate_flags_unknown_adapter_and_missing_secret(tmp_path):
    (tmp_path / "datasets" / "ds" / "docs").mkdir(parents=True)
    (tmp_path / "datasets" / "ds" / "docs" / "d.txt").write_text("x")
    (tmp_path / "datasets" / "ds" / "manifest.jsonl").write_text(
        json.dumps({"slug": "d", "path": "docs/d.txt", "mime": "text/plain"}) + "\n")
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "t.yaml").write_text(
        'name: t\nversion: "1"\ntype: extraction\nscorers: [{name: exact}]\n'
        "fields:\n  - {name: a, type: string, scorers: [{name: exact}]}\n")
    (tmp_path / "experiments").mkdir()
    exp = tmp_path / "experiments" / "bad.yaml"
    exp.write_text(
        "dataset: ds@1\ntask: t@1\npipelines:\n"
        "  - {adapter: nope, config: {api_key_env: MISSING_SECRET_XYZ}}\noptions: {}\n")

    result = CliRunner().invoke(app, ["validate", str(exp)])
    assert result.exit_code == 1
    assert "unknown adapter" in result.stdout
    assert "MISSING_SECRET_XYZ" in result.stdout
