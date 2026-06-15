"""End-to-end vertical slice on the no-network fake dataset: run -> normalize -> score -> store,
and a second run that is a full cache hit (no re-invoke)."""
from pathlib import Path

from ezpz.adapters.base import Capabilities, Pipeline
from ezpz.adapters.registry import get_adapter
from ezpz.config.experiment import ExperimentConfig, resolve_dataset, resolve_task
from ezpz.core.document import GroundTruth
from ezpz.core.result import ResultStatus
from ezpz.core.run import PipelineConfig, Run, RunStatus
from ezpz.engine.aggregate import pipeline_summary
from ezpz.engine.cache import RawResponseCache, cache_key
from ezpz.engine.executor import Executor, normalize_mapped, score_fields
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore

REPO = Path(__file__).resolve().parents[1]
ROOT = str(REPO / "examples")


def _harness(tmp_path):
    store = SqliteStore(str(tmp_path / "db.sqlite"))
    store.init_db()
    cache = RawResponseCache(store, BlobStore(str(tmp_path / "blobs")))
    exp = ExperimentConfig.load_yaml(str(REPO / "examples" / "experiments" / "fake_smoke.yaml"))
    dataset = resolve_dataset(ROOT, exp.dataset)
    task = resolve_task(ROOT, exp.task)
    pipelines = [get_adapter(pc.adapter)(pc) for pc in exp.pipelines]
    return store, cache, exp, dataset, task, pipelines


def _make_run(run_id, exp):
    return Run(
        run_id=run_id, dataset_ref=exp.dataset, task_ref=exp.task,
        pipelines=exp.pipelines, scorers=exp.scorers, options=exp.options,
        status=RunStatus.RUNNING,
    )


def test_executor_scores_perfect_and_noisy_pipelines(tmp_path):
    store, cache, exp, dataset, task, pipelines = _harness(tmp_path)
    Executor(store, cache).run(_make_run("r1", exp), dataset, task, pipelines, exp.scorers)

    summary = pipeline_summary(store.load_scores("r1"), store.load_results("r1"))
    perfect, noisy = exp.pipelines[0].config_hash, exp.pipelines[1].config_hash

    assert summary[perfect]["accuracy"] == 1.0
    assert summary[noisy]["accuracy"] < 1.0
    assert summary[perfect]["fields_scored"] == 8  # 2 docs x 4 fields
    assert store.load_run("r1").status == RunStatus.COMPLETE


def test_second_run_is_full_cache_hit_no_reinvoke(tmp_path):
    store, cache, exp, dataset, task, pipelines = _harness(tmp_path)
    ex = Executor(store, cache)

    ex.run(_make_run("r1", exp), dataset, task, pipelines, exp.scorers)
    after_first = sum(p.invocations for p in pipelines)
    assert after_first == 4  # 2 docs x 2 pipelines, each invoked once

    ex.run(_make_run("r2", exp), dataset, task, pipelines, exp.scorers)
    after_second = sum(p.invocations for p in pipelines)
    assert after_second == after_first  # every cell was a cache hit -> no re-invoke

    # the cached re-run still derives + scores correctly
    summary = pipeline_summary(store.load_scores("r2"), store.load_results("r2"))
    assert summary[exp.pipelines[0].config_hash]["accuracy"] == 1.0


def test_force_refresh_re_extracts_despite_warm_cache(tmp_path):
    store, cache, exp, dataset, task, pipelines = _harness(tmp_path)
    ex = Executor(store, cache)
    ex.run(_make_run("r1", exp), dataset, task, pipelines, exp.scorers)
    first = sum(p.invocations for p in pipelines)
    assert first == 4

    run2 = _make_run("r2", exp)
    run2.options = run2.options.model_copy(update={"force_refresh": True})
    ex.run(run2, dataset, task, pipelines, exp.scorers)
    assert sum(p.invocations for p in pipelines) == first * 2  # bypassed cache, re-extracted


class _BoomPipeline(Pipeline):
    """Raises inside invoke() to exercise the error path."""

    capabilities = Capabilities()

    def compile(self, task):
        return [f.name for f in task.fields]

    def ingest(self, document):
        return document

    def invoke(self, prepared, ingested):
        raise RuntimeError("boom")

    def map(self, raw, task):
        return {}


def test_failed_extraction_is_error_and_not_cached(tmp_path):
    store, cache, exp, dataset, task, _ = _harness(tmp_path)
    boom = _BoomPipeline(PipelineConfig(adapter="boom", config={"k": 1}))
    Executor(store, cache).run(_make_run("rb", exp), dataset, task, [boom], exp.scorers)

    results = store.load_results("rb")
    assert results and all(r.status is ResultStatus.ERROR for r in results)
    assert all(r.error.message == "boom" for r in results)  # the failure reason is persisted
    assert store.load_scores("rb") == []  # error -> no silently-empty scored fields
    schema_hash = task.schema_hash()
    for doc in dataset.documents:  # a failed cell leaves nothing in the cache
        assert cache.get(cache_key(doc.doc_id, boom.pipeline_id, schema_hash)) is None


def test_score_rederives_from_cache_without_reinvoke(tmp_path):
    store, cache, exp, dataset, task, pipelines = _harness(tmp_path)
    Executor(store, cache).run(_make_run("r1", exp), dataset, task, pipelines, exp.scorers)
    invoked = sum(p.invocations for p in pipelines)
    original = {
        (s.doc_id, s.pipeline_id, s.field, s.scorer): s.value for s in store.load_scores("r1")
    }

    # replicate the `ezpz score` core loop: derive from cached raw, never invoke
    schema_hash = task.schema_hash()
    by_id = {p.pipeline_id: p for p in pipelines}
    rederived = {}
    for r in store.load_results("r1"):
        cached = cache.get(cache_key(r.doc_id, r.pipeline_id, schema_hash))
        assert cached is not None
        gt = GroundTruth(doc_id=r.doc_id, fields=store.load_ground_truth("r1", r.doc_id))
        norm = normalize_mapped(by_id[r.pipeline_id].map(cached["raw"], task), task)
        for s in score_fields(norm, gt, task, exp.scorers, r.doc_id, r.pipeline_id):
            rederived[(s.doc_id, s.pipeline_id, s.field, s.scorer)] = s.value

    assert sum(p.invocations for p in pipelines) == invoked  # no re-extraction
    assert rederived == original


def test_unparseable_prediction_scores_as_parse_failure(tmp_path):
    import json

    from ezpz.core.document import Dataset
    from ezpz.core.task import FieldSpec, ScorerRef, Task
    from ezpz.core.values import ValueType

    ds = tmp_path / "ds"
    (ds / "docs").mkdir(parents=True)
    (ds / "ground_truth").mkdir(parents=True)
    (ds / "docs" / "d1.txt").write_text("doc")
    (ds / "ground_truth" / "d1.json").write_text(
        json.dumps({"fields": {"amount": {"amount": "10.00", "currency": "USD"}}})
    )
    (ds / "manifest.jsonl").write_text(
        json.dumps({"slug": "d1", "path": "docs/d1.txt", "ground_truth_path": "ground_truth/d1.json"})
        + "\n"
    )
    dataset = Dataset.load_from_manifest(str(ds), "ds", "1")
    task = Task(name="t", version="1", fields=[
        FieldSpec(
            name="amount", type=ValueType.CURRENCY,
            scorers=[ScorerRef(name="numeric_tolerance", config={"abs": 0.01})],
        ),
    ])
    cfg = PipelineConfig(adapter="fake", config={"by_slug": {"d1": {"amount": "not-a-number"}}})

    store = SqliteStore(str(tmp_path / "db.sqlite"))
    store.init_db()
    cache = RawResponseCache(store, BlobStore(str(tmp_path / "blobs")))
    run = Run(run_id="r1", dataset_ref="ds@1", task_ref="t@1", pipelines=[cfg], scorers=[],
              status=RunStatus.RUNNING)
    Executor(store, cache).run(run, dataset, task, [get_adapter("fake")(cfg)], [])

    scores = store.load_scores("r1")
    assert len(scores) == 1
    assert scores[0].detail["case"] == "parse_failure"  # un-canonical value, not silently wrong
