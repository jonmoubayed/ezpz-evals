"""M4 DoD: the invoice example (line_items list + ABSENT po_number) scores sensibly via the
fake adapter — hallucinations and missed rows surface, and aggregates carry macro/micro + CIs."""
from pathlib import Path

from ezpz.adapters.registry import get_adapter
from ezpz.config.experiment import ExperimentConfig, resolve_dataset, resolve_task
from ezpz.core.run import Run, RunStatus
from ezpz.engine.aggregate import pipeline_summary, slice_metrics
from ezpz.engine.cache import RawResponseCache
from ezpz.engine.executor import Executor
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore

REPO = Path(__file__).resolve().parents[1]
ROOT = str(REPO / "examples")


def _run(tmp_path):
    store = SqliteStore(str(tmp_path / "db.sqlite"))
    store.init_db()
    cache = RawResponseCache(store, BlobStore(str(tmp_path / "blobs")))
    exp = ExperimentConfig.load_yaml(str(REPO / "examples" / "experiments" / "fake_invoice_smoke.yaml"))
    dataset = resolve_dataset(ROOT, exp.dataset)
    task = resolve_task(ROOT, exp.task)
    pipelines = [get_adapter(p.adapter)(p) for p in exp.pipelines]
    run = Run(run_id="r1", dataset_ref=exp.dataset.ref, task_ref=exp.task,
              pipelines=exp.pipelines, scorers=exp.scorers, options=exp.options,
              status=RunStatus.RUNNING)
    Executor(store, cache).run(run, dataset, task, pipelines, exp.scorers)
    return store, exp, dataset


def test_invoice_example_scores_sensibly(tmp_path):
    store, exp, _ = _run(tmp_path)
    accurate, flawed = exp.pipelines[0].config_hash, exp.pipelines[1].config_hash
    summary = pipeline_summary(store.load_scores("r1"), store.load_results("r1"))

    # the accurate pipeline matches GT; the flawed one is clearly worse
    assert summary[accurate]["accuracy"] > summary[flawed]["accuracy"]
    # hallucination (PO returned for an ABSENT field) is visible on the flawed pipeline only
    assert summary[accurate]["hallucinations"] == 0
    assert summary[flawed]["hallucinations"] == 1
    # aggregates carry macro/micro + a bootstrap CI
    assert "accuracy_macro" in summary[accurate]
    assert summary[accurate]["ci_low"] <= summary[accurate]["accuracy"] <= summary[accurate]["ci_high"]


def test_list_table_surfaces_missed_row_for_flawed_pipeline(tmp_path):
    store, exp, _ = _run(tmp_path)
    flawed = exp.pipelines[1].config_hash
    li_scores = [
        s for s in store.load_scores("r1")
        if s.pipeline_id == flawed and s.scorer == "list_table"
    ]
    assert len(li_scores) == 1
    assert li_scores[0].detail["missed_rows"] == 1   # flawed dropped the "Setup fee" line
    assert li_scores[0].detail["recall"] == 0.5


def test_slice_breakdown_uses_document_tags(tmp_path):
    store, exp, dataset = _run(tmp_path)
    doc_tags = {d.doc_id: d.tags for d in dataset.documents}
    sliced = slice_metrics(store.load_scores("r1"), doc_tags)
    assert "clean" in sliced and sliced["clean"]["n"] > 0  # the doc is tagged 'clean'
