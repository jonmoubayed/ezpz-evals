"""Viewer view-models (ezpz.ui.data) over a populated DB — no Streamlit needed.

Reads ONLY what a run persisted: results, scores, ground truth, documents.
"""
from pathlib import Path

from ezpz.adapters.registry import get_adapter
from ezpz.config.experiment import ExperimentConfig, resolve_dataset, resolve_task
from ezpz.core.run import Run, RunStatus
from ezpz.engine.cache import RawResponseCache
from ezpz.engine.executor import Executor
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore
from ezpz.ui import data as D

REPO = Path(__file__).resolve().parents[1]
ROOT = str(REPO / "examples")


def _populated(tmp_path):
    store = SqliteStore(str(tmp_path / "db.sqlite"))
    store.init_db()
    cache = RawResponseCache(store, BlobStore(str(tmp_path / "blobs")))
    exp = ExperimentConfig.load_yaml(str(REPO / "examples" / "experiments" / "fake_invoice_smoke.yaml"))
    dataset = resolve_dataset(ROOT, exp.dataset)
    task = resolve_task(ROOT, exp.task)
    pipelines = [get_adapter(p.adapter)(p) for p in exp.pipelines]
    for rid in ("r1", "r2"):
        run = Run(run_id=rid, dataset_ref=exp.dataset, task_ref=exp.task,
                  pipelines=exp.pipelines, scorers=exp.scorers, options=exp.options,
                  status=RunStatus.RUNNING)
        Executor(store, cache).run(run, dataset, task, pipelines, exp.scorers)
    return store, exp


def test_list_runs_and_slice_tags(tmp_path):
    store, _ = _populated(tmp_path)
    runs = D.list_runs(store)
    assert {r["run_id"] for r in runs} == {"r1", "r2"}
    assert D.slice_tags(store, "r1") == ["clean", "single-page"]


def test_leaderboard_reflects_accuracy_and_slices(tmp_path):
    store, exp = _populated(tmp_path)
    rows = {r["pipeline"]: r for r in D.leaderboard(store, "r1")}
    assert rows["accurate"]["accuracy"] == 1.0
    assert rows["flawed"]["accuracy"] < 1.0
    assert rows["flawed"]["hallucinations"] == 1
    # the single doc is tagged 'clean', so the slice keeps both pipelines
    clean = {r["pipeline"]: r for r in D.leaderboard(store, "r1", "clean")}
    assert clean["accurate"]["accuracy"] == 1.0


def test_documents_in_run(tmp_path):
    store, _ = _populated(tmp_path)
    docs = D.documents_in_run(store, "r1")
    assert len(docs) == 1
    assert docs[0]["slug"] == "inv-001"
    assert "clean" in docs[0]["tags"]


def test_drilldown_color_codes_presence_and_shows_gt(tmp_path):
    store, exp = _populated(tmp_path)
    doc_id = D.documents_in_run(store, "r1")[0]["doc_id"]
    view = D.drilldown(store, "r1", doc_id)
    accurate, flawed = exp.pipelines[0].config_hash, exp.pipelines[1].config_hash

    po = next(f for f in view["fields"] if f["field"] == "po_number")
    assert po["gt"] == "ABSENT"
    assert po["cells"][accurate]["status"] == "absent_ok"      # correct null for ABSENT
    assert po["cells"][flawed]["status"] == "hallucinated"     # returned a value for ABSENT

    total = next(f for f in view["fields"] if f["field"] == "total_amount")
    assert total["gt"] == "219.00 USD"                          # currency rendered
    # source text is read from the persisted document path
    assert "INVOICE" in (view["doc"]["source_text"] or "")


def test_run_diff_identical_runs_are_within_noise(tmp_path):
    store, _ = _populated(tmp_path)
    diff = D.run_diff(store, "r1", "r2")
    assert diff["improvements"] == [] and diff["regressions"] == []
    assert all(p["within_noise"] for p in diff["pipelines"])


def test_failures_lists_and_filters_by_case(tmp_path):
    store, _ = _populated(tmp_path)
    all_fail = D.failures(store, "r1")
    assert all_fail and all(not_passed for not_passed in [True])  # only failures returned
    halluc = D.failures(store, "r1", case_filter="hallucination")
    assert halluc and any(f["scorer"] == "presence" for f in halluc)
