"""The viewer's one write path: launching a budget-gated re-run from a stored run.

Exercises `ezpz.ui.launch.launch` end-to-end (reconstruct experiment from the stored run → execute
in a background thread) plus its refusal/error guards. The `fake` adapter keeps it network-free.
"""
import time
from pathlib import Path

from ezpz.adapters.registry import get_adapter
from ezpz.config.experiment import ExperimentConfig, resolve_dataset, resolve_task
from ezpz.core.run import PipelineConfig, Run, RunStatus
from ezpz.engine.cache import RawResponseCache
from ezpz.engine.executor import Executor
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore
from ezpz.ui import launch as L

REPO = Path(__file__).resolve().parents[1]
ROOT = str(REPO / "examples")


def _populated(tmp_path):
    db = str(tmp_path / "db.sqlite")
    store = SqliteStore(db)
    store.init_db()
    cache = RawResponseCache(store, BlobStore(str(tmp_path / "blobs")))
    exp = ExperimentConfig.load_yaml(str(REPO / "examples" / "experiments" / "fake_invoice_smoke.yaml"))
    dataset = resolve_dataset(ROOT, exp.dataset)
    task = resolve_task(ROOT, exp.task)
    pipelines = [get_adapter(p.adapter)(p) for p in exp.pipelines]
    run = Run(run_id="r1", dataset_ref=exp.dataset, task_ref=exp.task, pipelines=exp.pipelines,
              scorers=exp.scorers, options=exp.options, status=RunStatus.RUNNING)
    Executor(store, cache).run(run, dataset, task, pipelines, exp.scorers)
    return db, store


def _wait(run_id, want="complete", timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = L.job_status(run_id)
        if job["status"] not in ("running", "starting"):
            return job
        time.sleep(0.05)
    return L.job_status(run_id)


def test_launch_reruns_experiment_and_persists_a_new_run(tmp_path):
    db, store = _populated(tmp_path)
    before = {r.run_id for r in store.list_runs()}
    job = L.launch(db, ROOT, "r1", sample=1, cap=10.0)
    assert job["status"] == "running" and job["run_id"] not in before
    done = _wait(job["run_id"])
    assert done["status"] == "complete", done
    after = {r.run_id for r in store.list_runs()}
    assert job["run_id"] in after                       # the re-run is a real, stored run
    assert store.load_results(job["run_id"])            # with results
    assert store.load_scores(job["run_id"])             # and scores
    new = store.load_run(job["run_id"])
    assert new.env.get("launched_from") == "r1"


def test_launch_refuses_when_estimate_exceeds_cap(tmp_path):
    db, store = _populated(tmp_path)
    # a source run whose pipeline carries a positive cost prior -> estimate > tiny cap
    pricey = Run(
        run_id="pricey", dataset_ref="fake_invoices_v1@1", task_ref="fake_invoice@1",
        pipelines=[PipelineConfig(adapter="fake", config={"label": "p", "cost_prior_usd": 9.0})],
        scorers=store.load_run("r1").scorers, status=RunStatus.COMPLETE,
    )
    store.save_run(pricey)
    job = L.launch(db, ROOT, "pricey", sample=1, cap=1.0)
    assert job["status"] == "refused" and job["estimate_usd"] >= 9.0
    # refusing must NOT create a run
    assert job["run_id"] not in {r.run_id for r in store.list_runs()}


def test_launch_errors_clearly_on_unknown_run_and_unresolvable_root(tmp_path):
    db, _ = _populated(tmp_path)
    assert L.launch(db, ROOT, "nope", sample=1, cap=10.0)["status"] == "error"
    bad_root = str(tmp_path / "not-an-eval-project")
    job = L.launch(db, bad_root, "r1", sample=1, cap=10.0)
    assert job["status"] == "error" and "resolve" in job["error"]
