"""The viewer's JSON API surface, exercised through the pure `api_route` (no socket needed).

Mirrors the SPA's requests so a regression in any endpoint's shape is caught without a browser.
"""
from pathlib import Path

from ezpz.adapters.registry import get_adapter
from ezpz.config.experiment import ExperimentConfig, resolve_dataset, resolve_task
from ezpz.core.run import Run, RunStatus
from ezpz.engine.cache import RawResponseCache
from ezpz.engine.executor import Executor
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore
from ezpz.ui.server import api_route

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
    return store


def test_state_lists_runs_and_picks_current(tmp_path):
    store = _populated(tmp_path)
    status, payload = api_route(store, "/api/state", {})
    assert status == 200
    assert {r["id"] for r in payload["runs"]} == {"r1", "r2"}
    assert payload["current"] in {"r1", "r2"}


def test_leaderboard_endpoint_shape(tmp_path):
    store = _populated(tmp_path)
    status, payload = api_route(store, "/api/leaderboard", {"run": "r1", "slice": "all"})
    assert status == 200 and payload["run"] == "r1"
    assert payload["board"] and "domain" in payload and "paired" in payload


def test_doc_endpoint_defaults_to_first_doc(tmp_path):
    store = _populated(tmp_path)
    status, payload = api_route(store, "/api/doc", {"run": "r1"})
    assert status == 200 and payload["fields"] and payload["pipelines"]


def test_diff_failures_analyze_estimate_endpoints(tmp_path):
    store = _populated(tmp_path)
    for path, q, key in (
        ("/api/diff", {"run": "r1", "base": "r2"}, "rows"),
        ("/api/failures", {"run": "r1"}, "rows"),
        ("/api/analyze", {"run": "r1"}, "calBins"),
        ("/api/estimate", {"run": "r1", "sample": "20", "cap": "5"}, "total"),
    ):
        status, payload = api_route(store, path, q)
        assert status == 200 and key in payload, path


def test_unknown_endpoint_and_missing_run(tmp_path):
    store = _populated(tmp_path)
    status, _ = api_route(store, "/api/nope", {})
    assert status == 404
    empty = SqliteStore(str(tmp_path / "empty.sqlite"))
    empty.init_db()
    status, payload = api_route(empty, "/api/leaderboard", {})
    assert status == 404 and "error" in payload


def test_estimate_rejects_non_numeric(tmp_path):
    store = _populated(tmp_path)
    status, payload = api_route(store, "/api/estimate", {"run": "r1", "sample": "abc"})
    assert status == 400
