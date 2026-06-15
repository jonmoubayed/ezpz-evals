"""Composite extraction strategies (cascade / ensemble / verify) + calibration / escalation metrics.
Runnable with no keys — every tier is the deterministic `fake` adapter."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ezpz.adapters.registry import get_adapter
from ezpz.cli.main import app
from ezpz.config.experiment import ExperimentConfig, resolve_dataset, resolve_task
from ezpz.core.run import Run, RunStatus
from ezpz.engine.aggregate import calibration, extras_rate, pipeline_summary
from ezpz.engine.cache import RawResponseCache
from ezpz.engine.executor import Executor
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore

REPO = Path(__file__).resolve().parents[1]
ROOT = str(REPO / "examples")
DEMO = str(REPO / "examples" / "experiments" / "strategies_demo.yaml")


def _run_demo(tmp_path):
    store = SqliteStore(str(tmp_path / "db.sqlite"))
    store.init_db()
    cache = RawResponseCache(store, BlobStore(str(tmp_path / "blobs")))
    exp = ExperimentConfig.load_yaml(DEMO)
    dataset = resolve_dataset(ROOT, exp.dataset)
    task = resolve_task(ROOT, exp.task)
    pipelines = [get_adapter(p.adapter)(p) for p in exp.pipelines]
    run = Run(run_id="r1", dataset_ref=exp.dataset, task_ref=exp.task, pipelines=exp.pipelines,
              scorers=exp.scorers, options=exp.options, status=RunStatus.RUNNING)
    Executor(store, cache).run(run, dataset, task, pipelines, exp.scorers)
    return store, {pc.config.get("label"): pc.config_hash for pc in exp.pipelines}


def test_methodologies_rank_and_cost_as_expected(tmp_path):
    store, label = _run_demo(tmp_path)
    summary = pipeline_summary(store.load_scores("r1"), store.load_results("r1"))
    acc = {lbl: summary[pid]["accuracy"] for lbl, pid in label.items()}
    assert acc["cheap"] == pytest.approx(0.5)
    assert acc["expensive"] == 1.0
    assert acc["cascade-0.7"] == pytest.approx(4 / 6, abs=1e-3)   # cheap on a/c, escalated b
    assert acc["ensemble-3"] == pytest.approx(5 / 6, abs=1e-3)    # vote fixes b, inherits c's majority
    assert acc["extract-then-verify"] == 1.0                     # verifier corrects every disagreement
    cost = {lbl: summary[pid]["cost_usd"] for lbl, pid in label.items()}
    assert cost["cheap"] < cost["cascade-0.7"] < cost["expensive"]  # cascade ≈ cheap, not always-expensive


def test_cascade_escalation_rate_persists_in_extras(tmp_path):
    store, label = _run_demo(tmp_path)
    results = [r for r in store.load_results("r1") if r.pipeline_id == label["cascade-0.7"]]
    assert extras_rate(results, "escalated") == pytest.approx(1 / 3, abs=1e-3)  # only doc-b escalated
    assert all("tier_used" in r.extras for r in results)             # extras round-trip the store


def test_calibration_flags_overconfidence(tmp_path):
    store, label = _run_demo(tmp_path)
    cheap = label["cheap"]
    cal = calibration([s for s in store.load_scores("r1") if s.pipeline_id == cheap],
                      [r for r in store.load_results("r1") if r.pipeline_id == cheap])
    # cheap is confident (0.90) on the WRONG doc-c, so its most-confident slice isn't all-correct
    assert cal["n"] > 0
    assert cal["accuracy_at_auto"] < 1.0


def test_run_then_analyze_via_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_res = CliRunner().invoke(app, ["run", DEMO, "--json"])
    assert run_res.exit_code == 0
    run_id = json.loads(run_res.stdout)["run_id"]
    analyze_res = CliRunner().invoke(app, ["analyze", run_id])
    assert analyze_res.exit_code == 0
    assert "escalated" in analyze_res.stdout  # the cascade flag is auto-discovered as a column
