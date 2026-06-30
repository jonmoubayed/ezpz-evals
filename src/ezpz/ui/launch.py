"""Run launcher for the viewer — the one write path in the UI.

The browser's budget modal POSTs here to *re-run an existing experiment* with a new sample size and
budget cap. We reconstruct the experiment from the stored Run (its dataset_ref / task_ref /
pipelines / scorers), resolve the dataset + task from the project root, enforce the budget the same
way the CLI does (estimate first, refuse if over cap), then execute in a background thread so the
HTTP request returns immediately with a new run_id the UI can poll.

This is deliberately the *only* part of the viewer that is not read-only; everything else reads
SQLite. Cells already cached cost nothing, so re-running is cheap unless inputs changed.
"""
from __future__ import annotations

import platform
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ezpz.adapters.registry import get_adapter
from ezpz.config.experiment import resolve_dataset, resolve_task
from ezpz.core.result import ResultStatus
from ezpz.core.run import Run, RunStatus
from ezpz.engine.aggregate import pipeline_summary
from ezpz.engine.cache import RawResponseCache
from ezpz.engine.cost import estimate_cost
from ezpz.engine.executor import Executor
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore

# new_run_id -> {status, run_id, source, ...}. Shared across handler threads (module-level).
_JOBS: dict[str, dict] = {}
_CANCELS: dict[str, threading.Event] = {}   # run_id -> cooperative-cancel flag
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_status(run_id: str) -> dict:
    with _LOCK:
        job = _JOBS.get(run_id)
        return dict(job) if job else {"status": "unknown", "run_id": run_id}


def cancel(run_id: str) -> dict:
    """Request cooperative cancellation; the run stops after the in-flight cell (kept)."""
    with _LOCK:
        ev = _CANCELS.get(run_id)
        if ev is None or _JOBS.get(run_id, {}).get("status") not in ("running", "starting"):
            return {"status": _JOBS.get(run_id, {}).get("status", "unknown"), "run_id": run_id}
        ev.set()
        _JOBS[run_id]["status"] = "cancelling"
        return dict(_JOBS[run_id])


def launch(db_path: str, root: str, source_run_id: str, sample: int, cap: float) -> dict:
    """Validate + budget-check synchronously; on success spawn the run and return its new id.

    Returns a job dict: status is one of refused | error | running (then complete | failed via
    job_status). Never raises for the expected failure modes — the UI shows the message."""
    with _LOCK:  # one run at a time — this is a local single-user tool
        if any(j["status"] in ("running", "starting", "cancelling") for j in _JOBS.values()):
            busy = next(j for j in _JOBS.values() if j["status"] in ("running", "starting", "cancelling"))
            return {"status": "busy", "error": f"a run is already in progress ({busy['run_id'][:7]})",
                    "run_id": busy["run_id"]}

    store = SqliteStore(db_path)
    store.init_db()
    try:
        src = store.load_run(source_run_id)
    except Exception as e:
        return {"status": "error", "error": f"unknown run {source_run_id}: {e}"}

    # reconstruct the experiment from the stored run + the on-disk dataset/task
    try:
        dataset = resolve_dataset(root, src.dataset_ref)
        task = resolve_task(root, src.task_ref)
    except Exception as e:
        return {"status": "error", "error": (
            f"could not resolve {src.dataset_ref} / {src.task_ref} from {root} — launch the viewer "
            f"from your eval project directory (where datasets/ and tasks/ live). ({e})")}

    if sample:
        dataset = dataset.sample(sample)
    pipelines = [get_adapter(pc.adapter)(pc) for pc in src.pipelines]
    options = src.options.model_copy(update={"budget_usd": cap, "sample": sample or None})
    new_run = Run(
        run_id=uuid.uuid4().hex[:12], dataset_ref=src.dataset_ref, task_ref=src.task_ref,
        pipelines=src.pipelines, scorers=src.scorers, options=options,
        env={"python": platform.python_version(), "launched_from": source_run_id},
        status=RunStatus.RUNNING, started_at=_now(),
    )

    # blobs live next to the DB (…/.ezpz/blobs), not relative to the server's cwd
    cache = RawResponseCache(store, BlobStore(str(Path(db_path).parent / "blobs")))
    est = estimate_cost(new_run, dataset, cache, task.schema_hash())
    if cap is not None and est["estimate_usd"] > cap:
        return {
            "status": "refused", "run_id": new_run.run_id,
            "error": f"estimate ${est['estimate_usd']:.4f} exceeds cap ${cap:.2f}",
            "estimate_usd": est["estimate_usd"],
        }

    cancel_ev = threading.Event()
    job = {
        "status": "running", "run_id": new_run.run_id, "source": source_run_id,
        "estimate_usd": est["estimate_usd"], "uncached_cells": est["uncached_cells"],
        "dataset": src.dataset_ref, "started_at": new_run.started_at,
    }
    with _LOCK:
        _JOBS[new_run.run_id] = job
        _CANCELS[new_run.run_id] = cancel_ev

    def worker() -> None:
        try:
            Executor(store, cache, cap).run(
                new_run, dataset, task, pipelines, src.scorers, should_stop=cancel_ev.is_set)
            results = store.load_results(new_run.run_id)
            summary = pipeline_summary(store.load_scores(new_run.run_id), results)
            errors = sum(1 for r in results if r.status is ResultStatus.ERROR)
            with _LOCK:
                _JOBS[new_run.run_id].update(
                    status="cancelled" if cancel_ev.is_set() else "complete",
                    errors=errors, pipelines=len(summary), finished_at=_now(),
                )
        except Exception as e:  # surface, never crash the server thread
            with _LOCK:
                _JOBS[new_run.run_id].update(status="failed", error=str(e), finished_at=_now())

    threading.Thread(target=worker, name=f"ezpz-run-{new_run.run_id}", daemon=True).start()
    return job
