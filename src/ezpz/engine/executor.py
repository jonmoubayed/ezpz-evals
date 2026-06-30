"""The executor runs the docs x pipelines grid.

  cache.get -> (miss) pipeline.extract -> cache.put(raw) -> map -> normalize -> score -> store.

Hardening (M7):
  - Per-provider concurrency (a semaphore per adapter; sync SDKs run in a thread pool). Serial
    when concurrency<=1 (deterministic).
  - Retries with exponential backoff on transient errors (transport/timeout) only.
  - Circuit breaker: after N consecutive failures a provider's remaining cells are short-circuited
    so one dead provider can't stall the run.
  - Budget guard: stop-early once actual spend exceeds the cap.
  - Resumable: cells already completed (status ok) for this run_id are skipped on re-invoke.

Normalization is applied here (centrally), AFTER the adapter's structural map and identically to
ground truth, so no adapter can skew comparability.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from ezpz.core.document import Dataset, GroundTruth
from ezpz.core.result import Cost, ErrorInfo, ExtractionResult, FieldValue, ResultStatus, Timing
from ezpz.core.run import Run, RunStatus
from ezpz.core.score import FieldScore
from ezpz.core.task import ScorerRef, Task
from ezpz.engine.cache import RawResponseCache, cache_key
from ezpz.engine.cost import BudgetGuard
from ezpz.normalize.canonical import normalize_value
from ezpz.scorers.base import ScoreContext
from ezpz.scorers.registry import get_scorer

_RETRYABLE = {"transport", "timeout"}


def normalize_mapped(mapped: dict[str, FieldValue], task: Task) -> dict[str, FieldValue]:
    """Central value-normalization of a mapped prediction (keeps confidence/provenance)."""
    spec_by_name = {f.name: f for f in task.fields}
    return {
        name: FieldValue(
            value=normalize_value(fv.value, spec_by_name[name]),
            confidence=fv.confidence,
            provenance=fv.provenance,
        )
        for name, fv in mapped.items()
        if name in spec_by_name
    }


def score_fields(
    norm_fields: dict[str, FieldValue],
    gt: Optional[GroundTruth],
    task: Task,
    default_scorers: list[ScorerRef],
    doc_id: str,
    pipeline_id: str,
) -> list[FieldScore]:
    """Score each labeled field against normalized GT. Not-labeled GT fields are excluded."""
    if gt is None:
        return []
    spec_by_name = {f.name: f for f in task.fields}
    gt_norm = {
        name: normalize_value(value, spec_by_name[name])
        for name, value in gt.fields.items()
        if name in spec_by_name
    }
    scores: list[FieldScore] = []
    for field in task.fields:
        if field.name not in gt_norm:  # not-labeled -> excluded from the denominator
            continue
        pred = norm_fields[field.name].value if field.name in norm_fields else None
        gt_value = gt_norm[field.name]
        for ref in (field.scorers or default_scorers):
            scorer = get_scorer(ref.name)()
            ctx = ScoreContext(field, ref.config, doc_id=doc_id, pipeline_id=pipeline_id)
            scores.append(scorer.score(pred, gt_value, ctx))
    return scores


def _circuit_result(doc_id: str, pipeline_id: str) -> ExtractionResult:
    return ExtractionResult(
        doc_id=doc_id, pipeline_id=pipeline_id, status=ResultStatus.ERROR,
        error=ErrorInfo(error_class="circuit_open",
                        message="provider circuit-broken after repeated failures"),
        timing=Timing(latency_ms=0.0),
    )


class Executor:
    def __init__(self, store, cache: RawResponseCache, budget=None, retries: int = 2,
                 backoff: float = 0.0, circuit_threshold: int = 4):
        self.store = store
        self.cache = cache
        self.budget = budget
        self.retries = retries
        self.backoff = backoff
        self.circuit_threshold = circuit_threshold

    def run(self, run: Run, dataset: Dataset, task: Task, pipelines, scorers,
            should_stop=None) -> str:
        """Execute the grid; persist results+scores+GT+documents; return run_id.

        `should_stop` is an optional zero-arg predicate checked between cells (like the budget
        guard) so a caller can cancel a long run cooperatively; already-done cells are kept."""
        self.store.save_run(run)
        self.store.save_task(task)
        schema_hash = task.schema_hash()
        cap = run.options.budget_usd if run.options.budget_usd is not None else self.budget
        guard = BudgetGuard(cap)

        gts: dict[str, Optional[GroundTruth]] = {}
        for doc in dataset.documents:
            self.store.save_document(doc, run.dataset_ref)  # so the viewer reads SQLite only
            gt = GroundTruth.load(dataset.root, doc)
            gts[doc.doc_id] = gt
            if gt is not None:
                self.store.save_ground_truth(run.run_id, doc.doc_id, gt.fields)

        done = {  # resume: skip cells already completed for this run_id
            (r.doc_id, r.pipeline_id)
            for r in self.store.load_results(run.run_id)
            if r.status is ResultStatus.OK
        }
        cells = [
            (doc, p) for doc in dataset.documents for p in pipelines
            if (doc.doc_id, p.pipeline_id) not in done
        ]

        concurrency = max(1, run.options.concurrency or 1)
        if concurrency <= 1:
            self._run_serial(run, task, schema_hash, gts, scorers, cells, guard, should_stop)
        else:
            self._run_concurrent(
                run, task, schema_hash, gts, scorers, cells, guard, concurrency, should_stop)

        self.store.finish_run(run.run_id, RunStatus.COMPLETE, _now())
        return run.run_id

    def _save(self, run_id: str, result: ExtractionResult, scores) -> None:
        self.store.save_result(run_id, result)
        self.store.save_scores(run_id, scores)

    def _run_serial(self, run, task, schema_hash, gts, scorers, cells, guard, should_stop=None) -> None:
        consecutive: dict[str, int] = defaultdict(int)
        for doc, pipeline in cells:
            pid = pipeline.pipeline_id
            if guard.over or (should_stop and should_stop()):
                break  # budget stop-early, or cooperative cancel
            if consecutive[pid] >= self.circuit_threshold:
                self._save(run.run_id, _circuit_result(doc.doc_id, pid), [])
                continue
            result, scores = self._cell_with_retries(
                run, doc, task, pipeline, schema_hash, gts.get(doc.doc_id), scorers
            )
            self._save(run.run_id, result, scores)
            if result.status is ResultStatus.ERROR:
                consecutive[pid] += 1
            else:
                consecutive[pid] = 0
                guard.add(result.cost.usd if result.cost else None)

    def _run_concurrent(self, run, task, schema_hash, gts, scorers, cells, guard, concurrency,
                        should_stop=None) -> None:
        lock = threading.Lock()
        adapters = {p.config.adapter for _, p in cells}
        sems = {a: threading.Semaphore(concurrency) for a in adapters}
        consecutive: dict[str, int] = defaultdict(int)
        state = {"stop": False}

        def work(doc, pipeline):
            pid = pipeline.pipeline_id
            with lock:
                if state["stop"] or (should_stop and should_stop()):
                    return None
                if consecutive[pid] >= self.circuit_threshold:
                    return _circuit_result(doc.doc_id, pid), []
            with sems[pipeline.config.adapter]:
                return self._cell_with_retries(
                    run, doc, task, pipeline, schema_hash, gts.get(doc.doc_id), scorers
                )

        with ThreadPoolExecutor(max_workers=concurrency * max(1, len(sems))) as pool:
            for fut in [pool.submit(work, d, p) for d, p in cells]:
                out = fut.result()
                if out is None:
                    continue
                result, scores = out
                with lock:
                    self._save(run.run_id, result, scores)
                    if result.status is ResultStatus.ERROR:
                        consecutive[result.pipeline_id] += 1
                    else:
                        consecutive[result.pipeline_id] = 0
                        guard.add(result.cost.usd if result.cost else None)
                        if guard.over:
                            state["stop"] = True

    def _cell_with_retries(self, run, doc, task, pipeline, schema_hash, gt, scorers):
        attempt = 0
        while True:
            result, scores = self._run_cell(run, doc, task, pipeline, schema_hash, gt, scorers)
            transient = result.error is not None and result.error.error_class in _RETRYABLE
            if result.status is not ResultStatus.ERROR or not transient or attempt >= self.retries:
                return result, scores
            attempt += 1
            if self.backoff:
                time.sleep(self.backoff * (2 ** (attempt - 1)))

    def _run_cell(self, run, doc, task, pipeline, schema_hash, gt, scorers):
        key = cache_key(doc.doc_id, pipeline.pipeline_id, schema_hash)
        cached = None if run.options.force_refresh else self.cache.get(key)

        if cached is not None:
            raw = cached["raw"]
            cost = Cost(**cached["cost"]) if cached["cost"] else None
            norm = normalize_mapped(pipeline.map(raw, task), task)
            extras = raw.get("extras", {}) if isinstance(raw, dict) else {}  # restore strategy meta
            result = ExtractionResult(
                doc_id=doc.doc_id, pipeline_id=pipeline.pipeline_id, status=ResultStatus.OK,
                fields=norm, raw_response_ref=cached["ref"], cost=cost, extras=extras,
                timing=Timing(latency_ms=0.0, stage_ms={"cache_hit": 0.0}),
            )
            scores = score_fields(norm, gt, task, scorers, doc.doc_id, pipeline.pipeline_id)
            return result, scores

        result0, raw = pipeline.extract(doc, task)
        if result0.status != ResultStatus.OK or raw is None:
            return result0, []  # a failed extraction is status=error, never silently-empty fields
        ref = self.cache.put(
            key, raw=raw,
            cost=(result0.cost.model_dump() if result0.cost else None),
            doc_id=doc.doc_id, pipeline_id=pipeline.pipeline_id,
        )
        norm = normalize_mapped(result0.fields, task)
        result = result0.model_copy(update={"fields": norm, "raw_response_ref": ref})
        scores = score_fields(norm, gt, task, scorers, doc.doc_id, pipeline.pipeline_id)
        return result, scores


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
