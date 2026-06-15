"""View-models for the viewer — pure functions over SqliteStore (no Streamlit, no pandas).

Everything here reads ONLY what a run persisted (results, scores, ground truth, documents); it
never re-runs a pipeline or re-resolves the dataset. Keeping this layer framework-free means the
leaderboard / drill-down / diff / failures logic is unit-testable without the `ui` extra, and the
UI framework can be swapped without touching it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ezpz.core.score import FieldScore
from ezpz.engine.aggregate import paired_compare, pipeline_summary
from ezpz.store.sqlite import SqliteStore

_METRIC_KEYS = (
    "accuracy", "accuracy_macro", "ci_low", "ci_high",
    "hallucinations", "missed", "fields_scored", "errors", "cost_usd", "latency_ms",
)


def _label(pc) -> str:
    return pc.config.get("label") or pc.adapter


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if value == "__ABSENT__":
        return "ABSENT"
    if value == "__UNPARSEABLE__":
        return "UNPARSEABLE"
    if isinstance(value, dict) and "amount" in value:
        cur = value.get("currency") or ""
        return f"{value.get('amount')} {cur}".strip()
    if isinstance(value, list):
        return f"[{len(value)} rows]"
    return str(value)


def _cell_status(cell_scores: list[FieldScore], labeled: bool) -> str:
    if not labeled:
        return "not_labeled"
    cases = {s.detail.get("case") for s in cell_scores}
    if "hallucination" in cases:
        return "hallucinated"
    if "correctly_absent" in cases:
        return "absent_ok"
    if "wrongly_absent" in cases:
        return "missing"
    if "parse_failure" in cases:
        return "parse_error"
    if not cell_scores:
        return "unscored"
    return "correct" if all(s.passed for s in cell_scores) else "wrong"


def list_runs(store: SqliteStore) -> list[dict]:
    return [
        {
            "run_id": r.run_id, "dataset": r.dataset_ref, "task": r.task_ref,
            "status": r.status.value, "started_at": r.started_at,
            "pipelines": len(r.pipelines),
        }
        for r in store.list_runs()
    ]


def slice_tags(store: SqliteStore, run_id: str) -> list[str]:
    docs = store.load_documents([r.doc_id for r in store.load_results(run_id)])
    return sorted({tag for d in docs.values() for tag in d.tags})


def leaderboard(store: SqliteStore, run_id: str, slice_tag: Optional[str] = None) -> list[dict]:
    run = store.load_run(run_id)
    scores = store.load_scores(run_id)
    results = store.load_results(run_id)
    if slice_tag:
        docs = store.load_documents([r.doc_id for r in results])
        keep = {doc_id for doc_id, d in docs.items() if slice_tag in d.tags}
        scores = [s for s in scores if s.doc_id in keep]
        results = [r for r in results if r.doc_id in keep]
    summary = pipeline_summary(scores, results)
    rows = []
    for pc in run.pipelines:
        s = summary.get(pc.config_hash, {})
        rows.append({
            "pipeline": _label(pc), "adapter": pc.adapter, "pipeline_id": pc.config_hash,
            **{k: s.get(k, 0) for k in _METRIC_KEYS},
        })
    return rows


def documents_in_run(store: SqliteStore, run_id: str) -> list[dict]:
    doc_ids = sorted({r.doc_id for r in store.load_results(run_id)})
    docs = store.load_documents(doc_ids)
    out = []
    for doc_id in doc_ids:
        d = docs.get(doc_id)
        out.append({
            "doc_id": doc_id,
            "slug": (d.slug if d else None) or doc_id[:12],
            "tags": d.tags if d else [],
        })
    return sorted(out, key=lambda x: x["slug"])


def _source_text(doc) -> Optional[str]:
    if not doc or not doc.source_path:
        return None
    path = Path(doc.source_path)
    mime = doc.mime or ""
    if not path.exists() or not (mime.startswith("text/") or path.suffix in (".txt", ".md")):
        return None
    return path.read_text(errors="replace")[:4000]


def drilldown(store: SqliteStore, run_id: str, doc_id: str) -> dict:
    run = store.load_run(run_id)
    task = store.load_task(run.task_ref)
    scores = [s for s in store.load_scores(run_id) if s.doc_id == doc_id]
    results = {r.pipeline_id: r for r in store.load_results(run_id) if r.doc_id == doc_id}
    gt_fields = store.load_ground_truth(run_id, doc_id) or {}
    doc = store.load_documents([doc_id]).get(doc_id)

    pipelines = [{"pipeline_id": pc.config_hash, "label": _label(pc)} for pc in run.pipelines]
    rows = []
    for field in task.fields:
        labeled = field.name in gt_fields
        cells = {}
        for pc in run.pipelines:
            pid = pc.config_hash
            res = results.get(pid)
            fv = res.fields.get(field.name) if res else None
            cell_scores = [s for s in scores if s.pipeline_id == pid and s.field == field.name]
            cells[pid] = {
                "value": _display(fv.value if fv else None),
                "status": _cell_status(cell_scores, labeled),
            }
        rows.append({
            "field": field.name, "type": field.type.value,
            "gt": _display(gt_fields[field.name]) if labeled else "(not labeled)",
            "cells": cells,
        })
    return {
        "doc": {"doc_id": doc_id, "slug": (doc.slug if doc else None) or doc_id[:12],
                "source_text": _source_text(doc)},
        "pipelines": pipelines,
        "fields": rows,
    }


def run_diff(store: SqliteStore, run_a: str, run_b: str) -> dict:
    """run_a is the NEW run, run_b the baseline. A field flipping fail->pass is an improvement."""
    run = store.load_run(run_a)
    a_scores, b_scores = store.load_scores(run_a), store.load_scores(run_b)

    pipelines = []
    for pc in run.pipelines:
        pid = pc.config_hash
        cmp = paired_compare(
            [s for s in a_scores if s.pipeline_id == pid],
            [s for s in b_scores if s.pipeline_id == pid],
        )
        pipelines.append({"label": _label(pc), "pipeline_id": pid, **cmp})

    def index(scores):
        return {(s.pipeline_id, s.doc_id, s.field, s.scorer): bool(s.passed) for s in scores}

    ai, bi = index(a_scores), index(b_scores)
    flips = []
    for key in sorted(set(ai) & set(bi)):
        if ai[key] != bi[key]:
            flips.append({
                "pipeline_id": key[0], "doc_id": key[1], "field": key[2], "scorer": key[3],
                "from": "pass" if bi[key] else "fail", "to": "pass" if ai[key] else "fail",
            })
    return {
        "pipelines": pipelines,
        "improvements": [f for f in flips if f["to"] == "pass"],
        "regressions": [f for f in flips if f["to"] == "fail"],
    }


def failures(store: SqliteStore, run_id: str, case_filter: Optional[str] = None) -> list[dict]:
    out = []
    for s in store.load_scores(run_id):
        if s.passed:
            continue
        case = s.detail.get("case", "")
        if case_filter and case != case_filter:
            continue
        out.append({
            "doc_id": s.doc_id, "pipeline_id": s.pipeline_id, "field": s.field,
            "scorer": s.scorer, "value": s.value, "case": case, "detail": s.detail,
        })
    return out
