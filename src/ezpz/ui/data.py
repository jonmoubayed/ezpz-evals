"""View-models for the viewer — pure functions over SqliteStore (no web framework, no pandas).

Everything here reads ONLY what a run persisted (results, scores, ground truth, documents); it
never re-runs a pipeline or re-resolves the dataset. Keeping this layer framework-free means the
leaderboard / drill-down / diff / failures / analyze logic is unit-testable without a server, and
the presentation layer (the static SPA in ui/static) can be swapped without touching it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ezpz.core.run import PipelineConfig
from ezpz.core.score import FieldScore
from ezpz.engine.aggregate import (
    calibration_buckets,
    extras_rate,
    paired_compare,
    pipeline_summary,
)
from ezpz.store.sqlite import SqliteStore

_METRIC_KEYS = (
    "accuracy", "accuracy_macro", "ci_low", "ci_high", "hallucinations", "missed",
    "fields_scored", "errors", "cost_usd", "latency_ms", "latency_p50", "docs",
)

_STRATEGY_ADAPTERS = {"cascade", "ensemble", "verify"}

# normalize raw presence/scorer detail cases -> the UI's failure-chip vocabulary
_CASE_NORM = {
    "hallucination": "hallucinated",
    "wrongly_absent": "missing",
    "correctly_absent": "absent_ok",
    "parse_failure": "parse_error",
}


def _label(pc: PipelineConfig) -> str:
    return pc.config.get("label") or pc.adapter


def _is_strategy(pc: PipelineConfig) -> bool:
    return pc.adapter in _STRATEGY_ADAPTERS


def _short_model(model: str) -> str:
    """Trim vendor prefixes/dates so 'models/gemini-2.5-pro-001' -> 'gemini-2.5-pro'."""
    name = str(model).split("/")[-1]
    return name


def _cfg(pc: PipelineConfig) -> str:
    """A short, honest config descriptor built from whatever the pipeline config actually pins."""
    c = pc.config
    parts: list[str] = []
    if _is_strategy(pc):
        stages = c.get("stages") or c.get("members") or c.get("pipeline") or []
        names: list[str] = []
        for s in stages:
            n = s.get("adapter") if isinstance(s, dict) else s
            if n:
                names.append(str(n))
        if names:
            joiner = " → " if pc.adapter in ("cascade", "verify") else " + "
            parts.append(joiner.join(names))
    if c.get("model"):
        parts.append(_short_model(c["model"]))
    if "temperature" in c:
        parts.append(f"temp {c['temperature']}")
    for key in ("schema_mode", "mode", "response_format", "parser"):
        if c.get(key):
            parts.append(str(c[key]))
            break
    return " · ".join(parts) or pc.adapter


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


def _ago(iso: Optional[str]) -> str:
    """Compact relative time ('3h', '2d') for the run menu; '' if unknown."""
    if not iso:
        return ""
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    secs = (datetime.now(timezone.utc) - then).total_seconds()
    if secs < 90:
        return "just now"
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= size:
            return f"{int(secs // size)}{unit}"
    return "just now"


def list_runs(store: SqliteStore) -> list[dict]:
    return [
        {
            "run_id": r.run_id, "dataset": r.dataset_ref, "task": r.task_ref,
            "status": r.status.value, "started_at": r.started_at,
            "pipelines": len(r.pipelines),
        }
        for r in store.list_runs()
    ]


def run_menu(store: SqliteStore) -> list[dict]:
    """Runs for the top-bar switcher: overall accuracy, total cost, doc count, age — newest first."""
    out = []
    for r in reversed(store.list_runs()):  # newest first for the switcher
        scores = store.load_scores(r.run_id)
        results = store.load_results(r.run_id)
        passed = [1.0 if s.passed else 0.0 for s in scores]
        acc = sum(passed) / len(passed) if passed else 0.0
        cost = sum(res.cost.usd for res in results if res.cost and res.cost.usd is not None)
        n_docs = len({res.doc_id for res in results})
        out.append({
            "id": r.run_id, "ds": r.dataset_ref, "task": r.task_ref,
            "n": n_docs, "acc": round(acc, 4), "cost": round(cost, 4),
            "ago": _ago(r.started_at), "note": f"{len(r.pipelines)} pipelines",
            "status": r.status.value,
        })
    return out


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
            "cfg": _cfg(pc), "strategy": _is_strategy(pc),
            **{k: s.get(k, 0) for k in _METRIC_KEYS},
        })
    return rows


def leaderboard_board(store: SqliteStore, run_id: str, slice_tag: Optional[str] = None) -> dict:
    """Leaderboard view-model for the SPA: ranked rows + CI-bar domain + paired-compare footer."""
    rows = leaderboard(store, run_id, slice_tag if slice_tag not in (None, "all") else None)
    rows.sort(key=lambda r: r["accuracy"], reverse=True)

    # CI-bar domain: pad the observed [min low, max high] so whiskers are legible.
    lows = [r["ci_low"] for r in rows if r["fields_scored"]]
    highs = [r["ci_high"] for r in rows if r["fields_scored"]]
    lo0 = max(0.0, (min(lows) if lows else 0.0) - 0.02)
    hi0 = min(1.0, (max(highs) if highs else 1.0) + 0.02)
    if hi0 <= lo0:
        lo0, hi0 = 0.0, 1.0

    scores = store.load_scores(run_id)
    paired = {"lead": "—", "by": "", "verdict": "n/a", "significant": False}
    if len(rows) >= 2 and rows[0]["fields_scored"]:
        a_id, b_id = rows[0]["pipeline_id"], rows[1]["pipeline_id"]
        cmp = paired_compare(
            [s for s in scores if s.pipeline_id == a_id],
            [s for s in scores if s.pipeline_id == b_id],
        )
        noise = cmp["within_noise"]
        paired = {
            "lead": f"{rows[0]['pipeline']} leads {rows[1]['pipeline']}",
            "by": f"{cmp['mean_delta'] * 100:+.1f} [{cmp['ci_low'] * 100:+.1f}, {cmp['ci_high'] * 100:+.1f}]",
            "verdict": "within noise" if noise else "significant",
            "significant": not noise,
        }
    docs = store.load_documents([r.doc_id for r in store.load_results(run_id)])
    tag_counts: dict[str, int] = {}
    for d in docs.values():
        for tag in d.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    slices = [{"id": "all", "label": "all", "n": len(docs)}]
    slices += [{"id": t, "label": t, "n": tag_counts[t]} for t in sorted(tag_counts)]
    n_docs = tag_counts.get(slice_tag, 0) if slice_tag and slice_tag != "all" else len(docs)
    return {
        "board": rows, "domain": {"lo0": lo0, "hi0": hi0},
        "paired": paired, "sliceMeta": f"{n_docs} documents", "slices": slices,
    }


def documents_in_run(store: SqliteStore, run_id: str) -> list[dict]:
    results = store.load_results(run_id)
    scores = store.load_scores(run_id)
    doc_ids = sorted({r.doc_id for r in results})
    docs = store.load_documents(doc_ids)
    acc_by_doc: dict[str, list[float]] = {}
    for s in scores:
        acc_by_doc.setdefault(s.doc_id, []).append(1.0 if s.passed else 0.0)
    out = []
    for doc_id in doc_ids:
        d = docs.get(doc_id)
        passes = acc_by_doc.get(doc_id, [])
        out.append({
            "doc_id": doc_id,
            "slug": (d.slug if d else None) or doc_id[:12],
            "tags": d.tags if d else [],
            "accuracy": round(sum(passes) / len(passes), 4) if passes else 0.0,
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


def _provenance(fv) -> Optional[dict]:
    p = getattr(fv, "provenance", None) if fv else None
    if not p:
        return None
    if p.page is None and p.bbox is None and p.text_span is None:
        return None
    return {"page": p.page, "bbox": p.bbox, "text_span": p.text_span}


def drilldown(store: SqliteStore, run_id: str, doc_id: str) -> dict:
    run = store.load_run(run_id)
    task = store.load_task(run.task_ref)
    scores = [s for s in store.load_scores(run_id) if s.doc_id == doc_id]
    results = {r.pipeline_id: r for r in store.load_results(run_id) if r.doc_id == doc_id}
    gt_fields = store.load_ground_truth(run_id, doc_id) or {}
    doc = store.load_documents([doc_id]).get(doc_id)

    # per-pipeline capability tag, inferred from what the results actually carry for this doc
    caps: dict[str, str] = {}
    for pc in run.pipelines:
        res = results.get(pc.config_hash)
        fields = list(res.fields.values()) if res else []
        bits = []
        if any(getattr(f, "confidence", None) is not None for f in fields):
            bits.append("conf")
        if any(_provenance(f) for f in fields):
            bits.append("bbox")
        caps[pc.config_hash] = " · ".join(bits) if bits else "no conf"

    pipelines = [
        {"pipeline_id": pc.config_hash, "label": _label(pc), "cap": caps[pc.config_hash]}
        for pc in run.pipelines
    ]
    rows = []
    fail_n = 0
    pages = 1  # real page count = max provenance page seen (stays 1 when no adapter emits pages)
    for field in task.fields:
        labeled = field.name in gt_fields
        gt_kind = "not_labeled"
        if labeled:
            gt_kind = "absent" if gt_fields[field.name] == "__ABSENT__" else "present"
        cells = {}
        for pc in run.pipelines:
            pid = pc.config_hash
            res = results.get(pid)
            fv = res.fields.get(field.name) if res else None
            cell_scores = [s for s in scores if s.pipeline_id == pid and s.field == field.name]
            status = _cell_status(cell_scores, labeled)
            if status in ("wrong", "missing", "hallucinated", "parse_error"):
                fail_n += 1
            conf = getattr(fv, "confidence", None) if fv else None
            prov = _provenance(fv)
            if prov and prov.get("page"):
                pages = max(pages, int(prov["page"]))
            cells[pid] = {
                "value": _display(fv.value if fv else None),
                "status": status,
                "confidence": round(conf, 3) if conf is not None else None,
                "provenance": prov,
            }
        rows.append({
            "field": field.name, "type": field.type.value,
            "gt": _display(gt_fields[field.name]) if labeled else "(not labeled)",
            "gt_kind": gt_kind,
            "cells": cells,
        })

    doc_passes = [1.0 if s.passed else 0.0 for s in scores]
    accuracy = round(sum(doc_passes) / len(doc_passes), 4) if doc_passes else 0.0
    mime = (doc.mime if doc else None) or ""
    return {
        "doc": {
            "doc_id": doc_id, "slug": (doc.slug if doc else None) or doc_id[:12],
            "source_text": _source_text(doc), "accuracy": accuracy,
            "summary": f"{fail_n} failing fields · {accuracy * 100:.0f}% accurate",
            "tags": doc.tags if doc else [],
            "pages": pages, "mime": mime,
            "is_image": mime.startswith("image/"),
        },
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


def _score_case(s: FieldScore) -> str:
    """A single field-score's case in the UI vocabulary (correct/wrong/missing/...)."""
    if s.passed:
        return "correct"
    return _CASE_NORM.get(s.detail.get("case", ""), "wrong")


def diff_view(store: SqliteStore, run_a: str, run_b: str) -> dict:
    """Run-diff view-model for the SPA: labelled change rows + improved/regressed/net counts.
    Each row carries the *actual* score case on both sides (e.g. hallucinated → correct)."""
    run = store.load_run(run_a)
    base = store.load_run(run_b)
    raw = run_diff(store, run_a, run_b)
    labels = {pc.config_hash: _label(pc) for pc in run.pipelines}
    docs = store.load_documents([f["doc_id"] for f in raw["improvements"] + raw["regressions"]])

    # case lookups keyed by (pipeline, doc, field, scorer) so a flip shows its real before/after
    def case_index(scores: list[FieldScore]) -> dict[tuple, str]:
        return {(s.pipeline_id, s.doc_id, s.field, s.scorer): _score_case(s) for s in scores}

    a_case = case_index(store.load_scores(run_a))
    b_case = case_index(store.load_scores(run_b))

    def slug(doc_id: str) -> str:
        d = docs.get(doc_id)
        return (d.slug if d else None) or doc_id[:12]

    rows = []
    for direction, flips in (("improved", raw["improvements"]), ("regressed", raw["regressions"])):
        for f in flips:
            key = (f["pipeline_id"], f["doc_id"], f["field"], f["scorer"])
            rows.append({
                "doc": slug(f["doc_id"]), "field": f["field"],
                "pipe": labels.get(f["pipeline_id"], f["pipeline_id"][:8]),
                "dir": direction,
                "from_case": b_case.get(key, "wrong" if direction == "improved" else "correct"),
                "to_case": a_case.get(key, "correct" if direction == "improved" else "wrong"),
            })
    rows.sort(key=lambda r: (r["dir"] != "regressed", r["doc"], r["field"]))
    improved, regressed = len(raw["improvements"]), len(raw["regressions"])
    net = improved - regressed
    return {
        "base": {"id": run_b, "note": f"{len(base.pipelines)} pipelines"},
        "cmp": {"id": run_a, "note": f"{len(run.pipelines)} pipelines"},
        "improved": improved, "regressed": regressed,
        "net": f"{net:+d}" if net else "0",
        "rows": rows,
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


def failure_rows(store: SqliteStore, run_id: str) -> dict:
    """Failure-explorer view-model: every non-passing field-score with predicted + expected values
    and a normalized case, labelled with doc slug + pipeline label. Filtering is done client-side."""
    run = store.load_run(run_id)
    labels = {pc.config_hash: _label(pc) for pc in run.pipelines}
    scores = [s for s in store.load_scores(run_id) if not s.passed]
    doc_ids = sorted({s.doc_id for s in scores})
    docs = store.load_documents(doc_ids)
    results = {(r.pipeline_id, r.doc_id): r for r in store.load_results(run_id)}
    gt_cache: dict[str, dict] = {}

    def gt_for(doc_id: str) -> dict:
        if doc_id not in gt_cache:
            gt_cache[doc_id] = store.load_ground_truth(run_id, doc_id) or {}
        return gt_cache[doc_id]

    rows = []
    counts: dict[str, int] = {}
    for s in scores:
        case = _CASE_NORM.get(s.detail.get("case", ""), "wrong")
        counts[case] = counts.get(case, 0) + 1
        res = results.get((s.pipeline_id, s.doc_id))
        fv = res.fields.get(s.field) if res else None
        gt = gt_for(s.doc_id)
        d = docs.get(s.doc_id)
        rows.append({
            "doc": (d.slug if d else None) or s.doc_id[:12],
            "field": s.field,
            "pipe": labels.get(s.pipeline_id, s.pipeline_id[:8]),
            "case": case,
            "predicted": _display(fv.value if fv else None),
            "expected": _display(gt[s.field]) if s.field in gt else "(not labeled)",
            "scorer": s.scorer,
        })
    rows.sort(key=lambda r: (r["doc"], r["field"], r["pipe"]))
    return {"total": len(rows), "counts": counts, "rows": rows}


def analyze(store: SqliteStore, run_id: str) -> dict:
    """Analyze view-model: confidence-calibration bins, paired comparison vs the leader, and rates
    of boolean strategy flags recorded in result extras (escalation, correction, disagreement…)."""
    run = store.load_run(run_id)
    scores = store.load_scores(run_id)
    results = store.load_results(run_id)
    labels = {pc.config_hash: _label(pc) for pc in run.pipelines}

    bins = calibration_buckets(scores, results)
    n_conf = sum(b["n"] for b in bins)

    # paired comparison: the leading pipeline vs each other, ranked by accuracy.
    summary = pipeline_summary(scores, results)
    ranked = sorted(
        run.pipelines, key=lambda pc: summary.get(pc.config_hash, {}).get("accuracy", 0.0),
        reverse=True,
    )
    paired = []
    if len(ranked) >= 2:
        lead = ranked[0]
        lead_scores = [s for s in scores if s.pipeline_id == lead.config_hash]
        for pc in ranked[1:]:
            cmp = paired_compare(
                lead_scores, [s for s in scores if s.pipeline_id == pc.config_hash]
            )
            paired.append({
                "label": f"{_label(lead)} vs {_label(pc)}",
                "delta": f"{cmp['mean_delta'] * 100:+.1f}pt",
                "ci": f"{(cmp['ci_high'] - cmp['ci_low']) / 2 * 100:.1f}",
                "significant": not cmp["within_noise"],
                "n": cmp["n"],
            })

    # strategy flags: any boolean key recorded in result extras
    flag_keys = sorted({
        k for r in results if r.extras for k, v in r.extras.items() if isinstance(v, bool)
    })
    strategy = []
    for key in flag_keys:
        for pc in run.pipelines:
            rs = [r for r in results if r.pipeline_id == pc.config_hash]
            rate = extras_rate(rs, key)
            if rate is None:
                continue
            n_flag = sum(1 for r in rs if r.extras.get(key))
            strategy.append({
                "label": f"{labels[pc.config_hash]} · {key}",
                "pct": f"{rate * 100:.1f}%", "w": round(rate * 100, 1),
                "note": f"{n_flag} / {len(rs)} docs",
            })

    # overall = pooled accuracy across the confidence-bearing predictions (matches the bins)
    overall = (
        round(sum(b["accuracy"] * b["n"] for b in bins) / n_conf * 100, 1) if n_conf else 0.0
    )
    return {
        "calBins": bins, "nConf": n_conf, "paired": paired, "strategy": strategy,
        "overall": overall,
    }


def estimate(store: SqliteStore, run_id: str, sample: int, cap: float) -> dict:
    """Estimate-only budget model: scale each pipeline's OBSERVED cost/doc (from this cached run) by
    the sample size. Reads SQLite only — never launches a run, never calls a provider."""
    run = store.load_run(run_id)
    summary = pipeline_summary(store.load_scores(run_id), store.load_results(run_id))
    rows = []
    total = 0.0
    for pc in run.pipelines:
        s = summary.get(pc.config_hash, {})
        docs = s.get("docs") or 0
        per = (s.get("cost_usd", 0.0) / docs) if docs else 0.0
        est = per * sample
        total += est
        rows.append({
            "name": _label(pc), "per": round(per, 4), "docs": sample,
            "est": round(est, 2), "over": est > cap,
        })
    over = total > cap
    command = f"ezpz run <experiment.yaml> --sample {sample} --budget-usd {int(cap)}"
    return {
        "rows": rows, "total": round(total, 2), "over": over, "cap": cap,
        "command": command, "dataset": run.dataset_ref, "task": run.task_ref,
    }
