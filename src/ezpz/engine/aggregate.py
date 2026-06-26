"""Aggregation over stored FieldScores. Always recomputable from raw per-field results.

Provides: per-field accuracy, macro vs micro averaging (they answer different questions),
per-slice breakdowns, bootstrap confidence intervals + n, a presence/hallucination breakdown,
and paired comparison between two pipelines on the same docs (with a within-noise flag).
"""
from __future__ import annotations

import random
from collections import defaultdict

from ezpz.core.result import ExtractionResult, ResultStatus
from ezpz.core.score import FieldScore


def bootstrap_ci(
    values: list[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of 0/1 (or continuous) values."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)
    )
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (round(lo, 4), round(hi, 4))


def _passes(scores: list[FieldScore]) -> list[float]:
    return [1.0 if s.passed else 0.0 for s in scores]


def macro_micro(scores: list[FieldScore]) -> tuple[float, float]:
    """(micro, macro): micro pools all field-scores; macro averages per-field accuracies."""
    if not scores:
        return (0.0, 0.0)
    micro = sum(_passes(scores)) / len(scores)
    by_field: dict[str, list[FieldScore]] = defaultdict(list)
    for s in scores:
        by_field[s.field].append(s)
    per_field = [sum(_passes(g)) / len(g) for g in by_field.values()]
    macro = sum(per_field) / len(per_field)
    return (round(micro, 4), round(macro, 4))


def slice_metrics(scores: list[FieldScore], doc_tags: dict[str, list[str]]) -> dict[str, dict]:
    """Accuracy + n per slice tag, using a doc_id -> tags map."""
    by_tag: dict[str, list[FieldScore]] = defaultdict(list)
    for s in scores:
        for tag in doc_tags.get(s.doc_id, []):
            by_tag[tag].append(s)
    out = {}
    for tag, group in by_tag.items():
        out[tag] = {"accuracy": round(sum(_passes(group)) / len(group), 4), "n": len(group)}
    return out


def presence_breakdown(scores: list[FieldScore]) -> dict[str, int]:
    """Counts of correctly_absent / hallucination / wrongly_absent / present from presence scores."""
    counts: dict[str, int] = defaultdict(int)
    for s in scores:
        if s.scorer == "presence":
            counts[s.detail.get("case", "unknown")] += 1
    return dict(counts)


def pipeline_summary(
    scores: list[FieldScore], results: list[ExtractionResult]
) -> dict[str, dict]:
    """Per-pipeline micro/macro accuracy (+ bootstrap CI), hallucinations, cost/latency/errors."""
    summary: dict[str, dict] = {}

    by_pipe: dict[str, list[FieldScore]] = defaultdict(list)
    for s in scores:
        by_pipe[s.pipeline_id].append(s)
    for pid, group in by_pipe.items():
        micro, macro = macro_micro(group)
        lo, hi = bootstrap_ci(_passes(group))
        breakdown = presence_breakdown(group)
        summary[pid] = {
            "accuracy": micro,
            "accuracy_macro": macro,
            "fields_scored": len(group),
            "ci_low": lo,
            "ci_high": hi,
            "hallucinations": breakdown.get("hallucination", 0),
            "missed": breakdown.get("wrongly_absent", 0),
        }

    by_res: dict[str, list[ExtractionResult]] = defaultdict(list)
    for r in results:
        by_res[r.pipeline_id].append(r)
    for pid, rs in by_res.items():
        d = summary.setdefault(pid, {"accuracy": 0.0, "fields_scored": 0})
        costs = [r.cost.usd for r in rs if r.cost and r.cost.usd is not None]
        lats = [r.timing.latency_ms for r in rs if r.timing and r.timing.latency_ms is not None]
        d["docs"] = len(rs)
        d["errors"] = sum(1 for r in rs if r.status is ResultStatus.ERROR)
        d["cost_usd"] = sum(costs) if costs else 0.0
        d["latency_ms"] = sum(lats) / len(lats) if lats else 0.0
        d["latency_p50"] = percentile(lats, 50) if lats else 0.0
    return summary


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). Stable for the small per-run sample sizes here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def paired_compare(
    scores_a: list[FieldScore], scores_b: list[FieldScore]
) -> dict:
    """Compare two pipelines on the SAME (doc, field, scorer) cells: mean delta + CI + noise flag."""
    def index(scores):
        return {(s.doc_id, s.field, s.scorer): (1.0 if s.passed else 0.0) for s in scores}

    a, b = index(scores_a), index(scores_b)
    common = sorted(set(a) & set(b))
    deltas = [a[k] - b[k] for k in common]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    lo, hi = bootstrap_ci(deltas) if deltas else (0.0, 0.0)
    return {
        "n": len(common),
        "mean_delta": round(mean_delta, 4),
        "ci_low": lo,
        "ci_high": hi,
        "within_noise": lo <= 0.0 <= hi,  # CI straddles zero -> difference is within noise
    }


def calibration(scores, results, automation_rate: float = 0.5) -> dict:
    """Is the emitted confidence informative? Reframes raw accuracy as 'accuracy at an automation
    rate': if you auto-accept the most-confident `automation_rate` of predictions, how accurate are
    they vs the overall accuracy. Joins each field-score's pass with that field's confidence."""
    conf = {
        (r.doc_id, r.pipeline_id, name): fv.confidence
        for r in results for name, fv in r.fields.items()
    }
    pairs = [
        (conf[(s.doc_id, s.pipeline_id, s.field)], 1.0 if s.passed else 0.0)
        for s in scores if conf.get((s.doc_id, s.pipeline_id, s.field)) is not None
    ]
    if not pairs:
        return {"n": 0}
    overall = sum(p for _, p in pairs) / len(pairs)
    pairs.sort(key=lambda x: x[0], reverse=True)  # most-confident first
    k = max(1, int(len(pairs) * automation_rate))
    return {
        "n": len(pairs),
        "overall_accuracy": round(overall, 4),
        "automation_rate": automation_rate,
        "accuracy_at_auto": round(sum(p for _, p in pairs[:k]) / k, 4),
    }


def calibration_buckets(scores, results, n_bins: int = 5) -> list[dict]:
    """Reliability diagram: bin predictions by emitted confidence, report accuracy + n per bin.

    Joins each field-score's pass/fail with that field's confidence (same join as `calibration`),
    then buckets into `n_bins` equal-width bins over [0, 1]. Only non-empty bins are returned, each
    as {lo, hi, accuracy, n}. A well-calibrated model has accuracy ≈ bin midpoint."""
    conf = {
        (r.doc_id, r.pipeline_id, name): fv.confidence
        for r in results for name, fv in r.fields.items()
    }
    bins: list[list[float]] = [[] for _ in range(n_bins)]
    for s in scores:
        c = conf.get((s.doc_id, s.pipeline_id, s.field))
        if c is None:
            continue
        idx = min(n_bins - 1, max(0, int(c * n_bins)))
        bins[idx].append(1.0 if s.passed else 0.0)
    out = []
    for i, group in enumerate(bins):
        if not group:
            continue
        out.append({
            "lo": round(i / n_bins, 4),
            "hi": round((i + 1) / n_bins, 4),
            "accuracy": round(sum(group) / len(group), 4),
            "n": len(group),
        })
    return out


def extras_rate(results, key: str) -> "float | None":
    """Fraction of a pipeline's results whose extras[key] is truthy (e.g. 'escalated'). None if the
    flag was never recorded."""
    rs = [r for r in results if r.extras and key in r.extras]
    if not rs:
        return None
    return round(sum(1 for r in rs if r.extras.get(key)) / len(rs), 4)
