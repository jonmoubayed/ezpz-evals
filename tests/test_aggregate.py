"""Aggregation: macro vs micro, bootstrap CIs, slices, presence breakdown, paired compare."""
from ezpz.core.score import FieldScore
from ezpz.engine.aggregate import (
    bootstrap_ci,
    macro_micro,
    paired_compare,
    presence_breakdown,
    slice_metrics,
)


def _fs(doc, field, scorer, passed, **detail):
    return FieldScore(
        doc_id=doc, pipeline_id="p", field=field, scorer=scorer,
        value=1.0 if passed else 0.0, passed=passed, detail=detail,
    )


def test_macro_and_micro_answer_different_questions():
    # field A: 1/1 pass; field B: 1/3 pass -> micro = 2/4 = 0.5; macro = (1 + 1/3) / 2 ≈ 0.667
    scores = [
        _fs("d1", "A", "exact", True),
        _fs("d1", "B", "exact", True),
        _fs("d2", "B", "exact", False),
        _fs("d3", "B", "exact", False),
    ]
    micro, macro = macro_micro(scores)
    assert micro == 0.5
    assert round(macro, 3) == 0.667


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([1.0] * 8 + [0.0] * 2, seed=1)  # mean 0.8
    assert lo <= 0.8 <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_slice_metrics_split_by_tag():
    scores = [_fs("d1", "A", "exact", True), _fs("d2", "A", "exact", False)]
    out = slice_metrics(scores, {"d1": ["clean"], "d2": ["scanned"]})
    assert out["clean"] == {"accuracy": 1.0, "n": 1}
    assert out["scanned"] == {"accuracy": 0.0, "n": 1}


def test_presence_breakdown_counts_hallucinations():
    scores = [
        _fs("d1", "po", "presence", False, case="hallucination"),
        _fs("d2", "po", "presence", True, case="correctly_absent"),
        _fs("d1", "x", "exact", True),  # non-presence scores ignored
    ]
    b = presence_breakdown(scores)
    assert b["hallucination"] == 1 and b["correctly_absent"] == 1


def test_paired_compare_flags_within_noise():
    a = [_fs("d1", "A", "exact", True), _fs("d2", "A", "exact", True)]
    b = [_fs("d1", "A", "exact", True), _fs("d2", "A", "exact", False)]
    out = paired_compare(a, b)
    assert out["n"] == 2
    assert out["mean_delta"] == 0.5  # a beats b on d2
    # identical pipelines -> zero delta, within noise
    same = paired_compare(a, a)
    assert same["mean_delta"] == 0.0 and same["within_noise"] is True
