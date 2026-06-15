"""Scorers must implement the §5.3 GT-state matrix, not just value comparison."""
from decimal import Decimal

from ezpz.core.document import ABSENT
from ezpz.core.task import FieldSpec
from ezpz.core.values import UNPARSEABLE, CurrencyValue, ValueType
from ezpz.scorers.base import ScoreContext
from ezpz.scorers.registry import get_scorer


def _ctx(t=ValueType.STRING, **config):
    return ScoreContext(FieldSpec(name="f", type=t), config, doc_id="d", pipeline_id="p")


def _exact():
    return get_scorer("exact")()


def _numeric():
    return get_scorer("numeric_tolerance")()


def test_exact_present_value_match_and_mismatch():
    assert _exact().score("INV-1", "INV-1", _ctx()).passed is True
    miss = _exact().score("INV-2", "INV-1", _ctx())
    assert miss.passed is False and miss.value == 0.0


def test_exact_gt_absent_pred_none_is_correctly_absent():
    fs = _exact().score(None, ABSENT, _ctx())
    assert fs.passed is True and fs.value == 1.0
    assert fs.detail["case"] == "correctly_absent"


def test_exact_gt_absent_pred_value_is_hallucination():
    fs = _exact().score("PO-9", ABSENT, _ctx())
    assert fs.passed is False and fs.detail["case"] == "hallucination"


def test_exact_gt_present_pred_none_is_wrongly_absent():
    fs = _exact().score(None, "INV-1", _ctx())
    assert fs.passed is False and fs.detail["case"] == "wrongly_absent"


def test_exact_unparseable_is_parse_failure():
    fs = _exact().score(UNPARSEABLE, "INV-1", _ctx())
    assert fs.passed is False and fs.detail["case"] == "parse_failure"


def test_field_score_carries_identity():
    fs = _exact().score("INV-1", "INV-1", _ctx())
    assert fs.doc_id == "d" and fs.pipeline_id == "p" and fs.field == "f" and fs.scorer == "exact"


def test_numeric_tolerance_currency_within_abs():
    a = CurrencyValue(amount=Decimal("1200.00"))
    b = CurrencyValue(amount=Decimal("1200.004"))
    assert _numeric().score(a, b, _ctx(ValueType.CURRENCY, abs=0.01)).passed is True


def test_numeric_tolerance_outside_abs_fails():
    assert _numeric().score(Decimal("10"), Decimal("12"), _ctx(ValueType.NUMBER, abs=0.5)).passed is False


def test_numeric_tolerance_relative():
    assert _numeric().score(Decimal("102"), Decimal("100"), _ctx(ValueType.NUMBER, rel=0.05)).passed is True


def test_numeric_tolerance_gt_absent_pred_none_correct():
    fs = _numeric().score(None, ABSENT, _ctx(ValueType.NUMBER, abs=0.01))
    assert fs.passed is True and fs.detail["case"] == "correctly_absent"
