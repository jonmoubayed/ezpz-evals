"""Normalization is where comparability is won or lost; test it hard.
Covers the required canonicalizations and the None / ABSENT / UNPARSEABLE distinction."""
from decimal import Decimal

from ezpz.core.document import ABSENT
from ezpz.core.task import FieldSpec
from ezpz.core.values import UNPARSEABLE, CurrencyValue, ValueType
from ezpz.normalize.canonical import normalize_fields, normalize_value


def _spec(t, name="f", **kw):
    return FieldSpec(name=name, type=t, **kw)


def test_currency_messy_and_clean_compare_equal():
    s = _spec(ValueType.CURRENCY)
    a = normalize_value("$1,200.00", s)
    b = normalize_value("1200", s)
    assert isinstance(a, CurrencyValue) and isinstance(b, CurrencyValue)
    assert a.amount == b.amount == Decimal("1200")  # messy "$1,200.00" == clean "1200"
    assert a.currency == "USD"                       # symbol mapped to ISO-4217
    assert b.currency is None


def test_currency_from_ground_truth_dict():
    v = normalize_value({"amount": "1284.50", "currency": "USD"}, _spec(ValueType.CURRENCY))
    assert v.amount == Decimal("1284.50") and v.currency == "USD"


def test_dates_canonicalize_to_iso():
    s = _spec(ValueType.DATE)
    assert normalize_value("Jan 3 2024", s) == "2024-01-03"
    assert normalize_value("2024-01-03", s) == "2024-01-03"
    assert normalize_value("01/03/2024", s) == "2024-01-03"  # US month-first default


def test_number_and_integer():
    assert normalize_value("1,234.56", _spec(ValueType.NUMBER)) == Decimal("1234.56")
    assert normalize_value("10", _spec(ValueType.INTEGER)) == 10
    assert normalize_value("3.5", _spec(ValueType.INTEGER)) is UNPARSEABLE


def test_string_trims_collapses_and_empties_to_none():
    s = _spec(ValueType.STRING)
    assert normalize_value("  Acme   Corp \n", s) == "Acme Corp"
    assert normalize_value("   ", s) is None


def test_three_empty_states_stay_distinct():
    s = _spec(ValueType.STRING)
    assert normalize_value(None, s) is None
    assert normalize_value(ABSENT, s) == ABSENT
    # an ungarbleable value yields UNPARSEABLE, not a wrong-but-parsed value
    assert normalize_value("not-a-number", _spec(ValueType.NUMBER)) is UNPARSEABLE
    assert None is not ABSENT
    assert ABSENT != UNPARSEABLE


def test_normalize_fields_preserves_presence():
    specs = [_spec(ValueType.STRING, "a"), _spec(ValueType.INTEGER, "b")]
    out = normalize_fields({"a": " x "}, specs)
    assert out == {"a": "x"}       # only present keys -> 'b' (not-labeled) stays absent
    assert "b" not in out
