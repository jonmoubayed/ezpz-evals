"""Canonicalize raw values into the representation documented in ezpz.core.values.

Why central: if each adapter normalized dates/currency its own way, "Jan 3 2024" from one
tool and "2024-01-03" from another would not compare equal and the whole comparison would lie.

Permissive on input, strict on output. The three empty states are preserved exactly:
``None`` (missing) / ``ABSENT`` (correctly-absent) / ``UNPARSEABLE`` (returned but un-canonical).
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from ezpz.core.document import ABSENT
from ezpz.core.task import FieldSpec
from ezpz.core.values import UNPARSEABLE, CurrencyValue, ValueType

# Explicit maps (be explicit; never guess silently). See PLAN §13 for locale/dayfirst policy.
_CURRENCY_SYMBOLS = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
_CURRENCY_CODES = ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF")
_BOOL_TRUE = {"true", "yes", "y", "1", "t", "✓"}
_BOOL_FALSE = {"false", "no", "n", "0", "f", "✗"}
# Ambiguous D/M vs M/D defaults to month-first (US); make configurable later (PLAN §13).
_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d",
    "%b %d %Y", "%B %d %Y", "%b %d, %Y", "%B %d, %Y",
    "%d %b %Y", "%d %B %Y",
    "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y",
)


def normalize_value(raw: Any, spec: FieldSpec) -> Any:
    """Return the canonical representation of ``raw`` for the given field spec."""
    # Preserve the three empty states before any type dispatch.
    if raw is None:
        return None
    if raw is UNPARSEABLE:
        return UNPARSEABLE
    if isinstance(raw, str) and raw == ABSENT:
        return ABSENT

    t = spec.type
    if t == ValueType.STRING:
        return _norm_string(raw)
    if t == ValueType.INTEGER:
        return _norm_integer(raw)
    if t == ValueType.NUMBER:
        return _norm_number(raw)
    if t == ValueType.BOOLEAN:
        return _norm_boolean(raw)
    if t == ValueType.DATE:
        return _norm_date(raw)
    if t == ValueType.DATETIME:
        return _norm_datetime(raw)
    if t == ValueType.ENUM:
        return _norm_enum(raw, spec)
    if t == ValueType.CURRENCY:
        return _norm_currency(raw)
    if t == ValueType.LIST:
        return _norm_list(raw, spec)
    if t == ValueType.OBJECT:
        return _norm_object(raw, spec)
    return UNPARSEABLE


def normalize_fields(raw_fields: dict[str, Any], specs: list[FieldSpec]) -> dict[str, Any]:
    """Normalize a field dict against specs. Used for predictions and ground truth.

    Only keys present in ``raw_fields`` are emitted, so a not-labeled ground-truth field
    (key omitted) stays absent and a missed prediction is the caller's ``.get`` -> None.
    """
    spec_by_name = {f.name: f for f in specs}
    return {
        name: normalize_value(value, spec_by_name[name])
        for name, value in raw_fields.items()
        if name in spec_by_name
    }


# ---- per-type canonicalizers (strict output; UNPARSEABLE on failure) ----

def _norm_string(raw: Any) -> Any:
    s = unicodedata.normalize("NFC", str(raw))
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _strip_number(s: str) -> str:
    # Keep digits, decimal point, and sign; drop currency symbols, codes, thousands separators.
    return re.sub(r"[^\d.\-]", "", s.strip())


def _norm_integer(raw: Any) -> Any:
    if isinstance(raw, bool):
        return UNPARSEABLE
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else UNPARSEABLE
    try:
        d = Decimal(_strip_number(str(raw)))
    except InvalidOperation:
        return UNPARSEABLE
    return int(d) if d == d.to_integral_value() else UNPARSEABLE


def _norm_number(raw: Any) -> Any:
    if isinstance(raw, bool):
        return UNPARSEABLE
    if isinstance(raw, Decimal):
        return raw
    if isinstance(raw, int):
        return Decimal(raw)
    if isinstance(raw, float):
        return Decimal(str(raw))
    try:
        return Decimal(_strip_number(str(raw)))
    except InvalidOperation:
        return UNPARSEABLE


def _norm_boolean(raw: Any) -> Any:
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    return UNPARSEABLE


def _norm_date(raw: Any) -> Any:
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    s = str(raw).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return UNPARSEABLE


def _norm_datetime(raw: Any) -> Any:
    if isinstance(raw, datetime):
        return raw.isoformat()
    try:
        return datetime.fromisoformat(str(raw).strip()).isoformat()
    except ValueError:
        return UNPARSEABLE


def _norm_enum(raw: Any, spec: FieldSpec) -> Any:
    s = _norm_string(raw)
    if s is None:
        return None
    for member in spec.enum_values or []:
        if s.lower() == member.lower():
            return member
    return UNPARSEABLE


def _detect_currency(s: str) -> Any:
    upper = s.upper()
    for code in _CURRENCY_CODES:
        if re.search(rf"\b{code}\b", upper):  # word-boundary: "AUDIT" must not match AUD
            return code
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in s:
            return code
    return None


def _norm_currency(raw: Any) -> Any:
    if isinstance(raw, CurrencyValue):
        return raw
    if isinstance(raw, dict):
        amount = _norm_number(raw.get("amount"))
        if amount is UNPARSEABLE or amount is None:
            return UNPARSEABLE
        return CurrencyValue(amount=amount, currency=raw.get("currency"))
    s = str(raw)
    amount = _norm_number(s)
    if amount is UNPARSEABLE or amount is None:
        return UNPARSEABLE
    return CurrencyValue(amount=amount, currency=_detect_currency(s))


def _norm_list(raw: Any, spec: FieldSpec) -> Any:
    if not isinstance(raw, (list, tuple)):
        return UNPARSEABLE
    if spec.item is None:
        return list(raw)
    return [normalize_value(x, spec.item) for x in raw]


def _norm_object(raw: Any, spec: FieldSpec) -> Any:
    if not isinstance(raw, dict):
        return UNPARSEABLE
    return {f.name: normalize_value(raw.get(f.name), f) for f in (spec.fields or [])}
