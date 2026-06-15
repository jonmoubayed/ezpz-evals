"""The canonical value-type system.

Every field in a task declares one of these types. The *canonical Python representation*
(documented per type) is what scorers compare. Turning a tool's raw value into the canonical
representation is done centrally in ``ezpz.normalize`` — NOT in each adapter — so that all
tools normalize identically.
"""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ValueType(str, Enum):
    STRING = "string"        # canonical: str (trimmed, whitespace-collapsed)
    INTEGER = "integer"      # canonical: int
    NUMBER = "number"        # canonical: float | Decimal
    BOOLEAN = "boolean"      # canonical: bool
    DATE = "date"            # canonical: ISO-8601 date string "YYYY-MM-DD"
    DATETIME = "datetime"    # canonical: ISO-8601 datetime string
    ENUM = "enum"            # canonical: one of FieldSpec.enum_values
    CURRENCY = "currency"    # canonical: CurrencyValue
    LIST = "list"            # canonical: list[<item canonical>]
    OBJECT = "object"        # canonical: dict[str, <field canonical>]


class CurrencyValue(BaseModel):
    amount: Decimal
    currency: Optional[str] = None  # ISO-4217 code, e.g. "USD"; None if undetermined


class _Unparseable:
    """Sentinel: a value was returned but could not be canonicalized.

    Kept distinct from ``None`` (field missing / not extracted) and from ``ABSENT``
    (field genuinely not present — a *correct* answer). Scorers count it as a parse
    failure rather than silently scoring it wrong. A singleton, so ``is`` works.
    """

    _instance: Optional[_Unparseable] = None

    def __new__(cls) -> _Unparseable:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNPARSEABLE"

    def __reduce__(self) -> tuple[type, tuple]:
        return (_Unparseable, ())


UNPARSEABLE = _Unparseable()  # the typed parse-failure marker (see ezpz.normalize)
