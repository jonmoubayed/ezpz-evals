"""Numeric match within tolerance. config: {abs: float} and/or {rel: float}.
For amounts/quantities; absorbs float noise and rounding."""
from decimal import Decimal, InvalidOperation

from ezpz.core.values import CurrencyValue
from ezpz.scorers.base import Scorer, resolve_empty_states
from ezpz.scorers.registry import register


def _amount(value):
    """Extract a Decimal magnitude from a canonical numeric/currency value, or None."""
    if isinstance(value, CurrencyValue):
        return value.amount
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


@register("numeric_tolerance")
class NumericTolerance(Scorer):
    name = "numeric_tolerance"

    def score(self, prediction, ground_truth, ctx):
        empty = resolve_empty_states(prediction, ground_truth)
        if empty is not None:
            value, passed, detail = empty
            return self._result(ctx, value, passed, **detail)

        # Differing *explicit* currencies are wrong regardless of amount; a missing currency
        # (undetermined) is a wildcard, so a tool that emits no ISO code isn't penalized.
        pc = prediction.currency if isinstance(prediction, CurrencyValue) else None
        gc = ground_truth.currency if isinstance(ground_truth, CurrencyValue) else None
        if pc and gc and pc != gc:
            return self._result(ctx, 0.0, False, case="currency_mismatch", pred=pc, gt=gc)

        p, g = _amount(prediction), _amount(ground_truth)
        if p is None or g is None:
            return self._result(ctx, 0.0, False, case="parse_failure")

        abs_tol = Decimal(str(ctx.config.get("abs", 0) or 0))
        rel_tol = Decimal(str(ctx.config.get("rel", 0) or 0))
        diff = abs(p - g)
        ok = diff <= abs_tol or (rel_tol > 0 and diff <= rel_tol * abs(g))
        return self._result(ctx, 1.0 if ok else 0.0, bool(ok), diff=str(diff))
