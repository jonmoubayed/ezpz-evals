"""Exact match after normalization. Baseline for IDs, enums, structured values.

Note: exact equality is strict by design — for CurrencyValue it compares amount AND currency.
Use numeric_tolerance (currency-aware) for amounts; exact is for IDs/enums/strings.
"""
from ezpz.scorers.base import Scorer, resolve_empty_states
from ezpz.scorers.registry import register


@register("exact")
class ExactMatch(Scorer):
    name = "exact"

    def score(self, prediction, ground_truth, ctx):
        empty = resolve_empty_states(prediction, ground_truth)
        if empty is not None:
            value, passed, detail = empty
            return self._result(ctx, value, passed, **detail)
        passed = prediction == ground_truth
        return self._result(ctx, 1.0 if passed else 0.0, passed)
