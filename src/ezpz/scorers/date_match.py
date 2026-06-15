"""Date equality on canonical ISO dates. config: {tolerance_days: int} optional."""
from datetime import date

from ezpz.scorers.base import Scorer, resolve_empty_states
from ezpz.scorers.registry import register


def _parse(value) -> "date | None":
    try:
        return date.fromisoformat(str(value)[:10])  # handles 'YYYY-MM-DD' and ISO datetimes
    except (ValueError, TypeError):
        return None


@register("date_match")
class DateMatch(Scorer):
    name = "date_match"

    def score(self, prediction, ground_truth, ctx):
        empty = resolve_empty_states(prediction, ground_truth)
        if empty is not None:
            value, passed, detail = empty
            return self._result(ctx, value, passed, **detail)

        p, g = _parse(prediction), _parse(ground_truth)
        if p is None or g is None:
            return self._result(ctx, 0.0, False, case="parse_failure")
        tolerance = int(ctx.config.get("tolerance_days", 0) or 0)
        diff = abs((p - g).days)
        ok = diff <= tolerance
        return self._result(ctx, 1.0 if ok else 0.0, ok, diff_days=diff)
