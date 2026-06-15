"""Normalized string similarity (edit-distance ratio / token-sort). config: {threshold, method}.
For names/addresses where minor variation is acceptable. Keeps a continuous score."""
from difflib import SequenceMatcher

from ezpz.scorers.base import Scorer, resolve_empty_states
from ezpz.scorers.registry import register


def _ratio(a: str, b: str, method: str) -> float:
    a, b = a.casefold().strip(), b.casefold().strip()
    if method == "token_sort":
        a = " ".join(sorted(a.split()))
        b = " ".join(sorted(b.split()))
    return SequenceMatcher(None, a, b).ratio()


@register("string_similarity")
class StringSimilarity(Scorer):
    name = "string_similarity"

    def score(self, prediction, ground_truth, ctx):
        empty = resolve_empty_states(prediction, ground_truth)
        if empty is not None:
            value, passed, detail = empty
            return self._result(ctx, value, passed, **detail)

        threshold = float(ctx.config.get("threshold", 0.85))
        method = ctx.config.get("method", "ratio")
        sim = _ratio(str(prediction), str(ground_truth), method)
        return self._result(ctx, round(sim, 4), sim >= threshold, similarity=round(sim, 4))
