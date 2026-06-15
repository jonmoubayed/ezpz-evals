"""Presence / hallucination handling. Treats the GT states distinctly and gives hallucination
(value returned for a field genuinely absent from the doc) its own metric.

This scorer judges ONLY whether the tool got *presence* right — did it return a value when it
should, and null when the field is genuinely absent. It does not check the value's correctness
(that's the value scorer's job). Run it ALONGSIDE the value scorer.
"""
from ezpz.core.document import ABSENT
from ezpz.scorers.base import Scorer
from ezpz.scorers.registry import register


@register("presence")
class Presence(Scorer):
    name = "presence"

    def score(self, prediction, ground_truth, ctx):
        gt_absent = isinstance(ground_truth, str) and ground_truth == ABSENT
        emitted = prediction is not None  # UNPARSEABLE counts as 'a value was emitted'
        if gt_absent:
            if not emitted:
                return self._result(ctx, 1.0, True, case="correctly_absent")
            return self._result(ctx, 0.0, False, case="hallucination")
        # ground truth has a value
        if emitted:
            return self._result(ctx, 1.0, True, case="present")
        return self._result(ctx, 0.0, False, case="wrongly_absent")
