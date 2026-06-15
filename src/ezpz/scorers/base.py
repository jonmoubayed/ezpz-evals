"""Scorer base. A scorer is a pure function over canonical values + context (the field spec)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ezpz.core.document import ABSENT
from ezpz.core.score import FieldScore
from ezpz.core.task import FieldSpec
from ezpz.core.values import UNPARSEABLE


class ScoreContext:
    """Carried into every scorer so it can be type-, config-, and identity-aware.

    The engine sets ``doc_id``/``pipeline_id`` so the scorer can emit a complete FieldScore.
    """

    def __init__(
        self,
        spec: FieldSpec,
        config: Optional[dict] = None,
        doc_id: str = "",
        pipeline_id: str = "",
    ):
        self.spec = spec
        self.config = config or {}
        self.doc_id = doc_id
        self.pipeline_id = pipeline_id


def resolve_empty_states(prediction: Any, ground_truth: Any) -> Optional[tuple[float, bool, dict]]:
    """Apply the §5.3 GT-state matrix for the non-comparison cells.

    Returns ``(value, passed, detail)`` for those cells, or ``None`` to proceed with a real
    value comparison. Assumes 'not-labeled' ground truth is already excluded by the engine.
    """
    gt_absent = isinstance(ground_truth, str) and ground_truth == ABSENT
    if gt_absent:
        if prediction is None:
            return (1.0, True, {"case": "correctly_absent"})
        if prediction is UNPARSEABLE:
            return (0.0, False, {"case": "parse_failure"})
        return (0.0, False, {"case": "hallucination"})
    # ground truth is a present value
    if prediction is UNPARSEABLE:
        return (0.0, False, {"case": "parse_failure"})
    if prediction is None:
        return (0.0, False, {"case": "wrongly_absent"})
    return None


class Scorer(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, prediction: Any, ground_truth: Any, ctx: ScoreContext) -> FieldScore:
        """Compare one predicted canonical value to its GT. Return a FieldScore (0..1 + passed).

        Implementations MUST handle the GT states explicitly:
          present-correct, present-wrong, correctly-absent, wrongly-absent, hallucinated.
        """

    def _result(
        self, ctx: ScoreContext, value: float, passed: bool, **detail: Any
    ) -> FieldScore:
        return FieldScore(
            doc_id=ctx.doc_id,
            pipeline_id=ctx.pipeline_id,
            field=ctx.spec.name,
            scorer=self.name,
            value=float(value),
            passed=passed,
            detail=detail,
        )
