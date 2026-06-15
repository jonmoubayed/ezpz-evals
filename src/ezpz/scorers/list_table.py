"""List/table scoring — the hard one (line items).

Algorithm:
  1. Align predicted rows to GT rows via spec.match_key (greedy best-match on the key), else by
     overall field similarity.
  2. Score matched pairs field-by-field, delegating to each sub-field's own scorers.
  3. Report set precision/recall/F1; surface missed rows (FN) and extra/hallucinated rows (FP).

The headline value is row-set F1 ("found the right rows"); `field_accuracy` in the detail says
"the rows we found are correct". Both are needed — a tool can find every row yet get the fields
wrong, or nail the fields on only half the rows.
"""
from difflib import SequenceMatcher
from typing import Any, Optional

from ezpz.core.task import FieldSpec, ScorerRef
from ezpz.scorers.base import Scorer, ScoreContext, resolve_empty_states
from ezpz.scorers.registry import get_scorer, register

_MATCH_THRESHOLD = 0.5  # minimum row-similarity to count two rows as the same row
_DEFAULT_EXACT = ScorerRef(name="exact")  # sub-field scorer when a list item declares none


def _str_ratio(a: Any, b: Any) -> float:
    return SequenceMatcher(None, str(a).casefold(), str(b).casefold()).ratio()


@register("list_table")
class ListTable(Scorer):
    name = "list_table"

    def score(self, prediction, ground_truth, ctx):
        empty = resolve_empty_states(prediction, ground_truth)
        if empty is not None:
            value, passed, detail = empty
            return self._result(ctx, value, passed, **detail)

        pred_rows = list(prediction) if isinstance(prediction, (list, tuple)) else []
        gt_rows = list(ground_truth) if isinstance(ground_truth, (list, tuple)) else []
        item = ctx.spec.item
        match_key = ctx.spec.match_key

        matches, n_extra, n_missed = self._align(pred_rows, gt_rows, item, match_key)
        n_match = len(matches)
        precision = n_match / len(pred_rows) if pred_rows else (1.0 if not gt_rows else 0.0)
        recall = n_match / len(gt_rows) if gt_rows else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        passes = [ok for p, g in matches for ok in self._score_row(p, g, item, ctx)]
        field_acc = sum(passes) / len(passes) if passes else (1.0 if n_match else 0.0)

        threshold = float(ctx.config.get("threshold", 0.999))
        passed = f1 >= threshold and field_acc >= threshold
        return self._result(
            ctx, round(f1, 4), passed,
            precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4),
            matched=n_match, extra_rows=n_extra, missed_rows=n_missed,
            field_accuracy=round(field_acc, 4),
        )

    def _align(self, pred, gt, item, match_key):
        """Greedy one-to-one row alignment. Returns (matched_pairs, n_extra_pred, n_missed_gt)."""
        used: set[int] = set()
        matches = []
        for grow in gt:
            best_i, best_s = None, 0.0
            for i, prow in enumerate(pred):
                if i in used:
                    continue
                s = self._row_similarity(prow, grow, item, match_key)
                if s > best_s:
                    best_i, best_s = i, s
            if best_i is not None and best_s >= _MATCH_THRESHOLD:
                used.add(best_i)
                matches.append((pred[best_i], grow))
        return matches, len(pred) - len(used), len(gt) - len(matches)

    def _row_similarity(self, prow, grow, item: Optional[FieldSpec], match_key) -> float:
        if not isinstance(prow, dict) or not isinstance(grow, dict):
            return 0.0
        if match_key:
            return _str_ratio(prow.get(match_key), grow.get(match_key))
        fields = item.fields if item and item.fields else []
        if not fields:
            return 1.0 if prow == grow else 0.0
        sims = [_str_ratio(prow.get(f.name), grow.get(f.name)) for f in fields]
        return sum(sims) / len(sims)

    def _score_row(self, prow, grow, item: Optional[FieldSpec], ctx: ScoreContext) -> list[bool]:
        """Score every sub-field of a matched row via that field's own scorers."""
        results: list[bool] = []
        fields = item.fields if item and item.fields else []
        for field in fields:
            pred_val = prow.get(field.name) if isinstance(prow, dict) else None
            gt_val = grow.get(field.name) if isinstance(grow, dict) else None
            refs = field.scorers or [_DEFAULT_EXACT]
            for ref in refs:
                scorer = get_scorer(ref.name)()
                sub_ctx = ScoreContext(field, ref.config, ctx.doc_id, ctx.pipeline_id)
                results.append(bool(scorer.score(pred_val, gt_val, sub_ctx).passed))
        return results
