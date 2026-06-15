"""M4 scorers: presence, date_match, string_similarity, list_table, embedding_similarity, llm_judge."""
from decimal import Decimal

from ezpz.core.document import ABSENT
from ezpz.core.task import FieldSpec, ScorerRef
from ezpz.core.values import UNPARSEABLE, ValueType
from ezpz.scorers.base import ScoreContext
from ezpz.scorers.embedding_similarity import EmbeddingSimilarity
from ezpz.scorers.list_table import ListTable
from ezpz.scorers.llm_judge import LLMJudge
from ezpz.scorers.registry import get_scorer


def _ctx(t=ValueType.STRING, spec=None, **cfg):
    return ScoreContext(spec or FieldSpec(name="f", type=t), cfg, doc_id="d", pipeline_id="p")


# ---- presence (the hallucination metric) ----

def test_presence_cases():
    presence = get_scorer("presence")()
    assert presence.score(None, ABSENT, _ctx()).detail["case"] == "correctly_absent"
    assert presence.score("PO-7", ABSENT, _ctx()).detail["case"] == "hallucination"
    assert presence.score(None, "INV-1", _ctx()).detail["case"] == "wrongly_absent"
    assert presence.score("INV-1", "INV-1", _ctx()).detail["case"] == "present"
    # an UNPARSEABLE value still counts as 'a value was emitted' for presence
    assert presence.score(UNPARSEABLE, "INV-1", _ctx()).detail["case"] == "present"


# ---- date_match ----

def test_date_match_exact_and_tolerance():
    dm = get_scorer("date_match")()
    assert dm.score("2024-01-15", "2024-01-15", _ctx(ValueType.DATE)).passed is True
    assert dm.score("2024-01-17", "2024-01-15", _ctx(ValueType.DATE, tolerance_days=3)).passed is True
    assert dm.score("2024-02-15", "2024-01-15", _ctx(ValueType.DATE, tolerance_days=0)).passed is False


# ---- string_similarity ----

def test_string_similarity_is_continuous_and_thresholded():
    ss = get_scorer("string_similarity")()
    exact = ss.score("Acme Corporation", "Acme Corporation", _ctx(threshold=0.9))
    assert exact.value == 1.0 and exact.passed
    close = ss.score("Acme Corp", "Acme Corporation", _ctx(threshold=0.9))
    assert 0.0 < close.value < 1.0
    far = ss.score("ACME Inc", "Acme Corporation", _ctx(threshold=0.9))
    assert far.passed is False


# ---- list_table (alignment + per-field scoring + set P/R/F1) ----

def _items_spec() -> FieldSpec:
    return FieldSpec(
        name="line_items", type=ValueType.LIST, match_key="description",
        item=FieldSpec(name="li", type=ValueType.OBJECT, fields=[
            FieldSpec(name="description", type=ValueType.STRING,
                      scorers=[ScorerRef(name="string_similarity", config={"threshold": 0.85})]),
            FieldSpec(name="quantity", type=ValueType.NUMBER,
                      scorers=[ScorerRef(name="numeric_tolerance", config={"abs": 0})]),
        ]),
    )


def _row(desc, qty):
    return {"description": desc, "quantity": Decimal(str(qty))}


def test_list_table_perfect_match_reordered():
    spec = _items_spec()
    gt = [_row("Widget A", 10), _row("Setup fee", 1)]
    pred = [_row("Setup fee", 1), _row("Widget A", 10)]  # reordered, still perfect
    fs = ListTable().score(pred, gt, _ctx(spec=spec))
    assert fs.detail["f1"] == 1.0 and fs.detail["field_accuracy"] == 1.0 and fs.passed


def test_list_table_missed_row_lowers_recall():
    spec = _items_spec()
    fs = ListTable().score([_row("Widget A", 10)], [_row("Widget A", 10), _row("Setup fee", 1)],
                           _ctx(spec=spec))
    assert fs.detail["missed_rows"] == 1 and fs.detail["recall"] == 0.5


def test_list_table_extra_row_is_hallucinated_fp():
    spec = _items_spec()
    fs = ListTable().score([_row("Widget A", 10), _row("Bogus", 5)], [_row("Widget A", 10)],
                           _ctx(spec=spec))
    assert fs.detail["extra_rows"] == 1 and fs.detail["precision"] == 0.5


def test_list_table_matched_row_with_wrong_field():
    spec = _items_spec()
    fs = ListTable().score([_row("Widget A", 99)], [_row("Widget A", 10)], _ctx(spec=spec))
    assert fs.detail["f1"] == 1.0           # the row was found (matched on description)
    assert fs.detail["field_accuracy"] < 1.0  # but its quantity is wrong


# ---- embedding_similarity (cosine over cached embeddings; backend injected) ----

class _LocalEmbedding(EmbeddingSimilarity):
    def _embed(self, text, model, ctx):
        v = [0.0] * 26
        for ch in text.lower():
            if "a" <= ch <= "z":
                v[ord(ch) - 97] += 1
        return v


def test_embedding_similarity_cosine_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EZPZ_HOME", str(tmp_path))
    emb = _LocalEmbedding()
    same = emb.score("hello world", "hello world", _ctx(threshold=0.9))
    assert same.value == 1.0 and same.passed
    diff = emb.score("abc", "xyz", _ctx(threshold=0.9))
    assert diff.value < 0.9 and not diff.passed
    assert list((tmp_path / "score_cache").glob("*.json"))  # embeddings were cached


# ---- llm_judge (rubric score; backend injected; cached) ----

class _StubJudge(LLMJudge):
    def _judge(self, prediction, ground_truth, ctx):
        return {"score": 1.0 if prediction == ground_truth else 0.0, "rationale": "stub"}


def test_llm_judge_scores_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("EZPZ_HOME", str(tmp_path))
    judge = _StubJudge()
    good = judge.score("same", "same", _ctx(threshold=0.5))
    assert good.value == 1.0 and good.passed and good.detail["rationale"] == "stub"
    bad = judge.score("a", "b", _ctx(threshold=0.5))
    assert bad.value == 0.0 and not bad.passed
    assert list((tmp_path / "score_cache").glob("*.json"))  # judge calls were cached
