"""`self_rate` is a PROVIDER-NEUTRAL flag on the shared LLM base: any LLM adapter (gemini /
anthropic / openai / llamaindex) can emit a per-field confidence that a cascade gates on. These
tests exercise it through the base + a provider-neutral stub — no SDK, no network."""
import json

from ezpz.adapters.llm_base import LLMExtractionPipeline
from ezpz.adapters.registry import get_adapter, register
from ezpz.core.document import Dataset
from ezpz.core.run import PipelineConfig, Run, RunStatus
from ezpz.core.task import FieldSpec, ScorerRef, Task
from ezpz.core.values import ValueType
from ezpz.engine.cache import RawResponseCache
from ezpz.engine.executor import Executor
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore


def _task() -> Task:
    return Task(name="t", version="1", fields=[
        FieldSpec(name="invoice_number", type=ValueType.STRING, scorers=[ScorerRef(name="exact")]),
        FieldSpec(name="total_amount", type=ValueType.CURRENCY,
                  scorers=[ScorerRef(name="numeric_tolerance", config={"abs": 0.01})]),
    ])


# A provider-neutral LLM adapter: the technique lives entirely in the shared base, so a one-line
# `_generate` stub stands in for ANY real provider (it's what gemini/openai/anthropic differ on).
@register("stub_llm")
class _StubLLM(LLMExtractionPipeline):
    default_model = "stub-1"

    def _generate(self, prepared, ingested):
        return json.dumps(self.config.config["canned"]), {"input_tokens": 10, "output_tokens": 5}, {}


def _pipe(**cfg) -> _StubLLM:
    return _StubLLM(PipelineConfig(adapter="stub_llm", config=cfg))


def test_self_rate_is_off_by_default_for_every_llm_adapter():
    # The base advertises no confidence unless self_rate is set — true for all real subclasses too.
    for adapter in ("anthropic", "gemini", "openai"):
        pipe = get_adapter(adapter)(PipelineConfig(adapter=adapter, config={}))
        assert pipe.capabilities.confidence is False
        rated = get_adapter(adapter)(PipelineConfig(adapter=adapter, config={"self_rate": True}))
        assert rated.capabilities.confidence is True


def test_self_rate_injects_per_field_confidence_into_schema_and_prompt():
    prepared = _pipe(self_rate=True).compile(_task())
    conf_schema = prepared["schema"]["properties"]["_confidence"]
    assert set(conf_schema["properties"]) == {"invoice_number", "total_amount"}
    assert "_confidence" in prepared["schema"]["required"]
    assert "confidence" in prepared["prompt"].lower()


def test_without_self_rate_schema_is_untouched():
    prepared = _pipe().compile(_task())
    assert "_confidence" not in prepared["schema"]["properties"]
    assert "_confidence" not in prepared["schema"]["required"]


def test_map_attaches_self_rated_confidence_and_drops_the_confidence_field():
    raw = {"fields": {
        "invoice_number": "INV-1",
        "total_amount": {"amount": 100.0, "currency": "USD"},
        "_confidence": {"invoice_number": 0.4, "total_amount": 0.9},
    }}
    mapped = _pipe(self_rate=True).map(raw, _task())
    assert set(mapped) == {"invoice_number", "total_amount"}  # _confidence is not a task field
    assert mapped["invoice_number"].confidence == 0.4
    assert mapped["total_amount"].confidence == 0.9


def test_map_leaves_confidence_none_when_self_rate_off():
    raw = {"fields": {"invoice_number": "INV-1",
                      "total_amount": {"amount": 100.0, "currency": "USD"}}}
    mapped = _pipe().map(raw, _task())
    assert mapped["invoice_number"].confidence is None
    assert mapped["total_amount"].confidence is None


# --- a real-shaped cascade: a low-confidence self-rating tier escalates to a strong tier ---


def _dataset(tmp_path) -> Dataset:
    ds = tmp_path / "ds"
    (ds / "docs").mkdir(parents=True)
    (ds / "ground_truth").mkdir(parents=True)
    (ds / "docs" / "d1.txt").write_text("Invoice INV-1 total $100.00")
    (ds / "ground_truth" / "d1.json").write_text(json.dumps({"fields": {
        "invoice_number": "INV-1", "total_amount": {"amount": "100.00", "currency": "USD"}}}))
    (ds / "manifest.jsonl").write_text(json.dumps({
        "slug": "d1", "path": "docs/d1.txt", "mime": "text/plain",
        "ground_truth_path": "ground_truth/d1.json"}) + "\n")
    return Dataset.load_from_manifest(str(ds), "ds", "1")


def test_cascade_escalates_on_low_self_rated_confidence(tmp_path):
    store = SqliteStore(str(tmp_path / "db.sqlite"))
    store.init_db()
    cache = RawResponseCache(store, BlobStore(str(tmp_path / "blobs")))
    cascade_cfg = PipelineConfig(adapter="cascade", config={
        "label": "cheap->strong", "threshold": 0.7,
        "tiers": [
            # The cheap tier is any LLM adapter with self_rate on — here the provider-neutral stub.
            {"adapter": "stub_llm", "config": {"self_rate": True, "canned": {
                "invoice_number": "WRONG",
                "total_amount": {"amount": 0.0, "currency": "USD"},
                "_confidence": {"invoice_number": 0.3, "total_amount": 0.3}}}},
            {"adapter": "fake", "config": {
                "cost_usd": 0.01,
                "by_slug": {"d1": {"invoice_number": "INV-1", "total_amount": "$100.00"}}}},
        ],
    })
    run = Run(run_id="r1", dataset_ref="ds@1", task_ref="t@1", pipelines=[cascade_cfg],
              scorers=[], status=RunStatus.RUNNING)
    Executor(store, cache).run(run, _dataset(tmp_path), _task(), [get_adapter("cascade")(cascade_cfg)], [])

    result = store.load_results("r1")[0]
    assert result.extras["escalated"] is True                  # cheap tier was unsure (0.3 < 0.7)
    assert result.fields["invoice_number"].value == "INV-1"    # took the strong tier's answer
