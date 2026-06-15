"""LLM extraction adapters (Gemini / Anthropic / OpenAI) share one base. We test the shared
contract (schema compile, prompt, map, cost, error classification) and run the full pipeline
end-to-end with a stub `_generate` — no SDKs, no network, no cost."""
import json
from decimal import Decimal
from pathlib import Path

import pytest

from ezpz.adapters.llm_base import (
    LLMExtractionPipeline,
    RefusalError,
    build_prompt,
    task_to_json_schema,
)
from ezpz.adapters.registry import get_adapter
from ezpz.core.document import Dataset
from ezpz.core.result import ResultStatus
from ezpz.core.run import PipelineConfig, Run, RunStatus
from ezpz.core.task import FieldSpec, ScorerRef, Task
from ezpz.core.values import ValueType
from ezpz.engine.aggregate import pipeline_summary
from ezpz.engine.cache import RawResponseCache
from ezpz.engine.executor import Executor
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore


def _task() -> Task:
    return Task(
        name="inv", version="1",
        fields=[
            FieldSpec(name="invoice_number", type=ValueType.STRING, description="The invoice id",
                      scorers=[ScorerRef(name="exact")]),
            FieldSpec(name="total_amount", type=ValueType.CURRENCY,
                      scorers=[ScorerRef(name="numeric_tolerance", config={"abs": 0.01})]),
            FieldSpec(name="status", type=ValueType.ENUM, enum_values=["paid", "due"]),
            FieldSpec(name="line_items", type=ValueType.LIST,
                      item=FieldSpec(name="li", type=ValueType.OBJECT, fields=[
                          FieldSpec(name="desc", type=ValueType.STRING),
                          FieldSpec(name="qty", type=ValueType.INTEGER),
                      ])),
        ],
    )


# ---- schema compilation ----

def test_schema_is_strict_and_nullable():
    schema = task_to_json_schema(_task())
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"invoice_number", "total_amount", "status", "line_items"}
    # every field is nullable so 'absent' is expressible as JSON null
    assert schema["properties"]["invoice_number"]["type"] == ["string", "null"]
    # currency compiles to a nested object
    cur = schema["properties"]["total_amount"]
    assert cur["type"] == ["object", "null"]
    assert set(cur["required"]) == {"amount", "currency"}
    assert cur["additionalProperties"] is False
    # enum carries its members plus null
    assert schema["properties"]["status"]["enum"] == ["paid", "due", None]
    # list of objects recurses
    items = schema["properties"]["line_items"]["items"]
    assert items["type"] == ["object", "null"]
    assert set(items["required"]) == {"desc", "qty"}


def test_prompt_lists_fields_and_does_not_restate_schema():
    prompt = build_prompt(_task())
    assert "invoice_number" in prompt and "The invoice id" in prompt
    assert "additionalProperties" not in prompt  # don't echo the raw schema (redundant)


def test_builtin_llm_adapters_are_registered():
    for name in ("gemini", "anthropic", "openai"):
        assert get_adapter(name) is not None  # importable + registered without their SDKs


# ---- stub subclass exercises the shared template without any SDK ----

class _StubLLM(LLMExtractionPipeline):
    default_model = "stub-model"
    default_prices = {"stub-model": (1.0, 2.0)}  # $1/$2 per 1M tokens

    def _generate(self, prepared, ingested):
        cfg = self.config.config
        if cfg.get("emit_garbage"):
            return "this is not json", {"input_tokens": 5, "output_tokens": 1}, {}
        usage = {"input_tokens": 100, "output_tokens": 20}
        return json.dumps(cfg["canned"]), usage, {"finish_reason": "stop"}


def _stub(canned=None, **extra) -> _StubLLM:
    cfg = {"canned": canned or {}, **extra}
    return _StubLLM(PipelineConfig(adapter="stub", config=cfg))


def test_map_reads_parsed_fields_and_missing_become_none():
    pipe = _stub()
    raw = {"fields": {"invoice_number": "INV-1"}}
    mapped = pipe.map(raw, _task())
    assert mapped["invoice_number"].value == "INV-1"
    assert mapped["total_amount"].value is None          # missing -> None
    assert mapped["invoice_number"].confidence is None   # these tools emit no confidence


def test_cost_computed_from_usage_and_price_table():
    cost = _stub()._cost({"input_tokens": 100, "output_tokens": 20})
    # 100/1e6*1 + 20/1e6*2 = 0.0001 + 0.00004
    assert cost.usd == pytest.approx(0.00014)
    assert cost.input_tokens == 100 and cost.output_tokens == 20


def test_anthropic_has_authoritative_prices():
    pipe = get_adapter("anthropic")(PipelineConfig(adapter="anthropic", config={}))
    assert pipe.model == "claude-opus-4-8"
    assert pipe._cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000}).usd == 30.0


def test_classify_error_maps_refusal_and_parse():
    pipe = _stub()
    assert pipe.classify_error(RefusalError("no")) == "refusal"
    assert pipe.classify_error(json.JSONDecodeError("x", "", 0)) == "parse"
    assert pipe.classify_error(TimeoutError()) == "timeout"
    assert pipe.classify_error(ValueError("?")) == "unknown"


def _write_dataset(tmp_path: Path) -> Dataset:
    ds = tmp_path / "ds"
    (ds / "docs").mkdir(parents=True)
    (ds / "ground_truth").mkdir(parents=True)
    (ds / "docs" / "d1.txt").write_text("ACME INVOICE\nInvoice #: INV-1\nTotal: $100.00")
    (ds / "ground_truth" / "d1.json").write_text(json.dumps({
        "fields": {
            "invoice_number": "INV-1",
            "total_amount": {"amount": "100.00", "currency": "USD"},
        }
    }))
    (ds / "manifest.jsonl").write_text(json.dumps({
        "slug": "d1", "path": "docs/d1.txt", "mime": "text/plain",
        "ground_truth_path": "ground_truth/d1.json",
    }) + "\n")
    return Dataset.load_from_manifest(str(ds), "ds", "1")


def _small_task() -> Task:
    return Task(name="t", version="1", fields=[
        FieldSpec(name="invoice_number", type=ValueType.STRING, scorers=[ScorerRef(name="exact")]),
        FieldSpec(name="total_amount", type=ValueType.CURRENCY,
                  scorers=[ScorerRef(name="numeric_tolerance", config={"abs": 0.01})]),
    ])


def test_llm_pipeline_runs_end_to_end_through_executor(tmp_path):
    store = SqliteStore(str(tmp_path / "db.sqlite"))
    store.init_db()
    cache = RawResponseCache(store, BlobStore(str(tmp_path / "blobs")))
    dataset = _write_dataset(tmp_path)
    task = _small_task()
    pipe = _stub(canned={
        "invoice_number": "INV-1",
        "total_amount": {"amount": 100.0, "currency": "USD"},
    })
    run = Run(run_id="r1", dataset_ref="ds@1", task_ref="t@1",
              pipelines=[pipe.config], scorers=[], status=RunStatus.RUNNING)

    Executor(store, cache).run(run, dataset, task, [pipe], [])

    summary = pipeline_summary(store.load_scores("r1"), store.load_results("r1"))
    assert summary[pipe.pipeline_id]["accuracy"] == 1.0  # both fields match GT after normalization
    result = store.load_results("r1")[0]
    assert result.status is ResultStatus.OK
    assert result.cost.usd == pytest.approx(0.00014)  # token usage -> USD via price table
    # the normalized currency round-trips its amount
    assert Decimal(str(result.fields["total_amount"].value["amount"])) == Decimal("100.00")


def test_invalid_json_yields_parse_error_not_silent_fields(tmp_path):
    dataset = _write_dataset(tmp_path)
    task = _small_task()
    pipe = _stub(emit_garbage=True)
    result, raw = pipe.extract(dataset.documents[0], task)
    assert result.status is ResultStatus.ERROR
    assert result.error.error_class == "parse"  # un-parseable response, never silently-empty fields
    assert not result.fields
