"""Contract sanity checks. These are the load-bearing shapes, so guard them first."""
import pytest

from ezpz.core.result import ExtractionResult, FieldValue, ResultStatus
from ezpz.core.run import PipelineConfig
from ezpz.core.task import FieldSpec, ScorerRef, Task
from ezpz.core.values import ValueType


def test_pipeline_config_hash_is_deterministic():
    a = PipelineConfig(adapter="gemini", config={"model": "x", "temperature": 0})
    b = PipelineConfig(adapter="gemini", config={"temperature": 0, "model": "x"})
    assert a.config_hash == b.config_hash  # key order must not change the hash


def test_pipeline_config_hash_changes_with_config():
    a = PipelineConfig(adapter="gemini", config={"model": "x"})
    b = PipelineConfig(adapter="gemini", config={"model": "y"})
    assert a.config_hash != b.config_hash


def test_schema_hash_is_stable_across_calls():
    task = Task(
        name="t",
        version="1",
        fields=[FieldSpec(name="a", type=ValueType.STRING)],
        scorers=[ScorerRef(name="exact")],
    )
    assert task.schema_hash() == task.schema_hash()


def test_schema_hash_changes_with_field_type():
    t1 = Task(name="t", version="1", fields=[FieldSpec(name="a", type=ValueType.STRING)])
    t2 = Task(name="t", version="1", fields=[FieldSpec(name="a", type=ValueType.INTEGER)])
    assert t1.schema_hash() != t2.schema_hash()


def test_unparseable_is_a_distinct_marker():
    from ezpz.core.document import ABSENT
    from ezpz.core.values import UNPARSEABLE

    assert UNPARSEABLE is UNPARSEABLE        # a singleton
    assert UNPARSEABLE is not None           # distinct from "missing / not extracted"
    assert UNPARSEABLE != ABSENT             # distinct from "correctly-absent"
    assert UNPARSEABLE != ""                 # never equal to a real value
    assert repr(UNPARSEABLE) == "UNPARSEABLE"


def test_adapter_registry_registers_resolves_and_rejects_duplicates():
    from ezpz.adapters import registry

    @registry.register("contracts_demo_adapter")
    class _Demo:
        pass

    assert "contracts_demo_adapter" in registry.available()
    assert registry.get_adapter("contracts_demo_adapter") is _Demo
    with pytest.raises(ValueError):
        registry.register("contracts_demo_adapter")(_Demo)
    with pytest.raises(KeyError):
        registry.get_adapter("no_such_adapter")


def test_fake_pipeline_yields_valid_extraction_result(
    fake_pipeline, sample_document, sample_task
):
    result = fake_pipeline.run(sample_document, sample_task)

    assert result.status is ResultStatus.OK
    assert result.doc_id == sample_document.doc_id
    assert result.pipeline_id == fake_pipeline.config.config_hash
    assert set(result.fields) == {"invoice_number", "total_amount"}
    assert all(isinstance(fv, FieldValue) for fv in result.fields.values())
    assert result.fields["invoice_number"].value == "INV-1"
    assert result.timing is not None
    assert set(result.timing.stage_ms) == {"compile", "ingest", "invoke", "map"}
    # round-trips through the canonical output contract
    ExtractionResult.model_validate(result.model_dump())
