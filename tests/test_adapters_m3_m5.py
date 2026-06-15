"""Extend (M3) + LlamaIndex (M5) adapters via mock seams — no SDKs, no network.

Extend stresses the abstraction: it's the first adapter with native confidence + bbox provenance
and per-page (not per-token) cost. LlamaIndex is parse+extract with a separate parse cost.
"""
import json

import pytest

from ezpz.adapters.extend import ExtendPipeline
from ezpz.adapters.llamaindex import LlamaIndexPipeline
from ezpz.adapters.registry import get_adapter
from ezpz.core.document import Dataset
from ezpz.core.result import ResultStatus
from ezpz.core.run import PipelineConfig, Run, RunStatus
from ezpz.core.task import FieldSpec, ScorerRef, Task
from ezpz.core.values import ValueType
from ezpz.engine.cache import RawResponseCache
from ezpz.engine.executor import Executor
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore


def _dataset(tmp_path) -> Dataset:
    ds = tmp_path / "ds"
    (ds / "docs").mkdir(parents=True)
    (ds / "ground_truth").mkdir(parents=True)
    (ds / "docs" / "d1.txt").write_text("ACME INVOICE  Invoice #: INV-1  Total: $100.00")
    (ds / "ground_truth" / "d1.json").write_text(json.dumps({"fields": {
        "invoice_number": "INV-1", "total_amount": {"amount": "100.00", "currency": "USD"},
    }}))
    (ds / "manifest.jsonl").write_text(json.dumps({
        "slug": "d1", "path": "docs/d1.txt", "mime": "text/plain",
        "ground_truth_path": "ground_truth/d1.json"}) + "\n")
    return Dataset.load_from_manifest(str(ds), "ds", "1")


def _task() -> Task:
    return Task(name="t", version="1", fields=[
        FieldSpec(name="invoice_number", type=ValueType.STRING, scorers=[ScorerRef(name="exact")]),
        FieldSpec(name="total_amount", type=ValueType.CURRENCY,
                  scorers=[ScorerRef(name="numeric_tolerance", config={"abs": 0.01})]),
    ])


def _run(tmp_path, pipe) -> SqliteStore:
    store = SqliteStore(str(tmp_path / "db.sqlite"))
    store.init_db()
    cache = RawResponseCache(store, BlobStore(str(tmp_path / "blobs")))
    run = Run(run_id="r1", dataset_ref="ds@1", task_ref="t@1", pipelines=[pipe.config],
              scorers=[], status=RunStatus.RUNNING)
    Executor(store, cache).run(run, _dataset(tmp_path), _task(), [pipe], [])
    return store


def test_extend_and_llamaindex_registered():
    assert get_adapter("extend") is not None
    assert get_adapter("llamaindex") is not None


# ---- Extend ----

class _StubExtend(ExtendPipeline):
    def _extract(self, prepared, ingested):
        return {
            "status": "PROCESSED", "metadata": {"pageCount": 2},
            "output": {
                "invoice_number": {"value": "INV-1", "confidence": 0.97,
                                   "references": [{"page": 1, "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.3, "y1": 0.13}}]},
                "total_amount": {"value": {"amount": "100.00", "currency": "USD"}, "confidence": 0.9},
            },
        }


def test_extend_maps_confidence_provenance_and_per_page_cost():
    pipe = _StubExtend(PipelineConfig(adapter="extend", config={"price_per_page": 0.05}))
    assert pipe.capabilities.confidence and pipe.capabilities.provenance
    raw, cost = pipe.invoke(None, None)
    assert cost.units == 2.0 and cost.usd == pytest.approx(0.10)  # 2 pages x $0.05
    inv = pipe.map(raw, _task())["invoice_number"]
    assert inv.confidence == 0.97
    assert inv.provenance.page == 1 and inv.provenance.bbox == [0.1, 0.1, 0.3, 0.13]


def test_extend_confidence_persists_through_run(tmp_path):
    store = _run(tmp_path, _StubExtend(PipelineConfig(adapter="extend", config={"price_per_page": 0.05})))
    result = store.load_results("r1")[0]
    assert result.status is ResultStatus.OK
    assert result.fields["invoice_number"].confidence == 0.97  # survived map+normalize+store
    assert result.cost.units == 2.0


def test_extend_type_map_covers_every_value_type():
    # compile() fails loudly on any type it can't express; assert today's map covers them all
    from ezpz.adapters.extend import _EXTEND_TYPES
    assert all(vt in _EXTEND_TYPES for vt in ValueType)


# ---- LlamaIndex ----

class _StubLlama(LlamaIndexPipeline):
    def _generate(self, prepared, ingested):
        assert ingested["parser"] == "simple" and "INVOICE" in ingested["text"]  # real parse ran
        return (
            json.dumps({"invoice_number": "INV-1", "total_amount": {"amount": 100.0, "currency": "USD"}}),
            {"input_tokens": 50, "output_tokens": 10},
            {"parser": ingested["parser"]},
        )


def test_llamaindex_parses_extracts_and_folds_in_parse_cost(tmp_path):
    pipe = _StubLlama(PipelineConfig(adapter="llamaindex", config={
        "parser": "simple", "backend_model": "gpt-4o-mini",
        "prices": {"input_per_1m": 0.15, "output_per_1m": 0.60}, "parse_cost_usd": 0.003,
    }))
    assert pipe.model == "gpt-4o-mini"
    result = _run(tmp_path, pipe).load_results("r1")[0]
    assert result.fields["invoice_number"].value == "INV-1"
    expected = 50 / 1e6 * 0.15 + 10 / 1e6 * 0.60 + 0.003  # backend tokens + parse cost
    assert result.cost.usd == pytest.approx(expected, abs=1e-6)
