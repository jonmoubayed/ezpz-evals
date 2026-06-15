"""Shared fixtures: a tiny in-memory dataset + a deterministic FakePipeline, so the
engine/scorers/store can be exercised without hitting any provider."""
from __future__ import annotations

import pytest

from ezpz.core.document import Document
from ezpz.core.run import PipelineConfig
from ezpz.core.task import FieldSpec, Task
from ezpz.core.values import ValueType
from ezpz.adapters.fake import FakePipeline


@pytest.fixture
def sample_task() -> Task:
    return Task(
        name="demo",
        version="1",
        fields=[
            FieldSpec(name="invoice_number", type=ValueType.STRING),
            FieldSpec(name="total_amount", type=ValueType.CURRENCY),
        ],
    )


@pytest.fixture
def sample_document() -> Document:
    return Document(doc_id="doc-abc", path="docs/demo.pdf")


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    config = PipelineConfig(
        adapter="fake",
        config={"fake_fields": {"invoice_number": "INV-1", "total_amount": "1200"}},
    )
    return FakePipeline(config)
