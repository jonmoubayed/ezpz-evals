"""Store contracts: schema bootstrap + content-addressed blob round-trips."""
import sqlite3

import pytest

from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore

EXPECTED_TABLES = {
    "datasets",
    "documents",
    "tasks",
    "pipelines",
    "runs",
    "results",
    "scores",
    "ground_truth",
    "extraction_cache",
}


def test_init_db_creates_database_and_expected_tables(tmp_path):
    db = tmp_path / "ezpz.sqlite"
    SqliteStore(str(db)).init_db()

    assert db.exists()
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        conn.close()
    assert EXPECTED_TABLES <= {r[0] for r in rows}


def test_init_db_is_idempotent(tmp_path):
    store = SqliteStore(str(tmp_path / "ezpz.sqlite"))
    store.init_db()
    store.init_db()  # CREATE TABLE IF NOT EXISTS -> safe to re-run


def test_blob_put_get_round_trip(tmp_path):
    store = BlobStore(str(tmp_path / "blobs"))
    data = b'{"raw": "provider response"}'
    digest = store.put(data)
    assert len(digest) == 64  # sha256 hex digest
    assert store.get(digest) == data


def test_blob_put_is_content_addressed_and_deduped(tmp_path):
    root = tmp_path / "blobs"
    store = BlobStore(str(root))
    d1 = store.put(b"same bytes")
    d2 = store.put(b"same bytes")
    assert d1 == d2
    files = [p for p in root.rglob("*") if p.is_file()]
    assert len(files) == 1  # identical content is stored exactly once


def test_blob_get_missing_raises(tmp_path):
    store = BlobStore(str(tmp_path / "blobs"))
    with pytest.raises(KeyError):
        store.get("0" * 64)


def test_run_result_score_and_gt_round_trip(tmp_path):
    from ezpz.core.result import Cost, ExtractionResult, FieldValue, ResultStatus, Timing
    from ezpz.core.run import PipelineConfig, Run, RunStatus
    from ezpz.core.score import FieldScore
    from ezpz.core.task import ScorerRef

    store = SqliteStore(str(tmp_path / "ezpz.sqlite"))
    store.init_db()

    run = Run(
        run_id="r1", dataset_ref="d@1", task_ref="t@1",
        pipelines=[PipelineConfig(adapter="fake", config={"x": 1})],
        scorers=[ScorerRef(name="exact")],
        status=RunStatus.RUNNING, started_at="t0",
    )
    store.save_run(run)
    loaded = store.load_run("r1")
    assert loaded.dataset_ref == "d@1"
    assert loaded.pipelines[0].adapter == "fake"
    assert loaded.scorers[0].name == "exact"

    result = ExtractionResult(
        doc_id="doc", pipeline_id="p", status=ResultStatus.OK,
        fields={"a": FieldValue(value="x")}, cost=Cost(usd=0.01), timing=Timing(latency_ms=5.0),
    )
    store.save_result("r1", result)
    assert store.load_results("r1")[0].fields["a"].value == "x"

    store.save_scores("r1", [FieldScore(
        doc_id="doc", pipeline_id="p", field="a", scorer="exact", value=1.0, passed=True,
    )])
    scores = store.load_scores("r1")
    assert scores[0].passed is True and scores[0].value == 1.0

    store.save_ground_truth("r1", "doc", {"a": "x"})
    assert store.load_ground_truth("r1", "doc") == {"a": "x"}

    store.finish_run("r1", RunStatus.COMPLETE, "t1")
    assert store.load_run("r1").status == RunStatus.COMPLETE


def test_result_error_message_round_trips(tmp_path):
    from ezpz.core.result import ErrorInfo, ExtractionResult, ResultStatus

    store = SqliteStore(str(tmp_path / "ezpz.sqlite"))
    store.init_db()
    store.save_result("r1", ExtractionResult(
        doc_id="d", pipeline_id="p", status=ResultStatus.ERROR,
        error=ErrorInfo(error_class="transport", message="401 authentication_error")))
    loaded = store.load_results("r1")[0]
    assert loaded.error.error_class == "transport"
    assert loaded.error.message == "401 authentication_error"


def test_init_db_migrates_old_results_table_missing_error_message(tmp_path):
    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(  # the pre-migration results schema (no error_message column)
        "CREATE TABLE results (run_id TEXT, doc_id TEXT, pipeline_id TEXT, status TEXT, "
        "fields_json TEXT, raw_response_ref TEXT, latency_ms REAL, cost_usd REAL, "
        "cost_units REAL, error_class TEXT, PRIMARY KEY (run_id, doc_id, pipeline_id));")
    conn.commit()
    conn.close()

    SqliteStore(str(db)).init_db()  # must ALTER the existing table, not just CREATE IF NOT EXISTS
    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(results)")}
    conn.close()
    assert "error_message" in cols
