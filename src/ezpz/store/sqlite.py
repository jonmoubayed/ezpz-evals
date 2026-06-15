"""SQLite-backed store. Local, zero-server, queryable, transactional. Schema in schema.sql.

Runs/results/scores/ground-truth round-trip through here. Large artifacts never live in the
DB — only blob hashes do (see ezpz.store.blobs).
"""
from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any

from ezpz.core.document import Document
from ezpz.core.result import Cost, ErrorInfo, ExtractionResult, FieldValue, ResultStatus, Timing
from ezpz.core.run import PipelineConfig, Run, RunOptions, RunStatus
from ezpz.core.score import FieldScore
from ezpz.core.task import ScorerRef, Task
from ezpz.core.values import UNPARSEABLE, CurrencyValue

UNPARSEABLE_JSON = "__UNPARSEABLE__"  # storage marker for the UNPARSEABLE singleton (display only)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class SqliteStore:
    def __init__(self, db_path: str = ".ezpz/ezpz.sqlite"):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent readers + one writer
        conn.execute("PRAGMA busy_timeout=5000")  # wait instead of erroring on a brief lock
        return conn

    def init_db(self) -> None:
        """Create tables from schema.sql if absent + apply additive migrations (idempotent)."""
        schema = _SCHEMA_PATH.read_text()
        conn = self.connect()
        try:
            conn.executescript(schema)
            _migrate(conn)
            conn.commit()
        finally:
            conn.close()

    # ---- runs ----
    def save_run(self, run: Run) -> None:
        conn = self.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, dataset_ref, task_ref, pipelines_json, "
                "scorers_json, options_json, env_json, status, started_at, finished_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    run.run_id, run.dataset_ref, run.task_ref,
                    json.dumps([p.model_dump() for p in run.pipelines]),
                    json.dumps([s.model_dump() for s in run.scorers]),
                    json.dumps(run.options.model_dump()),
                    json.dumps(run.env),
                    run.status.value, run.started_at, run.finished_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def finish_run(self, run_id: str, status: RunStatus, finished_at: str) -> None:
        conn = self.connect()
        try:
            conn.execute(
                "UPDATE runs SET status=?, finished_at=? WHERE run_id=?",
                (status.value, finished_at, run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def load_run(self, run_id: str) -> Run:
        conn = self.connect()
        try:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"run '{run_id}' not found")
        return _row_to_run(row)

    def list_runs(self) -> list[Run]:
        conn = self.connect()
        try:
            rows = conn.execute("SELECT * FROM runs ORDER BY started_at").fetchall()
        finally:
            conn.close()
        return [_row_to_run(r) for r in rows]

    # ---- documents (persisted so the viewer can read SQLite only) ----
    def save_document(self, document: "Document", dataset_ref: str = "") -> None:
        name, _, version = dataset_ref.partition("@")
        conn = self.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO documents (doc_id, dataset_name, dataset_version, slug, "
                "path, mime, doc_type, tags, ground_truth_path, source_path) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    document.doc_id, name or None, version or None, document.slug, document.path,
                    document.mime, document.doc_type, json.dumps(document.tags),
                    document.ground_truth_path, document.source_path,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_documents(self, doc_ids: list[str]) -> "dict[str, Document]":
        if not doc_ids:
            return {}
        placeholders = ",".join("?" * len(doc_ids))
        conn = self.connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM documents WHERE doc_id IN ({placeholders})", doc_ids
            ).fetchall()
        finally:
            conn.close()
        return {r["doc_id"]: _row_to_document(r) for r in rows}

    # ---- tasks ----
    def save_task(self, task: Task) -> None:
        conn = self.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO tasks (name, version, type, schema_hash, spec_json) "
                "VALUES (?,?,?,?,?)",
                (task.name, task.version, task.type.value, task.schema_hash(),
                 json.dumps(task.model_dump(mode="json"))),
            )
            conn.commit()
        finally:
            conn.close()

    def load_task(self, task_ref: str) -> Task:
        name, _, version = task_ref.partition("@")
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT spec_json FROM tasks WHERE name=? AND version=?", (name, version)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"task '{task_ref}' not found")
        return Task.model_validate(json.loads(row["spec_json"]))

    # ---- results ----
    def save_result(self, run_id: str, result: ExtractionResult) -> None:
        cost = result.cost
        conn = self.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO results (run_id, doc_id, pipeline_id, status, fields_json, "
                "raw_response_ref, latency_ms, cost_usd, cost_units, error_class, error_message, "
                "extras_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, result.doc_id, result.pipeline_id, result.status.value,
                    _fields_to_json(result.fields),
                    result.raw_response_ref,
                    result.timing.latency_ms if result.timing else None,
                    cost.usd if cost else None,
                    cost.units if cost else None,
                    result.error.error_class if result.error else None,
                    result.error.message if result.error else None,
                    json.dumps(result.extras) if result.extras else None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_results(self, run_id: str) -> list[ExtractionResult]:
        conn = self.connect()
        try:
            rows = conn.execute("SELECT * FROM results WHERE run_id=?", (run_id,)).fetchall()
        finally:
            conn.close()
        return [_row_to_result(r) for r in rows]

    # ---- scores ----
    def save_scores(self, run_id: str, scores: list[FieldScore]) -> None:
        conn = self.connect()
        try:
            conn.executemany(_SCORE_INSERT, _score_rows(run_id, scores))
            conn.commit()
        finally:
            conn.close()

    def replace_scores(self, run_id: str, scores: list[FieldScore]) -> None:
        # Delete + re-insert in ONE transaction so a crash can't leave a run with zero scores.
        conn = self.connect()
        try:
            conn.execute("DELETE FROM scores WHERE run_id=?", (run_id,))
            conn.executemany(_SCORE_INSERT, _score_rows(run_id, scores))
            conn.commit()
        finally:
            conn.close()

    def load_scores(self, run_id: str) -> list[FieldScore]:
        conn = self.connect()
        try:
            rows = conn.execute("SELECT * FROM scores WHERE run_id=?", (run_id,)).fetchall()
        finally:
            conn.close()
        return [
            FieldScore(
                doc_id=r["doc_id"], pipeline_id=r["pipeline_id"], field=r["field"],
                scorer=r["scorer"], value=r["value"],
                passed=None if r["passed"] is None else bool(r["passed"]),
                detail=json.loads(r["detail_json"]) if r["detail_json"] else {},
            )
            for r in rows
        ]

    # ---- ground truth (pinned per run) ----
    def save_ground_truth(self, run_id: str, doc_id: str, fields: dict) -> None:
        conn = self.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO ground_truth (run_id, doc_id, fields_json) VALUES (?,?,?)",
                (run_id, doc_id, json.dumps(fields)),
            )
            conn.commit()
        finally:
            conn.close()

    def load_ground_truth(self, run_id: str, doc_id: str) -> "dict | None":
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT fields_json FROM ground_truth WHERE run_id=? AND doc_id=?",
                (run_id, doc_id),
            ).fetchone()
        finally:
            conn.close()
        return json.loads(row["fields_json"]) if row else None

    def export(self, run_id: str) -> dict:
        """Serialize a whole run (run + results + scores) to a plain JSON-able dict."""
        return {
            "run": self.load_run(run_id).model_dump(mode="json"),
            "results": [r.model_dump(mode="json") for r in self.load_results(run_id)],
            "scores": [s.model_dump(mode="json") for s in self.load_scores(run_id)],
        }


def _jsonable(value: Any) -> Any:
    """JSON-safe view of a canonical value for storage (display/debug only; never re-scored).

    Handles the marker singleton and non-JSON-native canonical types that model_dump(mode="json")
    cannot serialize through a free-form Any field.
    """
    if value is UNPARSEABLE:
        return UNPARSEABLE_JSON
    if isinstance(value, CurrencyValue):
        return {"amount": str(value.amount), "currency": value.currency}
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _fields_to_json(fields: dict[str, FieldValue]) -> str:
    return json.dumps({
        name: {
            "value": _jsonable(fv.value),
            "confidence": fv.confidence,
            "provenance": fv.provenance.model_dump() if fv.provenance else None,
        }
        for name, fv in fields.items()
    })


_SCORE_INSERT = (
    "INSERT OR REPLACE INTO scores (run_id, doc_id, pipeline_id, field, scorer, "
    "value, passed, detail_json) VALUES (?,?,?,?,?,?,?,?)"
)


def _score_rows(run_id: str, scores: list[FieldScore]) -> list[tuple]:
    return [
        (
            run_id, s.doc_id, s.pipeline_id, s.field, s.scorer, s.value,
            None if s.passed is None else int(s.passed),
            json.dumps(s.detail),
        )
        for s in scores
    ]


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive column migrations so DBs created by an older schema still open and work."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    for col in ("error_message", "extras_json"):
        if col not in cols:
            conn.execute(f"ALTER TABLE results ADD COLUMN {col} TEXT")


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        doc_id=row["doc_id"], slug=row["slug"], path=row["path"] or "", mime=row["mime"],
        doc_type=row["doc_type"], tags=json.loads(row["tags"] or "[]"),
        ground_truth_path=row["ground_truth_path"], source_path=row["source_path"],
    )


def _row_to_run(row: sqlite3.Row) -> Run:
    return Run(
        run_id=row["run_id"], dataset_ref=row["dataset_ref"], task_ref=row["task_ref"],
        pipelines=[PipelineConfig.model_validate(p) for p in json.loads(row["pipelines_json"] or "[]")],
        scorers=[ScorerRef.model_validate(s) for s in json.loads(row["scorers_json"] or "[]")],
        options=RunOptions.model_validate(json.loads(row["options_json"] or "{}")),
        env=json.loads(row["env_json"] or "{}"),
        status=RunStatus(row["status"]),
        started_at=row["started_at"], finished_at=row["finished_at"],
    )


def _row_to_result(row: sqlite3.Row) -> ExtractionResult:
    fields = {
        name: FieldValue.model_validate(v)
        for name, v in json.loads(row["fields_json"] or "{}").items()
    }
    error = ErrorInfo(
        error_class=row["error_class"],
        message=row["error_message"] if "error_message" in row.keys() else None,
    ) if row["error_class"] else None
    keys = row.keys()
    extras = json.loads(row["extras_json"]) if ("extras_json" in keys and row["extras_json"]) else {}
    return ExtractionResult(
        doc_id=row["doc_id"], pipeline_id=row["pipeline_id"],
        status=ResultStatus(row["status"]), fields=fields,
        raw_response_ref=row["raw_response_ref"],
        timing=Timing(latency_ms=row["latency_ms"]) if row["latency_ms"] is not None else None,
        cost=Cost(usd=row["cost_usd"], units=row["cost_units"]),
        error=error, extras=extras,
    )
