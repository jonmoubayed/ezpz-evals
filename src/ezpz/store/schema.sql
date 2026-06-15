-- SQLite schema. Large artifacts (raw responses, parsed text, page images, uploads) live in
-- the blob store; tables hold references (blob hashes), not bytes.
--
-- NOTE: datasets/documents/pipelines are forward-looking scaffold (populated in M3/M7 for
-- dedup + the viewer); M1 writes runs/results/scores/ground_truth/extraction_cache.

CREATE TABLE IF NOT EXISTS datasets (
    name TEXT NOT NULL, version TEXT NOT NULL, root TEXT, manifest_hash TEXT,
    PRIMARY KEY (name, version)
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, dataset_name TEXT, dataset_version TEXT,
    slug TEXT, path TEXT, mime TEXT, doc_type TEXT, tags TEXT, ground_truth_path TEXT,
    source_path TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    name TEXT NOT NULL, version TEXT NOT NULL, type TEXT, schema_hash TEXT, spec_json TEXT,
    PRIMARY KEY (name, version)
);

CREATE TABLE IF NOT EXISTS pipelines (
    config_hash TEXT PRIMARY KEY, adapter TEXT, config_json TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, dataset_ref TEXT, task_ref TEXT,
    pipelines_json TEXT, scorers_json TEXT, options_json TEXT,
    env_json TEXT, status TEXT, started_at TEXT, finished_at TEXT
);

CREATE TABLE IF NOT EXISTS results (
    run_id TEXT, doc_id TEXT, pipeline_id TEXT, status TEXT,
    fields_json TEXT, raw_response_ref TEXT,
    latency_ms REAL, cost_usd REAL, cost_units REAL, error_class TEXT, error_message TEXT,
    extras_json TEXT,
    PRIMARY KEY (run_id, doc_id, pipeline_id)
);

CREATE TABLE IF NOT EXISTS scores (
    run_id TEXT, doc_id TEXT, pipeline_id TEXT, field TEXT, scorer TEXT,
    value REAL, passed INTEGER, detail_json TEXT,
    PRIMARY KEY (run_id, doc_id, pipeline_id, field, scorer)
);

-- Ground truth pinned per run (a run is immutable + self-contained, so re-scoring needs no
-- external files). Same GT across pipelines for a doc -> stored once per (run, doc).
CREATE TABLE IF NOT EXISTS ground_truth (
    run_id TEXT, doc_id TEXT, fields_json TEXT,
    PRIMARY KEY (run_id, doc_id)
);

-- Extraction cache: cache_key = sha256(doc_id|config_hash|schema_hash) -> raw-response blob.
-- The rawest durable artifact is cached so normalization/scoring re-derive for zero API calls.
CREATE TABLE IF NOT EXISTS extraction_cache (
    cache_key TEXT PRIMARY KEY, blob_hash TEXT, doc_id TEXT, pipeline_id TEXT, created_at TEXT
);
