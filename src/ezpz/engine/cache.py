"""Content-addressed extraction cache — the single highest-value piece for a local tool
hitting paid APIs.

Cache the RAWEST durable artifact (the provider response), then derive normalized fields on
read. That way fixing a mapping/normalization bug is free (re-derive over cached raw), and a
changed prompt yields a new config_hash -> natural cache miss only for what changed.

Key = sha256(doc_id + pipeline_config_hash + task_schema_hash).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore


def cache_key(doc_id: str, pipeline_config_hash: str, task_schema_hash: str) -> str:
    h = hashlib.sha256()
    h.update(doc_id.encode())
    h.update(b"|")
    h.update(pipeline_config_hash.encode())
    h.update(b"|")
    h.update(task_schema_hash.encode())
    return h.hexdigest()


class RawResponseCache:
    """Stores raw provider responses (+ reported cost) by cache_key.

    The raw response lives in the content-addressed blob store; the sqlite ``extraction_cache``
    table maps a cache_key to its blob. A hit returns ``{"raw", "cost", "ref"}``.
    """

    def __init__(self, store: SqliteStore, blobs: BlobStore):
        self.store = store
        self.blobs = blobs

    def get(self, key: str) -> Optional[dict]:
        conn = self.store.connect()
        try:
            row = conn.execute(
                "SELECT blob_hash FROM extraction_cache WHERE cache_key=?", (key,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        envelope = json.loads(self.blobs.get(row["blob_hash"]).decode())
        envelope["ref"] = row["blob_hash"]
        return envelope

    def put(
        self, key: str, raw: Any, cost: Optional[dict], doc_id: str = "", pipeline_id: str = ""
    ) -> str:
        # raw MUST be JSON-serializable: it is the durable artifact re-mapped on a cache hit, so
        # the cache-hit map() must see the same shape the cache-miss map() produced. A non-
        # serializable raw fails loudly here rather than being silently coerced (no default=str).
        envelope = {"raw": raw, "cost": cost}
        blob_hash = self.blobs.put(json.dumps(envelope, sort_keys=True).encode())
        conn = self.store.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO extraction_cache "
                "(cache_key, blob_hash, doc_id, pipeline_id, created_at) VALUES (?,?,?,?,?)",
                (key, blob_hash, doc_id, pipeline_id, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
        return blob_hash
