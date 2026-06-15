"""The extraction cache is what makes re-scoring free. Key stability + raw round-trip."""
from ezpz.engine.cache import RawResponseCache, cache_key
from ezpz.store.blobs import BlobStore
from ezpz.store.sqlite import SqliteStore


def test_cache_key_is_stable_and_sensitive():
    k = cache_key("doc", "pipe", "schema")
    assert k == cache_key("doc", "pipe", "schema")
    assert k != cache_key("doc", "pipe2", "schema")   # config change -> miss
    assert k != cache_key("doc", "pipe", "schema2")   # schema change -> miss


def test_raw_response_cache_round_trip(tmp_path):
    store = SqliteStore(str(tmp_path / "db.sqlite"))
    store.init_db()
    cache = RawResponseCache(store, BlobStore(str(tmp_path / "blobs")))

    assert cache.get("k") is None  # miss
    cache.put("k", raw={"a": 1, "amount": "12.00"}, cost={"usd": 0.0}, doc_id="d", pipeline_id="p")
    hit = cache.get("k")
    assert hit["raw"] == {"a": 1, "amount": "12.00"}
    assert hit["cost"] == {"usd": 0.0}
    assert "ref" in hit
