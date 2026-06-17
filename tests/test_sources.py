"""Document sources: registry, the default `local` source, config coercion, and the
remote-source contract (enumerate + materialize bytes -> content-hashed Documents)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ezpz.config.experiment import ExperimentConfig, resolve_dataset
from ezpz.core.document import ABSENT, Dataset, DatasetSpec, Document, GroundTruth
from ezpz.sources import registry as source_registry
from ezpz.sources.base import DocumentSource


# -- a tiny on-disk dataset (mirrors the `local` layout) --------------------------------------

def _write_local_dataset(root: Path, name: str = "mini") -> None:
    ds = root / "datasets" / name
    (ds / "docs").mkdir(parents=True)
    (ds / "ground_truth").mkdir(parents=True)
    (ds / "docs" / "a.txt").write_text("INVOICE\nTotal: $10.00\n")
    (ds / "ground_truth" / "a.json").write_text(
        json.dumps({"fields": {"total_amount": "10.00", "po_number": ABSENT}})
    )
    (ds / "manifest.jsonl").write_text(
        json.dumps(
            {
                "slug": "a",
                "path": "docs/a.txt",
                "mime": "text/plain",
                "ground_truth_path": "ground_truth/a.json",
            }
        )
        + "\n"
    )


# -- registry ---------------------------------------------------------------------------------

def test_builtin_sources_are_registered():
    import ezpz.sources  # noqa: F401  (import = register)

    have = source_registry.available()
    assert {"local", "s3", "langfuse", "extend"} <= set(have)


def test_unknown_source_raises():
    with pytest.raises(KeyError):
        source_registry.get_source("does-not-exist")


# -- DatasetSpec coercion ---------------------------------------------------------------------

def test_spec_coerces_bare_string_to_local():
    spec = DatasetSpec.coerce("invoices_v1@2")
    assert (spec.source, spec.name, spec.version, spec.ref) == ("local", "invoices_v1", "2", "invoices_v1@2")


def test_spec_unversioned_ref():
    assert DatasetSpec.coerce("cohort").ref == "cohort"


def test_experiment_accepts_mapping_dataset(tmp_path: Path):
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        "dataset: {source: s3, name: invoices, version: '1', config: {bucket: b, prefix: p/}}\n"
        "task: invoice_extraction@1\n"
        "pipelines: [{adapter: fake, config: {}}]\n"
    )
    exp = ExperimentConfig.load_yaml(str(exp_yaml))
    assert exp.dataset.source == "s3"
    assert exp.dataset.config["bucket"] == "b"
    assert exp.dataset.ref == "invoices@1"


# -- local source (the default) ---------------------------------------------------------------

def test_local_source_loads_manifest_and_ground_truth(tmp_path: Path):
    _write_local_dataset(tmp_path)
    ds = resolve_dataset(str(tmp_path), "mini@1")
    assert isinstance(ds, Dataset)
    assert [d.slug for d in ds.documents] == ["a"]
    doc = ds.documents[0]
    assert Path(doc.source_path).read_text().startswith("INVOICE")

    gt = GroundTruth.load(ds.root, doc)
    assert gt is not None
    assert gt.fields["po_number"] == ABSENT  # the three GT states survive sourcing


def test_local_source_via_explicit_spec(tmp_path: Path):
    _write_local_dataset(tmp_path)
    ds = resolve_dataset(str(tmp_path), DatasetSpec(source="local", name="mini", version="1"))
    assert len(ds.documents) == 1


# -- remote-source contract (no SDK needed: a fake in-memory remote) --------------------------

@source_registry.register("_memtest")
class _InMemorySource(DocumentSource):
    """Stand-in remote: holds bytes in `config['blobs']`, materializes them, content-hashes."""

    name = "_memtest"

    def load(self, spec: DatasetSpec, *, root: str, cache_dir: Path) -> Dataset:
        docs = []
        for slug, data in self.config["blobs"].items():
            path = self._cache_path(cache_dir / spec.name, slug, data, ".txt")
            import hashlib

            docs.append(
                Document(
                    doc_id=hashlib.sha256(data).hexdigest(),
                    slug=slug,
                    path=str(path.relative_to(cache_dir / spec.name)),
                    mime="text/plain",
                    source_path=str(path.resolve()),
                )
            )
        return Dataset(name=spec.name, version=spec.version, root=str(cache_dir / spec.name), documents=docs)


def test_remote_source_materializes_bytes_locally(tmp_path: Path):
    spec = DatasetSpec(source="_memtest", name="m", config={"blobs": {"x": b"hello world"}})
    ds = resolve_dataset(str(tmp_path), spec)
    doc = ds.documents[0]
    # content-addressed identity + locally readable bytes, exactly like a local doc
    assert Path(doc.source_path).read_bytes() == b"hello world"
    import hashlib

    assert doc.doc_id == hashlib.sha256(b"hello world").hexdigest()


def test_cache_path_is_idempotent(tmp_path: Path):
    src = _InMemorySource({"blobs": {}})
    p1 = src._cache_path(tmp_path, "k", b"same", ".txt")
    p2 = src._cache_path(tmp_path, "k", b"same", ".txt")
    assert p1 == p2 and p1.read_bytes() == b"same"


# -- langfuse GT coercion (pure; no SDK) ------------------------------------------------------

def test_langfuse_expected_output_coercion():
    from ezpz.sources.langfuse import _expected_to_fields, _input_bytes

    # bare {field: value} map, with null -> ABSENT (the "correctly-absent" state)
    assert _expected_to_fields({"total_amount": "10", "po_number": None}) == {
        "total_amount": "10",
        "po_number": ABSENT,
    }
    # canonical {"fields": {...}} envelope is unwrapped
    assert _expected_to_fields({"fields": {"x": 1}}) == {"x": 1}
    # JSON string is parsed
    assert _expected_to_fields('{"x": 2}') == {"x": 2}
    # unlabeled / unaddressable -> None (the "not-labeled" state), never a fabricated GT
    assert _expected_to_fields(None) is None
    assert _expected_to_fields([1, 2, 3]) is None

    # text input stays text; structured input is rendered to json
    assert _input_bytes("hi")[1] == ".txt"
    assert _input_bytes({"a": 1})[1] == ".json"
