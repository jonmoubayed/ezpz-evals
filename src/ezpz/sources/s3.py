"""``s3`` — documents stored in an S3 (or S3-compatible) bucket.

The bucket holds the same layout a ``local`` dataset would: a ``manifest.jsonl`` plus the
``path`` / ``ground_truth_path`` objects it references. This source downloads those objects into a
content-addressed cache and then delegates to :class:`LocalSource`, so manifest parsing, document
identity, and the three ground-truth states behave identically to a local run.

Credentials use boto3's standard chain (env vars / shared config / instance role); ``config`` may
pin ``region``, ``endpoint_url`` (for MinIO and other S3-compatible stores), and ``profile``.

Install: ``pip install -e ".[s3]"``.
"""
from __future__ import annotations

import json
from pathlib import Path

from ezpz.core.document import Dataset, DatasetSpec
from ezpz.sources.base import DocumentSource
from ezpz.sources.local import LocalSource
from ezpz.sources.registry import register


@register("s3")
class S3Source(DocumentSource):
    name = "s3"

    def _client(self):  # type: ignore[no-untyped-def]
        try:
            import boto3  # lazy: registry/validate must work without the [s3] extra
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise RuntimeError("the 's3' source needs boto3 — install with: pip install ezpz-evals[s3]") from e
        kwargs = {}
        if self.config.get("region"):
            kwargs["region_name"] = self.config["region"]
        if self.config.get("endpoint_url"):
            kwargs["endpoint_url"] = self.config["endpoint_url"]
        session = boto3.Session(profile_name=self.config.get("profile"))
        return session.client("s3", **kwargs)

    def load(self, spec: DatasetSpec, *, root: str, cache_dir: Path) -> Dataset:
        bucket = self.config.get("bucket")
        if not bucket:
            raise ValueError("s3 source requires config.bucket")
        prefix = self.config.get("prefix", "").lstrip("/")
        manifest_key = self.config.get("manifest_key", "manifest.jsonl")
        client = self._client()

        def _key(rel: str) -> str:
            return f"{prefix.rstrip('/')}/{rel}" if prefix else rel

        def _get(rel: str) -> bytes:
            obj = client.get_object(Bucket=bucket, Key=_key(rel))
            return obj["Body"].read()

        # Stage the dataset locally, mirroring the manifest's relative layout, then reuse `local`.
        staged = cache_dir / spec.name
        (staged).mkdir(parents=True, exist_ok=True)
        manifest_bytes = _get(manifest_key)
        (staged / "manifest.jsonl").write_bytes(manifest_bytes)

        for line in manifest_bytes.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for rel in (rec.get("path"), rec.get("ground_truth_path")):
                if not rel:
                    continue
                dest = staged / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(_get(rel))

        return LocalSource({"dir": str(staged)}).load(spec, root=root, cache_dir=cache_dir)
