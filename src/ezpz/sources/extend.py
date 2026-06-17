"""``extend`` — raw documents pulled from Extend's file storage.

Extend is used here purely as a *source* of files (distinct from the ``extend`` extraction
adapter): it lists files and downloads their bytes. Extend does not carry our task's ground truth,
so these documents are unlabeled ("not-labeled" state) — you attach ground truth separately if you
want to score them.

Auth is a Bearer token read from the env var named by ``config.api_key_env`` (default
``EXTEND_API_KEY``); ``config`` may also set ``base_url``, ``api_version``, and ``limit``.

Install: ``pip install -e ".[extend]"`` (reuses the httpx extra).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from ezpz.core.document import Dataset, DatasetSpec, Document
from ezpz.sources.base import DocumentSource
from ezpz.sources.registry import register

_DEFAULT_BASE = "https://api.extend.ai"
_DEFAULT_API_VERSION = "2025-04-21"


def _first(d: dict[str, Any], *keys: str) -> Any:
    """Return the first present, non-empty value among ``keys`` (Extend field names vary by route)."""
    for k in keys:
        if d.get(k):
            return d[k]
    return None


@register("extend")
class ExtendSource(DocumentSource):
    name = "extend"

    def _client(self):  # type: ignore[no-untyped-def]
        try:
            import httpx  # lazy: keep registry usable without the extra
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise RuntimeError("the 'extend' source needs httpx — install with: pip install ezpz-evals[extend]") from e
        env = self.config.get("api_key_env", "EXTEND_API_KEY")
        token = os.environ.get(env)
        if not token:
            raise RuntimeError(f"extend source: env var '{env}' is not set")
        return httpx.Client(
            base_url=self.config.get("base_url", _DEFAULT_BASE),
            headers={
                "Authorization": f"Bearer {token}",
                "x-extend-api-version": self.config.get("api_version", _DEFAULT_API_VERSION),
            },
            timeout=60.0,
        )

    def _list_files(self, client) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
        files: list[dict[str, Any]] = []
        params: dict[str, Any] = {}
        if self.config.get("limit"):
            params["limit"] = self.config["limit"]
        token: str | None = None
        while True:
            if token:
                params["nextPageToken"] = token
            resp = client.get("/files", params=params)
            resp.raise_for_status()
            body = resp.json()
            files.extend(body.get("files") or body.get("data") or [])
            token = _first(body, "nextPageToken", "next", "nextToken")
            if not token or (self.config.get("limit") and len(files) >= self.config["limit"]):
                break
        return files

    def _download(self, client, meta: dict[str, Any]) -> bytes:  # type: ignore[no-untyped-def]
        url = _first(meta, "presignedUrl", "url", "downloadUrl")
        if not url:  # detail route exposes the signed URL the list route omits
            detail = client.get(f"/files/{meta['id']}")
            detail.raise_for_status()
            body = detail.json()
            file_obj = body.get("file", body)
            url = _first(file_obj, "presignedUrl", "url", "downloadUrl")
        if not url:
            raise RuntimeError(f"extend file {meta.get('id')} has no downloadable URL")
        if url.startswith("/"):  # relative to the API base -> reuse the authed client
            return client.get(url).content
        import httpx  # presigned absolute URL: fetch without the API auth header
        return httpx.get(url, timeout=60.0, follow_redirects=True).content

    def load(self, spec: DatasetSpec, *, root: str, cache_dir: Path) -> Dataset:
        client = self._client()
        staged = cache_dir / spec.name
        documents: list[Document] = []
        try:
            for meta in self._list_files(client):
                name = _first(meta, "name", "fileName", "filename") or meta.get("id")
                data = self._download(client, meta)
                suffix = Path(str(name)).suffix
                doc_path = self._cache_path(staged / "docs", str(meta.get("id")), data, suffix)
                documents.append(
                    Document(
                        doc_id=hashlib.sha256(data).hexdigest(),
                        slug=str(name),
                        path=str(doc_path.relative_to(staged)),
                        mime=_first(meta, "mimeType", "type", "contentType"),
                        source_path=str(doc_path.resolve()),
                    )
                )
        finally:
            client.close()
        return Dataset(name=spec.name, version=spec.version, root=str(staged), documents=documents)
