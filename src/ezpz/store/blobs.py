"""Content-addressed blob store under .ezpz/blobs/. put(bytes)->sha256; get(hash)->bytes.
Dedupes raw responses/artifacts by content; sqlite holds the hashes."""
from __future__ import annotations

import hashlib
from pathlib import Path


class BlobStore:
    def __init__(self, root: str = ".ezpz/blobs"):
        self.root = Path(root)

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        dest = self.root / digest[:2] / digest
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        return digest

    def get(self, digest: str) -> bytes:
        dest = self.root / digest[:2] / digest
        if not dest.exists():
            raise KeyError(f"blob '{digest}' not found under {self.root}")
        return dest.read_bytes()
