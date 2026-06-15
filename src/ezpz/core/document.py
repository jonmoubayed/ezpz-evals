"""Documents, datasets, and ground truth.

Document identity is content-addressed (hash of file bytes) so the same file is the same
document across runs — which is also what makes the extraction cache sound.

Ground-truth fields have THREE states; conflating them corrupts metrics:
  - present-with-value   -> the actual value
  - correctly-absent     -> the ABSENT sentinel (field genuinely not in the doc)
  - not-yet-labeled      -> key omitted entirely
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

ABSENT = "__ABSENT__"  # sentinel: field is genuinely not present in this document


def _content_hash(path: Path, fallback: str) -> str:
    """Document identity = sha256 of file bytes; fall back to a stable id for placeholders."""
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(fallback.encode()).hexdigest()


class Document(BaseModel):
    doc_id: str                      # content hash (sha256 of bytes); stable identity
    slug: Optional[str] = None       # human-friendly id
    path: str                        # relative path under the dataset root
    mime: Optional[str] = None
    doc_type: Optional[str] = None   # invoice | receipt | contract | ...
    tags: list[str] = Field(default_factory=list)  # drive per-slice metrics
    ground_truth_path: Optional[str] = None        # one GT file per doc
    source_path: Optional[str] = None  # absolute path to the file; set by the loader (not identity)
    notes: Optional[str] = None


class Dataset(BaseModel):
    name: str
    version: str                     # immutable once published; changes -> new version
    root: str                        # directory containing docs/ + ground_truth/ + manifest
    documents: list[Document] = Field(default_factory=list)
    manifest_hash: Optional[str] = None  # hash of the manifest for reproducibility

    @classmethod
    def load_from_manifest(cls, root: str, name: str, version: str) -> "Dataset":
        """Parse manifest.jsonl under ``root`` into Documents (content-hashed by bytes)."""
        root_path = Path(root)
        manifest = root_path / "manifest.jsonl"
        manifest_h = hashlib.sha256(manifest.read_bytes()).hexdigest()
        documents: list[Document] = []
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            source = root_path / rec["path"]
            documents.append(
                Document(
                    doc_id=_content_hash(source, rec.get("slug") or rec["path"]),
                    slug=rec.get("slug"),
                    path=rec["path"],
                    mime=rec.get("mime"),
                    doc_type=rec.get("doc_type"),
                    tags=rec.get("tags", []),
                    ground_truth_path=rec.get("ground_truth_path"),
                    source_path=str(source.resolve()),
                )
            )
        return cls(
            name=name, version=version, root=str(root_path),
            documents=documents, manifest_hash=manifest_h,
        )

    def sample(self, n: int) -> "Dataset":
        return self.model_copy(update={"documents": self.documents[:n]})
    # TODO: slice(tag) -> Dataset  (first-class cohorts, M4)


class GroundTruth(BaseModel):
    """The expected answer for one document, conforming to the task schema.
    Values use canonical representations; ABSENT marks correctly-absent fields."""
    doc_id: str
    fields: dict[str, Any] = Field(default_factory=dict)  # field_name -> canonical value | ABSENT

    @classmethod
    def load(cls, dataset_root: str, document: "Document") -> Optional["GroundTruth"]:
        """Load the GT file referenced by the document, or None if unlabeled/missing."""
        if not document.ground_truth_path:
            return None
        path = Path(dataset_root) / document.ground_truth_path
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(doc_id=document.doc_id, fields=data.get("fields", {}))
