"""Task = what you are evaluating. Its schema is the input contract every adapter honors
and the thing scorers measure against."""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from ezpz.core.values import ValueType


class TaskType(str, Enum):
    EXTRACTION = "extraction"     # v1: doc -> schema-conforming fields
    # CLASSIFICATION = "classification"
    # QA = "qa"                   # doc(s) + question -> answer + retrieved contexts
    # SUMMARIZATION = "summarization"


class ScorerRef(BaseModel):
    """Reference a scorer by registered name + its config (e.g. tolerance, threshold)."""
    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class FieldSpec(BaseModel):
    name: str
    type: ValueType
    required: bool = True
    description: Optional[str] = None        # doubles as labeling guidance + model instruction
    enum_values: Optional[list[str]] = None  # for ValueType.ENUM
    item: Optional["FieldSpec"] = None       # for ValueType.LIST (the element spec)
    fields: Optional[list["FieldSpec"]] = None  # for ValueType.OBJECT / list-of-object
    match_key: Optional[str] = None          # for LIST of OBJECT: how to align predicted vs GT rows
    scorers: Optional[list[ScorerRef]] = None  # per-field override of the task default scorers


class Task(BaseModel):
    name: str
    version: str
    type: TaskType = TaskType.EXTRACTION
    instructions: Optional[str] = None       # optional NL hint adapters may use
    fields: list[FieldSpec] = Field(default_factory=list)
    scorers: list[ScorerRef] = Field(default_factory=list)  # defaults applied per field

    def schema_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    @classmethod
    def load_yaml(cls, path: str) -> "Task":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


FieldSpec.model_rebuild()
