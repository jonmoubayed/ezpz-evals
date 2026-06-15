"""Run = the immutable record of one experiment. Pins dataset/task/pipeline/scorer versions
plus the environment (SDK versions) so you can tell whether a number moved for real."""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from ezpz.core.task import ScorerRef


class PipelineConfig(BaseModel):
    """One pipeline under test. `config` must FULLY determine behavior so the hash is a sound
    cache key (model version, prompt template, schema mode, parser, temperature, ...)."""
    adapter: str                                   # registered adapter name
    config: dict[str, Any] = Field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        blob = json.dumps(
            {"adapter": self.adapter, "config": self.config}, sort_keys=True
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


class RunOptions(BaseModel):
    concurrency: int = 4              # NOTE: engine bounds concurrency PER PROVIDER
    sample: Optional[int] = None      # run on N docs for fast iteration
    slices: Optional[list[str]] = None
    budget_usd: Optional[float] = None
    force_refresh: bool = False       # bypass the extraction cache
    samples_per_doc: int = 1          # >1 to measure nondeterminism variance


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class Run(BaseModel):
    run_id: str
    dataset_ref: str                  # "name@version"
    task_ref: str                     # "name@version"
    pipelines: list[PipelineConfig]
    scorers: list[ScorerRef]
    options: RunOptions = Field(default_factory=RunOptions)
    env: dict[str, str] = Field(default_factory=dict)   # tool/SDK versions
    status: RunStatus = RunStatus.PENDING
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
