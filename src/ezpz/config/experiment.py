"""ExperimentConfig ties everything together and is what `ezpz run` executes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ezpz.core.document import Dataset
from ezpz.core.run import PipelineConfig, RunOptions
from ezpz.core.task import ScorerRef, Task


def _split_ref(ref: str) -> tuple[str, str]:
    name, _, version = ref.partition("@")
    return name, version


def resolve_dataset(root: str, ref: str) -> Dataset:
    """Resolve "name@version" to ``<root>/datasets/<name>/`` (manifest.jsonl + ground_truth/)."""
    name, version = _split_ref(ref)
    return Dataset.load_from_manifest(str(Path(root) / "datasets" / name), name, version)


def resolve_task(root: str, ref: str) -> Task:
    """Resolve "name@version" to ``<root>/tasks/<name>.yaml``."""
    name, _version = _split_ref(ref)
    return Task.load_yaml(str(Path(root) / "tasks" / f"{name}.yaml"))


class ExperimentConfig(BaseModel):
    dataset: str                      # "name@version"
    task: str                         # "name@version"
    pipelines: list[PipelineConfig]   # the swappable tools under test
    scorers: list[ScorerRef] = Field(default_factory=list)  # defaults; task per-field overrides win
    options: RunOptions = Field(default_factory=RunOptions)
    plugins: list[str] = Field(default_factory=list)  # extra adapter/scorer modules to import

    @classmethod
    def load_yaml(cls, path: str) -> "ExperimentConfig":
        import yaml
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls.model_validate(data)
