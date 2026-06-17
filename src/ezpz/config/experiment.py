"""ExperimentConfig ties everything together and is what `ezpz run` executes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ezpz.core.document import Dataset, DatasetSpec
from ezpz.core.run import PipelineConfig, RunOptions
from ezpz.core.task import ScorerRef, Task


def _split_ref(ref: str) -> tuple[str, str]:
    name, _, version = ref.partition("@")
    return name, version


def resolve_dataset(root: str, spec: "DatasetSpec | str") -> Dataset:
    """Load a dataset via its document source.

    ``spec`` is a :class:`DatasetSpec` (or a bare ``"name@version"`` string, treated as the default
    ``local`` source). The source enumerates and — for remote sources — materializes the documents;
    everything downstream is source-agnostic. Remote downloads are cached under
    ``<root>/.ezpz_cache/sources/<source>/``.
    """
    import ezpz.sources  # noqa: F401  (import = register built-in sources)
    from ezpz.sources.registry import get_source

    spec = DatasetSpec.coerce(spec)
    cache_dir = Path(root) / ".ezpz_cache" / "sources" / spec.source
    source = get_source(spec.source)(spec.config)
    return source.load(spec, root=root, cache_dir=cache_dir)


def resolve_task(root: str, ref: str) -> Task:
    """Resolve "name@version" to ``<root>/tasks/<name>.yaml``."""
    name, _version = _split_ref(ref)
    return Task.load_yaml(str(Path(root) / "tasks" / f"{name}.yaml"))


class ExperimentConfig(BaseModel):
    dataset: DatasetSpec              # "name@version" string OR a {source, name, version, config} map
    task: str                         # "name@version"
    pipelines: list[PipelineConfig]   # the swappable tools under test
    scorers: list[ScorerRef] = Field(default_factory=list)  # defaults; task per-field overrides win
    options: RunOptions = Field(default_factory=RunOptions)
    plugins: list[str] = Field(default_factory=list)  # extra adapter/scorer modules to import

    @field_validator("dataset", mode="before")
    @classmethod
    def _coerce_dataset(cls, v: Any) -> DatasetSpec:
        return DatasetSpec.coerce(v)

    @classmethod
    def load_yaml(cls, path: str) -> "ExperimentConfig":
        import yaml
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls.model_validate(data)
