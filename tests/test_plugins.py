"""Plugin discovery (entry points + experiment `plugins:`) and `ezpz init` scaffolding.

Custom adapters from a consuming project are additive — the built-ins still work (Case A).
"""
from typer.testing import CliRunner

from ezpz.adapters.registry import get_adapter
from ezpz.cli.main import app
from ezpz.config.experiment import ExperimentConfig, resolve_dataset, resolve_task


def test_init_scaffolds_a_resolvable_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["init", "."])
    assert result.exit_code == 0
    for rel in ("experiments/example.yaml", "tasks/example_task.yaml",
                "datasets/example/manifest.jsonl", "datasets/example/docs/doc-001.txt",
                "datasets/example/ground_truth/doc-001.json", "custom_adapter_example.py",
                ".env.example"):
        assert (tmp_path / rel).exists(), rel
    exp = ExperimentConfig.load_yaml(str(tmp_path / "experiments" / "example.yaml"))
    dataset = resolve_dataset(str(tmp_path), exp.dataset)
    task = resolve_task(str(tmp_path), exp.task)
    assert len(dataset.documents) == 1
    assert [f.name for f in task.fields] == ["invoice_number", "total_amount"]


def test_init_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "example.yaml").write_text("MINE")
    CliRunner().invoke(app, ["init", "."])
    assert (tmp_path / "experiments" / "example.yaml").read_text() == "MINE"


def test_scaffolded_example_runs_with_no_key(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    CliRunner().invoke(app, ["init", "."])
    result = CliRunner().invoke(app, ["run", "experiments/example.yaml", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    accuracies = sorted(s["accuracy"] for s in data["summary"].values())
    assert accuracies == [0.5, 1.0]  # 'typo' 50%, 'perfect' 100% — both fake pipelines ran


def test_experiment_plugins_field_imports_a_local_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "my_local_adapter.py").write_text(
        "from ezpz.adapters.base import Capabilities, Pipeline\n"
        "from ezpz.adapters.registry import register\n"
        "from ezpz.core.result import Cost, FieldValue\n"
        "@register('plugin_local')\n"
        "class P(Pipeline):\n"
        "    capabilities = Capabilities()\n"
        "    def compile(self, task): return [f.name for f in task.fields]\n"
        "    def ingest(self, document): return document\n"
        "    def invoke(self, prepared, ingested): return {n: None for n in prepared}, Cost(usd=0.0)\n"
        "    def map(self, raw, task): return {n: FieldValue(value=v) for n, v in raw.items()}\n"
    )
    from ezpz.plugins import import_modules
    import_modules(["my_local_adapter"])
    assert get_adapter("plugin_local") is not None


def test_load_plugins_imports_entry_point_modules(monkeypatch):
    import ezpz.plugins as plugins
    from ezpz.adapters.base import Pipeline
    from ezpz.adapters.registry import register

    class _FakeEP:
        name = "demo"

        def load(self):
            @register("entrypoint_demo_adapter")
            class _Demo(Pipeline):
                pass
            return _Demo

    monkeypatch.setattr(plugins, "_loaded", False)
    monkeypatch.setattr(plugins, "entry_points",
                        lambda group=None: [_FakeEP()] if group == "ezpz.adapters" else [])
    loaded = plugins.load_plugins(force=True)
    assert "ezpz.adapters:demo" in loaded
    assert get_adapter("entrypoint_demo_adapter") is not None
