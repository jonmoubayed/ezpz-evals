"""YAML loaders: the example task and experiment must parse into the contracts."""
from pathlib import Path

from ezpz.config.experiment import ExperimentConfig
from ezpz.core.task import Task

REPO = Path(__file__).resolve().parents[1]


def test_task_load_yaml_parses_example_task():
    task = Task.load_yaml(str(REPO / "examples" / "tasks" / "invoice_extraction.yaml"))
    assert task.name == "invoice_extraction"
    assert task.version == "1"

    line_items = next(f for f in task.fields if f.name == "line_items")
    assert line_items.match_key == "description"
    assert line_items.item is not None
    assert [g.name for g in line_items.item.fields] == [
        "description",
        "quantity",
        "unit_price",
    ]


def test_experiment_load_yaml_parses_example_experiment():
    exp = ExperimentConfig.load_yaml(
        str(REPO / "examples" / "experiments" / "extend_vs_gemini_vs_llamaindex.yaml")
    )
    assert exp.dataset == "invoices_v1@1"
    assert exp.task == "invoice_extraction@1"
    assert len(exp.pipelines) == 4
    assert exp.options.budget_usd == 5
    # two gemini configs differ only by model -> distinct pipeline identities
    hashes = [p.config_hash for p in exp.pipelines]
    assert len(set(hashes)) == len(hashes)
