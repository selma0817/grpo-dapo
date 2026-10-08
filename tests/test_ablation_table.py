"""Tests for the ablation table: technique flags, Δ with standard errors, rendering."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "ablation_table.py"
_SPEC = importlib.util.spec_from_file_location("ablation_table", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
ablation_table = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ablation_table
_SPEC.loader.exec_module(ablation_table)

FINAL = {
    "eval/greedy_accuracy": 0.5,
    "eval/greedy_format_rate": 0.9,
    "eval/greedy_truncation_rate": 0.05,
    "eval/mean_length": 180.0,
    "eval/pass@1": 0.45,
    "eval/pass@8": 0.8,
}


def _make_run(tmp_path: Path, name: str, config: dict, *, accuracy: float = 0.5,
              greedy: list[bool] | None = None) -> Path:
    run_directory = tmp_path / name
    run_directory.mkdir()
    (run_directory / "run.json").write_text(
        json.dumps({"config": {"run_name": name, **config}})
    )
    (run_directory / "summary.json").write_text(
        json.dumps({"run_name": name, "runtime_seconds": 3600,
                    "final_eval": {**FINAL, "eval/greedy_accuracy": accuracy}})
    )
    (run_directory / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "cost/samples_generated_total": 25600}) + "\n"
    )
    (run_directory / "eval.jsonl").write_text("")
    if greedy is not None:
        records = [{"question_id": index, "correct": correct}
                   for index, correct in enumerate(greedy)]
        (run_directory / "final_eval.json").write_text(
            json.dumps({"metrics": {}, "records": {"greedy": records, "sampled": []}})
        )
    return run_directory


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, (False, False, False, False)),
        ({"eps_low": 0.2, "eps_high": 0.28}, (True, False, False, False)),
        ({"dynamic_sampling": True}, (False, True, False, False)),
        ({"aggregation": "token"}, (False, False, True, False)),
        ({"overlong_cache": 64}, (False, False, False, True)),
        ({"overlong_filter": True}, (False, False, False, True)),
        ({"eps_high": 0.28, "aggregation": "token", "dynamic_sampling": True,
          "overlong_cache": 64}, (True, True, True, True)),
    ],
)
def test_techniques_from_config(config: dict, expected: tuple) -> None:
    assert ablation_table.techniques(config) == expected


def test_paired_difference_uses_per_question_records(tmp_path: Path) -> None:
    baseline = ablation_table.load_run(
        _make_run(tmp_path, "vanilla", {}, greedy=[True, False, False, True])
    )
    run = ablation_table.load_run(
        _make_run(tmp_path, "clip", {"eps_high": 0.28}, greedy=[True, True, False, True])
    )

    difference = ablation_table.accuracy_difference(run, baseline, n_questions=4)

    # d = [0, 1, 0, 0]: mean 0.25, sample std 0.5, SE 0.5 / 2
    assert difference.paired
    assert difference.value == pytest.approx(0.25)
    assert difference.standard_error == pytest.approx(0.25)


def test_unpaired_difference_without_records(tmp_path: Path) -> None:
    baseline = ablation_table.load_run(_make_run(tmp_path, "vanilla", {}, accuracy=0.4))
    run = ablation_table.load_run(_make_run(tmp_path, "dapo", {}, accuracy=0.5))

    difference = ablation_table.accuracy_difference(run, baseline, n_questions=100)

    assert not difference.paired
    assert difference.value == pytest.approx(0.1)
    assert difference.standard_error == pytest.approx(
        math.sqrt((0.5 * 0.5 + 0.4 * 0.6) / 100)
    )


def test_main_renders_table_with_baseline_row_blank(tmp_path: Path, capsys) -> None:
    vanilla = _make_run(tmp_path, "vanilla", {}, accuracy=0.4)
    dapo = _make_run(tmp_path, "dapo", {"eps_high": 0.28, "aggregation": "token",
                                        "dynamic_sampling": True, "overlong_cache": 64},
                     accuracy=0.5)
    out = tmp_path / "table.md"

    assert ablation_table.main(
        [str(vanilla), str(dapo), "--baseline", str(vanilla), "--out", str(out)]
    ) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("| Run | Clip-Higher | Dynamic Sampling")
    assert "Δ vs vanilla" in lines[0]
    vanilla_row = lines[2].split(" | ")
    dapo_row = lines[3].split(" | ")
    assert vanilla_row[5] == "40.0%" and vanilla_row[6] == "—"
    assert dapo_row[1:5] == ["✓", "✓", "✓", "✓"]
    assert dapo_row[6].startswith("+10.0 ± ") and dapo_row[6].endswith("†")
    assert "25,600" in lines[3] and "60 min" in lines[3]
    assert out.read_text() == "\n".join(lines) + "\n"
