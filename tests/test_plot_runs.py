"""Tests for loading, summarizing, and plotting training runs."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "plot_runs.py"
_SPEC = importlib.util.spec_from_file_location("plot_runs", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
plot_runs_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = plot_runs_module
_SPEC.loader.exec_module(plot_runs_module)

FIGURES = plot_runs_module.FIGURES
load_baseline = plot_runs_module.load_baseline
load_run = plot_runs_module.load_run
main = plot_runs_module.main
moving_average = plot_runs_module.moving_average
render_summary = plot_runs_module.render_summary
summarize_run = plot_runs_module.summarize_run


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, float | int]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _make_run(tmp_path: Path, name: str, *, scale: float = 1.0) -> Path:
    run_directory = tmp_path / name
    run_directory.mkdir()
    _write_json(run_directory / "run.json", {"config": {"run_name": name}})

    metrics: list[dict[str, float | int]] = []
    for step in range(1, 7):
        metrics.append(
            {
                "step": step,
                "reward/format_rate": scale * step / 10,
                "length/truncation_rate": scale * (7 - step) / 10,
                "length/mean": scale * (10 + step),
                "length/max": scale * (20 + step),
                "groups/all_correct": scale * step / 20,
                "groups/all_wrong": scale * (7 - step) / 20,
                "groups/mixed": 0.5,
                "policy/entropy": scale * (7 - step) / 10,
                "policy/kl_mean": scale * step / 1000,
                "clip/low": scale * step / 100,
                "clip/high": scale * step / 200,
                "clip/total": scale * 3 * step / 200,
                "ratio/min": 1 - scale * step / 100,
                "ratio/max": 1 + scale * step / 100,
                "health/first_update_max_abs_log_ratio": scale * step / 1000,
                "optim/grad_norm": scale * step,
                "optim/loss": scale / step,
                "cost/step_seconds": float(step),
                "cost/peak_gpu_memory_gb": scale * (2 + step / 10),
                "cost/tokens_generated_total": 100 * step,
            }
        )
    _write_jsonl(run_directory / "metrics.jsonl", metrics)
    _write_jsonl(
        run_directory / "eval.jsonl",
        [
            {
                "step": 0,
                "eval/greedy_accuracy": 0.1 * scale,
                "eval/pass@1": 0.2 * scale,
                "eval/pass@4": 0.4 * scale,
                "eval/sampled_format_rate": 0.6 * scale,
                "eval/sampled_truncation_rate": 0.3 * scale,
            },
            {
                "step": 6,
                "eval/greedy_accuracy": 0.3 * scale,
                "eval/pass@1": 0.4 * scale,
                "eval/pass@4": 0.7 * scale,
                "eval/sampled_format_rate": 0.8 * scale,
                "eval/sampled_truncation_rate": 0.1 * scale,
            },
        ],
    )
    _write_json(
        run_directory / "summary.json",
        {"run_name": name, "steps": 6, "runtime_seconds": 42.0},
    )
    _write_json(
        run_directory / "final_eval.json",
        {
            "metrics": {
                "eval/greedy_accuracy": 0.35 * scale,
                "eval/pass@1": 0.45 * scale,
                "eval/pass@8": 0.9 * scale,
                "eval/sampled_format_rate": 0.85 * scale,
                "eval/sampled_truncation_rate": 0.05 * scale,
            }
        },
    )
    return run_directory


def _make_baseline(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.json"
    _write_json(
        path,
        {
            "metrics": {
                "greedy": {"accuracy": 0.25},
                "sampled": {"pass_at_k": {"1": 0.2}},
            }
        },
    )
    return path


def test_moving_average_uses_trailing_partial_windows() -> None:
    assert moving_average([1.0, 2.0, 6.0, 7.0], 3) == pytest.approx(
        [1.0, 1.5, 3.0, 5.0]
    )
    with pytest.raises(ValueError, match="positive"):
        moving_average([1.0], 0)


def test_load_run_and_baseline(tmp_path: Path) -> None:
    run_directory = _make_run(tmp_path, "vanilla_s0")

    run = load_run(run_directory)
    baseline = load_baseline(_make_baseline(tmp_path))

    assert run.name == "vanilla_s0"
    assert len(run.metrics) == 6
    assert [record["step"] for record in run.evaluations] == [0, 6]
    assert run.final_evaluation is not None
    assert baseline.greedy_accuracy == 0.25
    assert baseline.sampled_pass_at_1 == 0.2


def test_summarize_run_and_render_markdown(tmp_path: Path) -> None:
    summary = summarize_run(load_run(_make_run(tmp_path, "vanilla_s0")))

    assert summary.steps == 6
    assert summary.runtime_seconds == 42.0
    assert summary.mean_step_seconds == 3.5
    assert summary.peak_gpu_memory_gb == 2.6
    assert summary.format_first_five == pytest.approx(0.3)
    assert summary.format_last_five == pytest.approx(0.4)
    assert summary.truncation_first_five == pytest.approx(0.4)
    assert summary.truncation_last_five == pytest.approx(0.3)
    assert summary.entropy_first_five == pytest.approx(0.4)
    assert summary.entropy_last_five == pytest.approx(0.3)
    assert summary.last_kl == 0.006
    assert summary.initial_evaluation is not None
    assert summary.initial_evaluation.greedy_accuracy == 0.1
    assert summary.last_evaluation is not None
    assert summary.last_evaluation.largest_pass_k == 4
    assert summary.final_evaluation is not None
    assert summary.final_evaluation.largest_pass_k == 8

    markdown = render_summary([summary])
    assert markdown.count("\n| vanilla_s0 |") == 1
    assert "p@8=0.9000" in markdown
    assert "0.3000 → 0.4000" in markdown


def test_cli_writes_all_figures_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _make_run(tmp_path, "vanilla_s0")
    second = _make_run(tmp_path, "vanilla_s1", scale=0.8)
    baseline = _make_baseline(tmp_path)
    output = tmp_path / "plots"

    assert main(
        [
            str(first),
            str(second),
            "--out",
            str(output),
            "--smooth",
            "2",
            "--baseline",
            str(baseline),
        ]
    ) == 0

    for filename in FIGURES:
        figure = output / filename
        assert figure.is_file()
        assert figure.stat().st_size > 0
    summary_path = output / "summary.md"
    assert summary_path.is_file()
    printed = capsys.readouterr().out
    assert printed == summary_path.read_text(encoding="utf-8")
    assert "vanilla_s0" in printed
    assert "vanilla_s1" in printed
