#!/usr/bin/env python3
"""Plot training runs and write a compact Markdown comparison table."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunData:
    """Loaded artifacts for one training run."""

    path: Path
    name: str
    run: dict[str, Any]
    metrics: tuple[dict[str, Any], ...]
    evaluations: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    final_evaluation: dict[str, Any] | None


@dataclass(frozen=True)
class Baseline:
    """Baseline values shown on the evaluation figure."""

    greedy_accuracy: float
    sampled_pass_at_1: float


@dataclass(frozen=True)
class EvaluationSnapshot:
    """Metrics from one evaluation record."""

    greedy_accuracy: float | None
    pass_at_1: float | None
    largest_pass_k: int | None
    largest_pass_value: float | None
    sampled_format_rate: float | None
    sampled_truncation_rate: float | None


@dataclass(frozen=True)
class RunSummary:
    """Values used for one row of the Markdown summary."""

    name: str
    steps: int
    runtime_seconds: float | None
    mean_step_seconds: float | None
    peak_gpu_memory_gb: float | None
    initial_evaluation: EvaluationSnapshot | None
    last_evaluation: EvaluationSnapshot | None
    final_evaluation: EvaluationSnapshot | None
    format_first_five: float | None
    format_last_five: float | None
    truncation_first_five: float | None
    truncation_last_five: float | None
    entropy_first_five: float | None
    entropy_last_five: float | None
    last_kl: float | None


@dataclass(frozen=True)
class Panel:
    """Definition of one subplot."""

    title: str
    key: str
    source: str = "train"
    log_scale: bool = False


FIGURES: dict[str, tuple[Panel, ...]] = {
    "eval.png": (
        Panel("Greedy accuracy", "eval/greedy_accuracy", "eval"),
        Panel("Sampled pass@1", "eval/pass@1", "eval"),
    ),
    "format_truncation.png": (
        Panel("Training format rate", "reward/format_rate"),
        Panel("Training truncation rate", "length/truncation_rate"),
        Panel("Eval sampled format rate", "eval/sampled_format_rate", "eval"),
        Panel(
            "Eval sampled truncation rate",
            "eval/sampled_truncation_rate",
            "eval",
        ),
    ),
    "length.png": (
        Panel("Mean completion length", "length/mean"),
        Panel("Maximum completion length", "length/max"),
    ),
    "groups.png": (
        Panel("All correct groups", "groups/all_correct"),
        Panel("All wrong groups", "groups/all_wrong"),
        Panel("Mixed groups", "groups/mixed"),
    ),
    "policy.png": (
        Panel("Policy entropy", "policy/entropy"),
        Panel("Mean KL", "policy/kl_mean"),
    ),
    "clipping.png": (
        Panel("Low clipping fraction", "clip/low"),
        Panel("High clipping fraction", "clip/high"),
        Panel("Total clipping fraction", "clip/total"),
        Panel("Minimum ratio", "ratio/min"),
        Panel("Maximum ratio", "ratio/max"),
    ),
    "health_optim.png": (
        Panel(
            "First-update max |log ratio|",
            "health/first_update_max_abs_log_ratio",
            log_scale=True,
        ),
        Panel("Gradient norm", "optim/grad_norm"),
        Panel("Loss", "optim/loss"),
    ),
    "cost.png": (
        Panel("Step time (seconds)", "cost/step_seconds"),
        Panel("Peak GPU memory (GiB)", "cost/peak_gpu_memory_gb"),
        Panel("Tokens generated (cumulative)", "cost/tokens_generated_total"),
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(value)
    return tuple(records)


def load_run(run_directory: str | Path) -> RunData:
    """Load one run directory without mutating its artifacts."""
    path = Path(run_directory)
    run = _read_json(path / "run.json")
    summary = _read_json(path / "summary.json")
    final_path = path / "final_eval.json"
    final_evaluation = _read_json(final_path) if final_path.is_file() else None
    config = run.get("config", {})
    name = str(
        summary.get("run_name")
        or (config.get("run_name") if isinstance(config, dict) else None)
        or path.name
    )
    return RunData(
        path=path,
        name=name,
        run=run,
        metrics=_read_jsonl(path / "metrics.jsonl"),
        evaluations=_read_jsonl(path / "eval.jsonl"),
        summary=summary,
        final_evaluation=final_evaluation,
    )


def load_baseline(path: str | Path) -> Baseline:
    """Load the two baseline values used in the evaluation plot."""
    payload = _read_json(Path(path))
    metrics = payload["metrics"]
    return Baseline(
        greedy_accuracy=float(metrics["greedy"]["accuracy"]),
        sampled_pass_at_1=float(metrics["sampled"]["pass_at_k"]["1"]),
    )


def moving_average(values: Sequence[float], window: int) -> list[float]:
    """Return a trailing moving average with partial windows at the start."""
    if window < 1:
        raise ValueError("moving-average window must be positive")
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def _numeric(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _evaluation_snapshot(record: dict[str, Any] | None) -> EvaluationSnapshot | None:
    if record is None:
        return None
    pass_values: list[tuple[int, float]] = []
    for key, value in record.items():
        if not key.startswith("eval/pass@"):
            continue
        try:
            k = int(key.removeprefix("eval/pass@"))
        except ValueError:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            pass_values.append((k, float(value)))
    largest = max(pass_values, default=None)
    return EvaluationSnapshot(
        greedy_accuracy=_numeric(record, "eval/greedy_accuracy"),
        pass_at_1=_numeric(record, "eval/pass@1"),
        largest_pass_k=largest[0] if largest else None,
        largest_pass_value=largest[1] if largest else None,
        sampled_format_rate=_numeric(record, "eval/sampled_format_rate"),
        sampled_truncation_rate=_numeric(
            record, "eval/sampled_truncation_rate"
        ),
    )


def _final_metrics(run: RunData) -> dict[str, Any] | None:
    final = run.final_evaluation
    if final is not None:
        metrics = final.get("metrics", final)
        if isinstance(metrics, dict):
            return metrics
    summary_final = run.summary.get("final_eval")
    return summary_final if isinstance(summary_final, dict) else None


def _edge_mean(
    records: Sequence[dict[str, Any]], key: str, *, first: bool
) -> float | None:
    values = [value for record in records if (value := _numeric(record, key)) is not None]
    selected = values[:5] if first else values[-5:]
    return sum(selected) / len(selected) if selected else None


def summarize_run(run: RunData) -> RunSummary:
    """Compute one run's report values without changing the loaded data."""
    metrics = sorted(run.metrics, key=lambda record: float(record.get("step", 0)))
    evaluations = sorted(
        run.evaluations, key=lambda record: float(record.get("step", 0))
    )
    initial_record = next(
        (record for record in evaluations if record.get("step") == 0), None
    )
    last_record = evaluations[-1] if evaluations else None
    step_times = [
        value
        for record in metrics
        if (value := _numeric(record, "cost/step_seconds")) is not None
    ]
    peak_memories = [
        value
        for record in metrics
        if (value := _numeric(record, "cost/peak_gpu_memory_gb")) is not None
    ]
    kl_values = [
        value
        for record in metrics
        if (value := _numeric(record, "policy/kl_mean")) is not None
    ]
    fallback_steps = int(metrics[-1].get("step", 0)) if metrics else 0
    steps = int(run.summary.get("steps", fallback_steps))
    runtime = _numeric(run.summary, "runtime_seconds")

    return RunSummary(
        name=run.name,
        steps=steps,
        runtime_seconds=runtime,
        mean_step_seconds=sum(step_times) / len(step_times) if step_times else None,
        peak_gpu_memory_gb=max(peak_memories, default=None),
        initial_evaluation=_evaluation_snapshot(initial_record),
        last_evaluation=_evaluation_snapshot(last_record),
        final_evaluation=_evaluation_snapshot(_final_metrics(run)),
        format_first_five=_edge_mean(metrics, "reward/format_rate", first=True),
        format_last_five=_edge_mean(metrics, "reward/format_rate", first=False),
        truncation_first_five=_edge_mean(
            metrics, "length/truncation_rate", first=True
        ),
        truncation_last_five=_edge_mean(
            metrics, "length/truncation_rate", first=False
        ),
        entropy_first_five=_edge_mean(metrics, "policy/entropy", first=True),
        entropy_last_five=_edge_mean(metrics, "policy/entropy", first=False),
        last_kl=kl_values[-1] if kl_values else None,
    )


def _format_number(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _format_duration(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(value, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def _format_evaluation(snapshot: EvaluationSnapshot | None) -> str:
    if snapshot is None:
        return "—"
    largest = (
        "—"
        if snapshot.largest_pass_k is None
        else f"p@{snapshot.largest_pass_k}="
        f"{_format_number(snapshot.largest_pass_value)}"
    )
    return ", ".join(
        (
            f"greedy={_format_number(snapshot.greedy_accuracy)}",
            f"p@1={_format_number(snapshot.pass_at_1)}",
            largest,
            f"format={_format_number(snapshot.sampled_format_rate)}",
            f"trunc={_format_number(snapshot.sampled_truncation_rate)}",
        )
    )


def _format_change(first: float | None, last: float | None) -> str:
    return f"{_format_number(first)} → {_format_number(last)}"


def render_summary(summaries: Sequence[RunSummary]) -> str:
    """Render run summaries as a Markdown table."""
    headers = (
        "Run",
        "Steps",
        "Runtime",
        "Mean step",
        "Peak GPU GiB",
        "Eval step 0",
        "Last eval",
        "Final eval",
        "Train format first→last 5",
        "Train trunc first→last 5",
        "Entropy first→last 5",
        "Last KL",
    )
    rows: list[tuple[str, ...]] = []
    for summary in summaries:
        rows.append(
            (
                summary.name.replace("|", "\\|"),
                str(summary.steps),
                _format_duration(summary.runtime_seconds),
                _format_duration(summary.mean_step_seconds),
                _format_number(summary.peak_gpu_memory_gb, 2),
                _format_evaluation(summary.initial_evaluation),
                _format_evaluation(summary.last_evaluation),
                _format_evaluation(summary.final_evaluation),
                _format_change(
                    summary.format_first_five, summary.format_last_five
                ),
                _format_change(
                    summary.truncation_first_five,
                    summary.truncation_last_five,
                ),
                _format_change(
                    summary.entropy_first_five,
                    summary.entropy_last_five,
                ),
                _format_number(summary.last_kl, 6),
            )
        )
    separator = tuple("---" for _ in headers)
    lines = [
        "# Training run summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _plot_panel(
    axis: Any,
    panel: Panel,
    runs: Sequence[RunData],
    smooth: int,
    baseline: Baseline | None,
) -> None:
    for run in runs:
        records = run.evaluations if panel.source == "eval" else run.metrics
        points = [
            (float(record["step"]), value)
            for record in records
            if "step" in record and (value := _numeric(record, panel.key)) is not None
        ]
        if not points:
            continue
        points.sort()
        steps, values = zip(*points)
        plotted_values = (
            list(values)
            if panel.source == "eval"
            else moving_average(values, smooth)
        )
        if panel.log_scale:
            plotted_values = [max(value, 1e-12) for value in plotted_values]
        axis.plot(steps, plotted_values, marker="o", markersize=3, label=run.name)

    if baseline is not None and panel.key == "eval/greedy_accuracy":
        axis.axhline(
            baseline.greedy_accuracy,
            color="black",
            linestyle="--",
            linewidth=1.25,
            label="baseline",
        )
    if baseline is not None and panel.key == "eval/pass@1":
        axis.axhline(
            baseline.sampled_pass_at_1,
            color="black",
            linestyle="--",
            linewidth=1.25,
            label="baseline",
        )
    if panel.log_scale:
        axis.set_yscale("log", nonpositive="clip")
    axis.set_title(panel.title)
    axis.set_xlabel("Training step")
    axis.grid(alpha=0.25)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize="small")


def plot_runs(
    runs: Sequence[RunData],
    output_directory: str | Path,
    *,
    smooth: int = 5,
    baseline: Baseline | None = None,
) -> tuple[Path, ...]:
    """Write all run-comparison figures and return their paths."""
    if smooth < 1:
        raise ValueError("smooth must be positive")
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for filename, panels in FIGURES.items():
        columns = min(3, len(panels))
        rows = math.ceil(len(panels) / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(5.25 * columns, 3.8 * rows),
            squeeze=False,
        )
        flat_axes = list(axes.flat)
        for axis, panel in zip(flat_axes, panels):
            _plot_panel(axis, panel, runs, smooth, baseline)
        for axis in flat_axes[len(panels) :]:
            axis.remove()
        figure.tight_layout()
        destination = output_path / filename
        figure.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(figure)
        written.append(destination)
    return tuple(written)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Overlay GRPO training runs and write a Markdown summary."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, metavar="RUN_DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--smooth", type=int, default=5, metavar="STEPS")
    parser.add_argument("--baseline", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the plotting command."""
    args = build_parser().parse_args(argv)
    if args.smooth < 1:
        raise SystemExit("--smooth must be positive")
    runs = tuple(load_run(path) for path in args.run_dirs)
    baseline = load_baseline(args.baseline) if args.baseline else None
    plot_runs(runs, args.out, smooth=args.smooth, baseline=baseline)
    markdown = render_summary(tuple(summarize_run(run) for run in runs))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
