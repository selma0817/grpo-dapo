#!/usr/bin/env python3
"""Write the DAPO ablation table (Markdown) from training run directories."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_runs import RunData, load_run  # noqa: E402

TECHNIQUES = ("Clip-Higher", "Dynamic Sampling", "Token-level", "Overlong")
GSM8K_TEST_QUESTIONS = 1319


@dataclass(frozen=True)
class Difference:
    """Greedy-accuracy difference from the baseline, in fractions."""

    value: float
    standard_error: float
    paired: bool


@dataclass(frozen=True)
class AblationRow:
    """One run's line in the ablation table."""

    name: str
    techniques: tuple[bool, bool, bool, bool]
    greedy_accuracy: float
    difference: Difference | None
    pass_at_1: float
    pass_at_8: float
    greedy_format: float
    greedy_truncation: float
    mean_length: float
    samples_generated: float | None
    runtime_seconds: float | None


def techniques(config: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
    """Which DAPO techniques a run's resolved config turns on."""
    clip_higher = config.get("eps_high", 0.2) > config.get("eps_low", 0.2)
    dynamic = bool(config.get("dynamic_sampling", False))
    token_level = config.get("aggregation", "sample") == "token"
    overlong = config.get("overlong_cache", 0) > 0 or bool(
        config.get("overlong_filter", False)
    )
    return clip_higher, dynamic, token_level, overlong


def greedy_correct(run: RunData) -> dict[Any, bool] | None:
    """Per-question greedy correctness from final_eval.json, if it was kept."""
    if run.final_evaluation is None:
        return None
    records = run.final_evaluation.get("records", {}).get("greedy")
    if not records:
        return None
    return {record["question_id"]: bool(record["correct"]) for record in records}


def _final(run: RunData, key: str) -> float:
    return float(run.summary["final_eval"][f"eval/{key}"])


def accuracy_difference(
    run: RunData, baseline: RunData, n_questions: int
) -> Difference:
    """Greedy-accuracy difference with one standard error.

    Paired over the same questions when both runs kept per-question records
    (the test set is identical across runs, so question difficulty cancels);
    otherwise the unpaired binomial standard error over ``n_questions``.
    """
    run_correct = greedy_correct(run)
    baseline_correct = greedy_correct(baseline)
    if (
        run_correct is not None
        and baseline_correct is not None
        and run_correct.keys() == baseline_correct.keys()
        and len(run_correct) > 1
    ):
        differences = [
            float(run_correct[key]) - float(baseline_correct[key])
            for key in run_correct
        ]
        return Difference(
            value=statistics.fmean(differences),
            standard_error=statistics.stdev(differences) / math.sqrt(len(differences)),
            paired=True,
        )
    p_run = _final(run, "greedy_accuracy")
    p_baseline = _final(baseline, "greedy_accuracy")
    return Difference(
        value=p_run - p_baseline,
        standard_error=math.sqrt(
            (p_run * (1 - p_run) + p_baseline * (1 - p_baseline)) / n_questions
        ),
        paired=False,
    )


def build_row(
    run: RunData, baseline: RunData | None, n_questions: int
) -> AblationRow:
    """Collect one run's final-evaluation numbers and cost."""
    last_metrics = run.metrics[-1] if run.metrics else {}
    difference = (
        accuracy_difference(run, baseline, n_questions)
        if baseline is not None and run.path.resolve() != baseline.path.resolve()
        else None
    )
    return AblationRow(
        name=run.name,
        techniques=techniques(run.run.get("config", {})),
        greedy_accuracy=_final(run, "greedy_accuracy"),
        difference=difference,
        pass_at_1=_final(run, "pass@1"),
        pass_at_8=_final(run, "pass@8"),
        greedy_format=_final(run, "greedy_format_rate"),
        greedy_truncation=_final(run, "greedy_truncation_rate"),
        mean_length=_final(run, "mean_length"),
        samples_generated=last_metrics.get("cost/samples_generated_total"),
        runtime_seconds=run.summary.get("runtime_seconds"),
    )


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _difference(difference: Difference | None) -> str:
    if difference is None:
        return "—"
    marker = "" if difference.paired else "†"
    return (
        f"{100 * difference.value:+.1f} ± {100 * difference.standard_error:.1f}"
        f"{marker}"
    )


def render_table(rows: Sequence[AblationRow], baseline_name: str | None) -> str:
    """Render ablation rows as a Markdown table with a footnote."""
    headers = (
        "Run",
        *TECHNIQUES,
        "Greedy acc.",
        f"Δ vs {baseline_name}" if baseline_name else "Δ",
        "pass@1",
        "pass@8",
        "Greedy format",
        "Greedy trunc.",
        "Mean length",
        "Samples generated",
        "Runtime",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = (
            row.name.replace("|", "\\|"),
            *("✓" if on else "" for on in row.techniques),
            _percent(row.greedy_accuracy),
            _difference(row.difference),
            _percent(row.pass_at_1),
            _percent(row.pass_at_8),
            _percent(row.greedy_format),
            _percent(row.greedy_truncation),
            f"{row.mean_length:.0f}",
            "—" if row.samples_generated is None else f"{row.samples_generated:,.0f}",
            "—"
            if row.runtime_seconds is None
            else f"{row.runtime_seconds / 60:.0f} min",
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Final evaluation on all GSM8K test questions (greedy, plus 8 samples at "
        "T = 1 for pass@k). Δ is in percentage points with one standard error, "
        "paired over the same questions; † marks an unpaired standard error "
        "(per-question records not available)."
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, metavar="RUN_DIR")
    parser.add_argument("--baseline", type=Path, help="run to compute Δ against")
    parser.add_argument(
        "--n-questions",
        type=int,
        default=GSM8K_TEST_QUESTIONS,
        help="test-set size for the unpaired standard error",
    )
    parser.add_argument("--out", type=Path, help="also write the table here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print (and optionally write) the ablation table."""
    args = build_parser().parse_args(argv)
    runs = [load_run(path) for path in args.run_dirs]
    baseline = load_run(args.baseline) if args.baseline else None
    rows = [build_row(run, baseline, args.n_questions) for run in runs]
    markdown = render_table(rows, baseline.name if baseline else None)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
