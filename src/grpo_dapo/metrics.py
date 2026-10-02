"""Pure metric helpers for baseline evaluation."""

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute the unbiased pass-at-k estimator."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= c <= n:
        raise ValueError("c must be between zero and n")
    if not 1 <= k <= n:
        raise ValueError("k must be between one and n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def group_stats(correct_counts: Sequence[int], n: int) -> dict[str, Any]:
    """Summarize correctness counts for fixed-size sampled groups."""
    if n <= 0:
        raise ValueError("n must be positive")
    if any(not 0 <= count <= n for count in correct_counts):
        raise ValueError("correct counts must be between zero and n")

    total = len(correct_counts)
    histogram = {count: 0 for count in range(n + 1)}
    for count in correct_counts:
        histogram[count] += 1

    if total == 0:
        return {
            "all_correct_rate": 0.0,
            "all_wrong_rate": 0.0,
            "mixed_rate": 0.0,
            "histogram": histogram,
            "mean_p_one_minus_p": 0.0,
        }

    all_correct = histogram[n]
    all_wrong = histogram[0]
    variances = [(count / n) * (1.0 - count / n) for count in correct_counts]
    return {
        "all_correct_rate": all_correct / total,
        "all_wrong_rate": all_wrong / total,
        "mixed_rate": (total - all_correct - all_wrong) / total,
        "histogram": histogram,
        "mean_p_one_minus_p": statistics.fmean(variances),
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize completion-level accuracy, validity, and length metrics."""
    total = len(records)
    if total == 0:
        return {
            "accuracy": 0.0,
            "format_rate": 0.0,
            "unparseable_rate": 0.0,
            "truncation_rate": 0.0,
            "response_length": {
                "mean": 0.0,
                "median": 0.0,
                "p90": 0.0,
                "max": 0,
            },
        }

    lengths = [int(record["num_tokens"]) for record in records]
    sorted_lengths = sorted(lengths)
    p90_index = max(0, math.ceil(0.9 * total) - 1)
    return {
        "accuracy": sum(bool(record["correct"]) for record in records) / total,
        "format_rate": (
            sum(record["extracted_answer"] is not None for record in records)
            / total
        ),
        "unparseable_rate": (
            sum(
                record["extracted_answer"] is not None
                and record["parsed_answer"] is None
                for record in records
            )
            / total
        ),
        "truncation_rate": (
            sum(bool(record["truncated"]) for record in records) / total
        ),
        "response_length": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p90": float(sorted_lengths[p90_index]),
            "max": max(lengths),
        },
    }
