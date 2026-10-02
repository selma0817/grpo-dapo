"""Offline tests for baseline evaluation metrics."""

import pytest

from grpo_dapo.metrics import group_stats, pass_at_k, summarize


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        # pass@1 is c/n.
        pytest.param(8, 3, 1, 0.375),
        # Standard pass@4 estimator.
        pytest.param(8, 3, 4, 1 - 5 / 70),
        # Standard pass@2 estimator.
        pytest.param(8, 3, 2, 1 - 10 / 28),
        # No correct samples can never pass.
        pytest.param(8, 0, 4, 0.0),
        # All-correct samples always pass.
        pytest.param(8, 8, 1, 1.0),
        # Every four-sample subset contains a correct answer.
        pytest.param(8, 5, 4, 1.0),
    ],
)
def test_pass_at_k(n: int, c: int, k: int, expected: float) -> None:
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_group_stats_classification_and_histogram() -> None:
    stats = group_stats([8, 0, 4, 1], n=8)

    assert stats["all_correct_rate"] == 0.25
    assert stats["all_wrong_rate"] == 0.25
    assert stats["mixed_rate"] == 0.5
    assert stats["histogram"] == {
        0: 1,
        1: 1,
        2: 0,
        3: 0,
        4: 1,
        5: 0,
        6: 0,
        7: 0,
        8: 1,
    }


def test_group_stats_maximal_variance() -> None:
    assert group_stats([4], n=8)["mean_p_one_minus_p"] == 0.25


def test_group_stats_uniform_groups_have_no_signal() -> None:
    assert group_stats([0, 8], n=8)["mean_p_one_minus_p"] == 0.0


def test_summarize() -> None:
    records = [
        {
            "correct": True,
            "extracted_answer": "72",
            "parsed_answer": "72",
            "truncated": False,
            "num_tokens": 1,
        },
        {
            "correct": False,
            "extracted_answer": "70",
            "parsed_answer": "70",
            "truncated": False,
            "num_tokens": 2,
        },
        {
            "correct": False,
            "extracted_answer": None,
            "parsed_answer": None,
            "truncated": False,
            "num_tokens": 3,
        },
        {
            "correct": False,
            "extracted_answer": "12 or 72",
            "parsed_answer": None,
            "truncated": False,
            "num_tokens": 4,
        },
        {
            "correct": False,
            "extracted_answer": None,
            "parsed_answer": None,
            "truncated": True,
            "num_tokens": 5,
        },
    ]

    summary = summarize(records)

    assert summary["accuracy"] == 1 / 5
    assert summary["format_rate"] == 3 / 5
    assert summary["unparseable_rate"] == 1 / 5
    assert summary["truncation_rate"] == 1 / 5
    assert summary["response_length"] == {
        "mean": 3.0,
        "median": 3,
        "p90": 5.0,
        "max": 5,
    }
