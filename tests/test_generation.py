"""Offline tests for generation token accounting."""

import pytest

from grpo_dapo.generation import completion_length, is_truncated


@pytest.mark.parametrize(
    ("token_ids", "expected_length", "expected_truncated"),
    [
        # Stop at the first EOS; trailing padding does not count.
        pytest.param([7, 8, 2, 2, 2], 3, False),
        # Using every token without EOS is truncation.
        pytest.param([7, 8, 9, 10, 11], 5, True),
        # EOS on the final allowed token is a normal stop.
        pytest.param([7, 8, 9, 10, 2], 5, False),
        # Immediate EOS is an empty one-token answer.
        pytest.param([2, 2, 2, 2, 2], 1, False),
    ],
)
def test_completion_accounting(
    token_ids: list[int],
    expected_length: int,
    expected_truncated: bool,
) -> None:
    assert completion_length(token_ids, {2}) == expected_length
    assert is_truncated(token_ids, {2}, max_new_tokens=5) is expected_truncated
