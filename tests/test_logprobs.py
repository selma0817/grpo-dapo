"""Tests for shifted token log-probabilities, entropy, and masks."""

import math

import torch

from grpo_dapo.logprobs import completion_mask, token_entropy, token_log_probs


def test_token_log_probs_matches_log_softmax_reference() -> None:
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(2, 5, 7, dtype=torch.float64, generator=generator)
    input_ids = torch.randint(7, (2, 5), generator=generator)

    actual = token_log_probs(logits, input_ids)
    expected = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(
        -1, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_token_log_probs_bfloat16_returns_shifted_float32() -> None:
    logits = torch.randn(3, 6, 9, dtype=torch.bfloat16)
    input_ids = torch.randint(9, (3, 6))

    actual = token_log_probs(logits, input_ids)

    assert actual.dtype == torch.float32
    assert actual.shape == (3, 5)


def test_token_log_probs_uses_next_token_alignment() -> None:
    input_ids = torch.tensor([[1, 3, 0, 4, 2]])
    logits = torch.full((1, 5, 5), -100.0)
    for position, next_token in enumerate(input_ids[0, 1:]):
        logits[0, position, next_token] = 100.0

    actual = token_log_probs(logits, input_ids)

    torch.testing.assert_close(actual, torch.zeros(1, 4), atol=1e-6, rtol=0.0)


def test_token_entropy_uniform_distribution_is_log_vocab_size() -> None:
    logits = torch.zeros(2, 5, 7, dtype=torch.float64)

    actual = token_entropy(logits)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual,
        torch.full((2, 4), math.log(7)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_token_entropy_one_dominant_logit_is_near_zero() -> None:
    logits = torch.full((2, 5, 7), -100.0)
    logits[:, :, 3] = 100.0

    actual = token_entropy(logits)

    torch.testing.assert_close(actual, torch.zeros(2, 4), atol=1e-6, rtol=0.0)


def test_completion_mask_matches_documented_example() -> None:
    actual = completion_mask(
        prompt_lengths=torch.tensor([3]),
        completion_lengths=torch.tensor([4]),
        seq_len=10,
    )

    expected = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_completion_mask_handles_different_lengths_per_row() -> None:
    actual = completion_mask(
        prompt_lengths=torch.tensor([2, 4]),
        completion_lengths=torch.tensor([3, 2]),
        seq_len=8,
    )

    expected = torch.tensor(
        [
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(dim=1), torch.tensor([3.0, 2.0]))
