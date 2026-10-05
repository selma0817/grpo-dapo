"""Tests for group-relative advantages."""

import pytest
import torch

from grpo_dapo.advantage import group_advantages, zero_variance_groups


def test_group_advantages_balanced_binary_rewards() -> None:
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    expected_scale = 0.5 / (0.5773502691896257 + 1e-4)
    expected = torch.tensor(
        [[expected_scale, -expected_scale, -expected_scale, expected_scale]]
    )

    torch.testing.assert_close(group_advantages(rewards), expected, atol=5e-5, rtol=0.0)


def test_group_advantages_rare_positive_reward() -> None:
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    expected = torch.tensor([[0.75, -0.25, -0.25, -0.25]]) / (0.5 + 1e-4)

    torch.testing.assert_close(group_advantages(rewards), expected)


@pytest.mark.parametrize("reward", [1.0, 0.0])
def test_equal_reward_group_has_zero_advantages(reward: float) -> None:
    rewards = torch.full((1, 4), reward)

    torch.testing.assert_close(group_advantages(rewards), torch.zeros_like(rewards))


@pytest.mark.parametrize("reward", [1.0, 0.0])
def test_equal_reward_group_is_flagged_as_zero_variance(reward: float) -> None:
    rewards = torch.full((1, 4), reward)

    torch.testing.assert_close(
        zero_variance_groups(rewards), torch.tensor([True])
    )


def test_group_advantages_matches_float64_trl_formula() -> None:
    generator = torch.Generator().manual_seed(23)
    rewards = torch.randn(16, 8, generator=generator)
    rewards64 = rewards.double()
    expected = (
        rewards64 - rewards64.mean(dim=1, keepdim=True)
    ) / (rewards64.std(dim=1, keepdim=True, correction=1) + 1e-4)

    actual = group_advantages(rewards)

    torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-6)


def test_group_advantages_rejects_single_completion_groups() -> None:
    with pytest.raises(ValueError):
        group_advantages(torch.tensor([[1.0], [0.0]]))


def test_zero_variance_groups_flags_only_equal_rows() -> None:
    rewards = torch.tensor(
        [[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    )

    torch.testing.assert_close(
        zero_variance_groups(rewards), torch.tensor([False, True, True])
    )
