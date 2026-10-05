"""Tests for the clipped GRPO and DAPO policy loss."""

import pytest
import torch
from torch import Tensor

from grpo_dapo.loss import grpo_loss


OLD_PROBS = torch.tensor([[0.9, 0.3], [0.9, 0.6]], dtype=torch.float64)
ADVANTAGES = torch.tensor([1.0, -1.0], dtype=torch.float64)
FULL_MASK = torch.ones(2, 2, dtype=torch.float64)


def _loss_reference(
    logp_new: Tensor,
    logp_old: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    eps_low: float,
    eps_high: float,
    beta: float,
    logp_ref: Tensor | None,
    aggregation: str,
) -> Tensor:
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantages.unsqueeze(1)
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * advantages.unsqueeze(1)
    per_token_loss = -torch.minimum(unclipped, clipped)
    if beta > 0:
        assert logp_ref is not None
        log_ratio = logp_ref - logp_new
        per_token_loss = per_token_loss + beta * (
            torch.exp(log_ratio) - log_ratio - 1
        )
    if aggregation == "sample":
        return ((per_token_loss * mask).sum(1) / mask.sum(1).clamp(min=1)).sum() / len(mask)
    return (per_token_loss * mask).sum() / mask.sum()


def test_step_two_sample_loss_is_unclipped() -> None:
    logp_new = torch.log(
        torch.tensor([[0.9, 0.345], [0.9, 0.555]], dtype=torch.float64)
    )

    loss, stats = grpo_loss(
        logp_new, OLD_PROBS.log(), ADVANTAGES, FULL_MASK
    )

    torch.testing.assert_close(
        loss, torch.tensor(-0.05625, dtype=torch.float64), atol=1e-7, rtol=0.0
    )
    assert stats["clip_low_fraction"] == 0.0
    assert stats["clip_high_fraction"] == 0.0
    assert stats["clip_fraction"] == 0.0


def test_step_three_clips_high_ratio() -> None:
    logp_new = torch.log(
        torch.tensor([[0.9, 0.39], [0.9, 0.51]], dtype=torch.float64)
    )

    loss, stats = grpo_loss(
        logp_new, OLD_PROBS.log(), ADVANTAGES, FULL_MASK
    )

    torch.testing.assert_close(
        loss, torch.tensor(-0.0875, dtype=torch.float64), atol=1e-7, rtol=0.0
    )
    assert stats["clip_high_fraction"] == pytest.approx(0.25)
    assert stats["clip_low_fraction"] == 0.0
    assert stats["clip_fraction"] == pytest.approx(0.25)


def test_asymmetric_high_clip_changes_step_three_loss() -> None:
    logp_new = torch.log(
        torch.tensor([[0.9, 0.39], [0.9, 0.51]], dtype=torch.float64)
    )

    loss, _ = grpo_loss(
        logp_new,
        OLD_PROBS.log(),
        ADVANTAGES,
        FULL_MASK,
        eps_high=0.28,
    )

    torch.testing.assert_close(
        loss, torch.tensor(-0.1075, dtype=torch.float64), atol=1e-7, rtol=0.0
    )


def test_ratio_one_has_plain_policy_gradient_and_zero_log_ratio() -> None:
    logp_old = OLD_PROBS.log()
    logp_new = logp_old.clone().detach().requires_grad_(True)

    loss, stats = grpo_loss(logp_new, logp_old, ADVANTAGES, FULL_MASK)
    loss.backward()

    expected_gradient = torch.tensor(
        [[-0.25, -0.25], [0.25, 0.25]], dtype=torch.float64
    )
    torch.testing.assert_close(logp_new.grad, expected_gradient)
    assert stats["max_abs_log_ratio"] == 0.0
    assert stats["ratio_mean"] == pytest.approx(1.0)
    assert stats["ratio_min"] == pytest.approx(1.0)
    assert stats["ratio_max"] == pytest.approx(1.0)


def test_clipped_token_has_zero_gradient() -> None:
    logp_new = torch.log(
        torch.tensor([[0.9, 0.39], [0.9, 0.51]], dtype=torch.float64)
    ).requires_grad_(True)

    loss, _ = grpo_loss(logp_new, OLD_PROBS.log(), ADVANTAGES, FULL_MASK)
    loss.backward()

    assert logp_new.grad is not None
    assert logp_new.grad[0, 1].item() == 0.0
    assert logp_new.grad[1, 1].item() != 0.0


def test_harmful_direction_tokens_are_not_clipped() -> None:
    logp_old = torch.zeros(2, 1, dtype=torch.float64)
    logp_new = torch.log(torch.tensor([[0.7], [1.3]], dtype=torch.float64)).requires_grad_(True)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)

    loss, _ = grpo_loss(logp_new, logp_old, advantages, torch.ones_like(logp_old))
    loss.backward()

    assert logp_new.grad is not None
    assert torch.all(logp_new.grad != 0)


def test_kl_values_mean_and_beta_contribution() -> None:
    policy = torch.tensor([[0.4, 0.2, 0.1]], dtype=torch.float64)
    reference = torch.tensor([[0.2, 0.4, 0.3]], dtype=torch.float64)
    logp_new = policy.log()
    logp_ref = reference.log()
    advantages = torch.zeros(1, dtype=torch.float64)
    mask = torch.ones_like(logp_new)
    expected_k3 = torch.tensor(
        [[0.1931471805599453, 0.3068528194400547, 0.9013877113318902]],
        dtype=torch.float64,
    )

    loss, stats = grpo_loss(
        logp_new,
        logp_new,
        advantages,
        mask,
        beta=0.04,
        logp_ref=logp_ref,
    )
    loss_without_kl, zero_beta_stats = grpo_loss(
        logp_new,
        logp_new,
        advantages,
        mask,
        beta=0.0,
        logp_ref=logp_ref,
    )

    torch.testing.assert_close(loss, 0.04 * expected_k3.mean(), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(loss_without_kl, torch.tensor(0.0, dtype=torch.float64))
    assert stats["kl_mean"] == pytest.approx(expected_k3.mean().item())
    assert zero_beta_stats["kl_mean"] == pytest.approx(expected_k3.mean().item())


def test_unequal_lengths_distinguish_sample_and_token_aggregation() -> None:
    ratios = torch.tensor([[1.1, 1.0, 1.0], [0.9, 0.8, 1.0]], dtype=torch.float64)
    logp_old = torch.zeros_like(ratios)
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.float64)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)

    sample_loss, _ = grpo_loss(
        ratios.log(), logp_old, advantages, mask, aggregation="sample"
    )
    token_loss, _ = grpo_loss(
        ratios.log(), logp_old, advantages, mask, aggregation="token"
    )

    torch.testing.assert_close(sample_loss, torch.tensor(-0.1, dtype=torch.float64))
    torch.testing.assert_close(token_loss, torch.tensor(0.4, dtype=torch.float64))


@pytest.mark.parametrize("aggregation", ["sample", "token"])
def test_accumulated_microbatches_equal_full_batch_values_and_gradients(
    aggregation: str,
) -> None:
    logp_old = torch.tensor(
        [[-0.8, -1.0, -1.2], [-1.1, -0.7, -0.9],
         [-0.6, -1.3, -0.8], [-1.2, -0.9, -0.7]],
        dtype=torch.float64,
    )
    deltas = torch.tensor(
        [[0.05, 0.30, -0.02], [-0.25, 0.04, 0.10],
         [0.12, -0.08, 0.01], [-0.03, 0.18, -0.35]],
        dtype=torch.float64,
    )
    advantages = torch.tensor([1.2, -0.7, 0.4, -1.1], dtype=torch.float64)
    mask = torch.tensor(
        [[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=torch.float64
    )
    denominator = float(len(mask) if aggregation == "sample" else mask.sum())
    full_new = (logp_old + deltas).detach().requires_grad_(True)
    micro_new = (logp_old + deltas).detach().requires_grad_(True)

    full_loss, _ = grpo_loss(
        full_new,
        logp_old,
        advantages,
        mask,
        aggregation=aggregation,
        denominator=denominator,
    )
    full_loss.backward()
    micro_losses = []
    for start in (0, 2):
        micro_loss, _ = grpo_loss(
            micro_new[start : start + 2],
            logp_old[start : start + 2],
            advantages[start : start + 2],
            mask[start : start + 2],
            aggregation=aggregation,
            denominator=denominator,
        )
        micro_losses.append(micro_loss)
    accumulated_loss = sum(micro_losses)
    accumulated_loss.backward()

    torch.testing.assert_close(accumulated_loss, full_loss)
    torch.testing.assert_close(micro_new.grad, full_new.grad)


@pytest.mark.parametrize("aggregation", ["sample", "token"])
def test_random_float64_loss_matches_reference(aggregation: str) -> None:
    generator = torch.Generator().manual_seed(41)
    logp_old = torch.randn(5, 4, dtype=torch.float64, generator=generator)
    logp_new = logp_old + 0.3 * torch.randn(
        5, 4, dtype=torch.float64, generator=generator
    )
    logp_ref = logp_old + 0.2 * torch.randn(
        5, 4, dtype=torch.float64, generator=generator
    )
    advantages = torch.randn(5, dtype=torch.float64, generator=generator)
    mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0],
         [1, 0, 0, 0], [1, 1, 1, 0]],
        dtype=torch.float64,
    )
    kwargs = {
        "eps_low": 0.17,
        "eps_high": 0.24,
        "beta": 0.03,
        "logp_ref": logp_ref,
        "aggregation": aggregation,
    }

    actual, _ = grpo_loss(logp_new, logp_old, advantages, mask, **kwargs)
    expected = _loss_reference(
        logp_new, logp_old, advantages, mask, **kwargs
    )

    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_positive_beta_requires_reference_log_probs() -> None:
    with pytest.raises(ValueError):
        grpo_loss(
            OLD_PROBS.log(),
            OLD_PROBS.log(),
            ADVANTAGES,
            FULL_MASK,
            beta=0.04,
        )


def test_invalid_aggregation_is_rejected() -> None:
    with pytest.raises(ValueError):
        grpo_loss(
            OLD_PROBS.log(),
            OLD_PROBS.log(),
            ADVANTAGES,
            FULL_MASK,
            aggregation="x",
        )


@pytest.mark.parametrize(
    ("logp_old", "advantages", "mask", "logp_ref"),
    [
        pytest.param(torch.zeros(2, 3), torch.zeros(2), torch.ones(2, 2), None),
        pytest.param(torch.zeros(2, 2), torch.zeros(3), torch.ones(2, 2), None),
        pytest.param(torch.zeros(2, 2), torch.zeros(2), torch.ones(2, 3), None),
        pytest.param(torch.zeros(2, 2), torch.zeros(2), torch.ones(2, 2), torch.zeros(1, 2)),
    ],
)
def test_shape_mismatches_are_rejected(
    logp_old: Tensor,
    advantages: Tensor,
    mask: Tensor,
    logp_ref: Tensor | None,
) -> None:
    with pytest.raises(ValueError):
        grpo_loss(
            torch.zeros(2, 2),
            logp_old,
            advantages,
            mask,
            logp_ref=logp_ref,
        )


def test_kl_mean_is_none_without_reference() -> None:
    _, stats = grpo_loss(OLD_PROBS.log(), OLD_PROBS.log(), ADVANTAGES, FULL_MASK)

    assert stats["kl_mean"] is None
