"""Clipped GRPO and DAPO policy losses."""

import torch
from torch import Tensor


def grpo_loss(
    logp_new: Tensor,
    logp_old: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    beta: float = 0.0,
    logp_ref: Tensor | None = None,
    aggregation: str = "sample",
    denominator: float | None = None,
) -> tuple[Tensor, dict[str, float | None]]:
    """Return the clipped policy loss and masked-token diagnostics."""
    if logp_ref is None and beta > 0:
        raise ValueError("Cannot use KL penalty without reference log-probabilities")
    if aggregation not in ("sample", "token"):
        raise ValueError(f"Invalid aggregation mode: {aggregation}")

    if logp_new.ndim != 2:
        raise ValueError("logp_new must have shape [B, T]")
    batch_size = logp_new.shape[0]
    if logp_old.shape != logp_new.shape or mask.shape != logp_new.shape:
        raise ValueError("logp_new, logp_old, and mask must have matching shapes")
    if advantages.shape != (batch_size,):
        raise ValueError("advantages must have shape [B]")
    if logp_ref is not None and logp_ref.shape != logp_new.shape:
        raise ValueError("logp_ref must match the shape of logp_new")

    advantages_per_token = advantages.unsqueeze(-1)
    log_ratio = logp_new - logp_old
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages_per_token
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * advantages_per_token
    loss_per_token = -torch.minimum(unclipped, clipped)

    k3: Tensor | None = None
    if logp_ref is not None:
        difference = logp_ref - logp_new
        k3 = torch.exp(difference) - difference - 1
        if beta > 0:
            loss_per_token = loss_per_token + beta * k3

    if aggregation == "sample":
        sample_denominator = denominator if denominator is not None else batch_size
        token_counts = mask.sum(dim=1).clamp(min=1)
        per_sample_loss = (loss_per_token * mask).sum(dim=1) / token_counts
        loss = per_sample_loss.sum() / sample_denominator
    else:
        token_denominator = denominator if denominator is not None else mask.sum()
        loss = (loss_per_token * mask).sum() / token_denominator

    with torch.no_grad():
        active = mask.bool()
        masked_ratio = ratio[active]
        masked_log_ratio = log_ratio[active]
        low_clipped = (
            (advantages_per_token < 0) & (ratio < 1 - eps_low)
        )[active]
        high_clipped = (
            (advantages_per_token > 0) & (ratio > 1 + eps_high)
        )[active]

        clip_low_fraction = low_clipped.float().mean().item()
        clip_high_fraction = high_clipped.float().mean().item()
        stats: dict[str, float | None] = {
            "clip_low_fraction": clip_low_fraction,
            "clip_high_fraction": clip_high_fraction,
            "clip_fraction": clip_low_fraction + clip_high_fraction,
            "ratio_mean": masked_ratio.mean().item(),
            "ratio_min": masked_ratio.min().item(),
            "ratio_max": masked_ratio.max().item(),
            "max_abs_log_ratio": masked_log_ratio.abs().max().item(),
            "kl_mean": k3[active].mean().item() if k3 is not None else None,
        }

    return loss, stats
