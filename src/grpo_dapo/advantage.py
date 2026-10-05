"""Group-relative advantage utilities."""

from torch import Tensor


def group_advantages(rewards: Tensor, eps: float = 1e-4) -> Tensor:
    """Standardize rewards within each prompt group."""
    # rewards shape is [N, G] # N for distinct prompt, G for 
    # number of output generated per prompt
    _, G = rewards.shape
    if G < 2:
        raise ValueError("Cannot compute group advantages with fewer than 2 completions per group")
    row_mean = rewards.mean(dim=1, keepdim=True)
    row_std = rewards.std(dim=1, keepdim=True, correction=1)
    return (rewards - row_mean) / (row_std + eps)


def zero_variance_groups(rewards: Tensor) -> Tensor:
    """Identify reward groups containing no learning signal."""
    first_reward_per_row = rewards[:, 0].unsqueeze(-1)
    comparison = rewards == first_reward_per_row
    return comparison.all(dim=1)