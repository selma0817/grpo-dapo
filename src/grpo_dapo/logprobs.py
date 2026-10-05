"""Token-level probability utilities."""
import torch
from torch import Tensor

def token_log_probs(logits: Tensor, input_ids: Tensor) -> Tensor:
    """Return next-token log-probabilities aligned to shifted input IDs."""
    #[B, T, V] is shape of logits
    # [B, T] is input_ids shape

    # output should have shape [B, T-1]
    relevant_logits = logits[:, :-1, :].to(torch.float32)  # [B, T-1, V]
    next_token_ids = input_ids[:, 1:].unsqueeze(-1)
    gathered_logits = torch.gather(relevant_logits, dim=2, index=next_token_ids).squeeze(-1)  # [B, T-1]
    return gathered_logits - torch.logsumexp(relevant_logits, dim=-1)


def token_entropy(logits: Tensor) -> Tensor:
    """Return entropy for each next-token distribution."""
    # logits [B, T, V]
    relevant_logits = logits[:, :-1, :].to(torch.float32)  # [B, T-1, V]
    entropy = torch.logsumexp(relevant_logits, dim=-1) - torch.sum(torch.softmax(relevant_logits, dim=-1) * relevant_logits, dim=-1)
    return entropy

def completion_mask(
    prompt_lengths: Tensor,
    completion_lengths: Tensor,
    seq_len: int,
) -> Tensor:
    """Return a shifted float mask selecting completion tokens."""
    starts = prompt_lengths.unsqueeze(1)
    ends = starts + completion_lengths.unsqueeze(1)
    positions = torch.arange(1, seq_len, device=prompt_lengths.device).unsqueeze(0)
    mask = (positions >= starts) & (positions < ends)
    return mask.to(torch.float32)
