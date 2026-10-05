"""Rollout collection and deterministic micro-batch construction."""

import statistics
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from grpo_dapo.advantage import group_advantages, zero_variance_groups
from grpo_dapo.config import Config
from grpo_dapo.data import Example
from grpo_dapo.generation import Completion
from grpo_dapo.logprobs import completion_mask
from grpo_dapo.prompts import build_messages
from grpo_dapo.reward import compute_reward, extract_answer, parse_number


@dataclass
class MicroBatch:
    """One right-padded scoring and optimization micro-batch."""

    input_ids: Tensor
    attention_mask: Tensor
    completion_mask: Tensor
    advantages: Tensor
    row_ids: Tensor
    logp_old: Tensor | None = None
    logp_ref: Tensor | None = None
    entropy: Tensor | None = None


@dataclass
class RolloutBatch:
    """A complete grouped rollout and its fixed training partition."""

    examples: Sequence[Example]
    completions: list[list[Completion]]
    rewards: Tensor
    advantages: Tensor
    zero_variance: Tensor
    micro_batches: list[MicroBatch]
    stats: dict[str, float]


def build_micro_batches(
    prompt_ids: Sequence[Sequence[int]],
    completion_ids: Sequence[Sequence[Sequence[int]]],
    advantages: Tensor,
    micro_batch_size: int,
    pad_token_id: int,
    generator: torch.Generator,
) -> list[MicroBatch]:
    """Shuffle grouped rows once and right-pad each resulting micro-batch."""
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if advantages.ndim != 2:
        raise ValueError("advantages must have shape [Q, G]")
    num_questions, num_samples = advantages.shape
    if len(prompt_ids) != num_questions or len(completion_ids) != num_questions:
        raise ValueError("prompt and completion groups must match advantages")
    if any(len(group) != num_samples for group in completion_ids):
        raise ValueError("every completion group must contain G rows")

    total_rows = num_questions * num_samples
    order = torch.randperm(total_rows, generator=generator).tolist()
    micro_batches: list[MicroBatch] = []
    for start in range(0, total_rows, micro_batch_size):
        row_ids = order[start : start + micro_batch_size]
        sequences: list[list[int]] = []
        prompt_lengths: list[int] = []
        completion_lengths: list[int] = []
        row_advantages: list[Tensor] = []
        for row_id in row_ids:
            question_index, sample_index = divmod(row_id, num_samples)
            prompt = [int(token_id) for token_id in prompt_ids[question_index]]
            completion = [
                int(token_id)
                for token_id in completion_ids[question_index][sample_index]
            ]
            sequences.append(prompt + completion)
            prompt_lengths.append(len(prompt))
            completion_lengths.append(len(completion))
            row_advantages.append(advantages[question_index, sample_index])

        sequence_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), sequence_length), pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row_index, :length] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row_index, :length] = 1

        micro_batches.append(
            MicroBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                completion_mask=completion_mask(
                    torch.tensor(prompt_lengths),
                    torch.tensor(completion_lengths),
                    sequence_length,
                ),
                advantages=torch.stack(row_advantages),
                row_ids=torch.tensor(row_ids, dtype=torch.long),
            )
        )
    return micro_batches


def _move_micro_batch(micro_batch: MicroBatch, device: torch.device) -> None:
    for name in (
        "input_ids",
        "attention_mask",
        "completion_mask",
        "advantages",
        "row_ids",
    ):
        setattr(micro_batch, name, getattr(micro_batch, name).to(device))


def collect_rollout(
    policy: Any,
    examples: Sequence[Example],
    config: Config,
    generator: torch.Generator,
) -> RolloutBatch:
    """Generate, reward, partition, and score one training rollout."""
    prompts = [build_messages(example.question) for example in examples]
    completions = policy.generate(
        prompts,
        num_samples=config.samples_per_question,
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.generation_batch_size,
    )

    rewards = torch.tensor(
        [
            [compute_reward(completion.text, example.gold) for completion in group]
            for example, group in zip(examples, completions, strict=True)
        ],
        dtype=torch.float32,
    )
    advantages = group_advantages(rewards)
    zero_variance = zero_variance_groups(rewards)
    prompt_ids = [group[0].prompt_token_ids for group in completions]
    longest_prompt = max(len(ids) for ids in prompt_ids)
    if longest_prompt > config.max_prompt_tokens:
        raise ValueError(
            f"prompt has {longest_prompt} tokens, more than max_prompt_tokens="
            f"{config.max_prompt_tokens}; refusing to train on a truncated question"
        )
    completion_ids = [
        [completion.token_ids for completion in group] for group in completions
    ]
    micro_batches = build_micro_batches(
        prompt_ids,
        completion_ids,
        advantages,
        config.micro_batch_size,
        policy.tokenizer.pad_token_id,
        generator,
    )

    score_started = time.perf_counter()
    for micro_batch in micro_batches:
        _move_micro_batch(micro_batch, policy.device)
        old_scores = policy.score(
            micro_batch,
            with_grad=False,
            use_adapter=True,
            return_entropy=True,
        )
        reference_scores = policy.score(
            micro_batch,
            with_grad=False,
            use_adapter=False,
        )
        micro_batch.logp_old = old_scores.logp
        micro_batch.logp_ref = reference_scores.logp
        micro_batch.entropy = old_scores.entropy
    score_seconds = time.perf_counter() - score_started

    flat_completions = [completion for group in completions for completion in group]
    extracted = [extract_answer(completion.text) for completion in flat_completions]
    correct_counts = rewards.sum(dim=1)
    entropy_sum = sum(
        float((micro_batch.entropy * micro_batch.completion_mask).sum().item())
        for micro_batch in micro_batches
        if micro_batch.entropy is not None
    )
    completion_tokens = sum(
        float(micro_batch.completion_mask.sum().item())
        for micro_batch in micro_batches
    )
    lengths = [completion.num_tokens for completion in flat_completions]
    stats = {
        "reward_mean": float(rewards.mean().item()),
        "accuracy": float(rewards.mean().item()),
        "format_rate": sum(answer is not None for answer in extracted)
        / len(flat_completions),
        "unparseable_rate": sum(
            answer is not None and parse_number(answer) is None
            for answer in extracted
        )
        / len(flat_completions),
        "all_correct": float(
            (correct_counts == config.samples_per_question).float().mean().item()
        ),
        "all_wrong": float((correct_counts == 0).float().mean().item()),
        "mixed": float(
            (
                (correct_counts > 0)
                & (correct_counts < config.samples_per_question)
            )
            .float()
            .mean()
            .item()
        ),
        "mean_completion_length": statistics.fmean(lengths),
        "max_completion_length": float(max(lengths)),
        "truncation_rate": sum(
            completion.truncated for completion in flat_completions
        )
        / len(flat_completions),
        "mean_entropy": entropy_sum / completion_tokens,
        "score_seconds": score_seconds,
        "tokens_generated": float(sum(lengths)),
    }
    return RolloutBatch(
        examples=examples,
        completions=completions,
        rewards=rewards,
        advantages=advantages,
        zero_variance=zero_variance,
        micro_batches=micro_batches,
        stats=stats,
    )
