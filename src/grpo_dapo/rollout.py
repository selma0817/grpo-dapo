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
from grpo_dapo.reward import (
    compute_reward,
    extract_answer,
    overlong_penalty,
    parse_number,
)


@dataclass
class MicroBatch:
    """One right-padded scoring and optimization micro-batch."""

    input_ids: Tensor
    attention_mask: Tensor
    completion_mask: Tensor
    advantages: Tensor
    row_ids: Tensor
    loss_mask: Tensor | None = None
    logp_old: Tensor | None = None
    logp_ref: Tensor | None = None
    entropy: Tensor | None = None

    def __post_init__(self) -> None:
        """Default the optimization mask to the full completion mask."""
        if self.loss_mask is None:
            self.loss_mask = self.completion_mask.clone()


@dataclass
class GroupBatch:
    """Generated completion groups and their reward components."""

    examples: Sequence[Example]
    completions: list[list[Completion]]
    correct: Tensor
    boxed: Tensor
    penalty: Tensor
    truncated: Tensor
    rewards: Tensor

    def select(self, indices: Sequence[int] | Tensor) -> "GroupBatch":
        """Return groups at ``indices``, preserving the requested order."""
        if isinstance(indices, Tensor):
            if indices.dtype == torch.bool:
                indices = indices.nonzero(as_tuple=False).flatten()
            index_list = [int(index) for index in indices.flatten().tolist()]
        else:
            index_list = [int(index) for index in indices]
        tensor_indices = torch.tensor(index_list, dtype=torch.long)
        return GroupBatch(
            examples=[self.examples[index] for index in index_list],
            completions=[self.completions[index] for index in index_list],
            correct=self.correct[tensor_indices],
            boxed=self.boxed[tensor_indices],
            penalty=self.penalty[tensor_indices],
            truncated=self.truncated[tensor_indices],
            rewards=self.rewards[tensor_indices],
        )

    @classmethod
    def concat(cls, batches: Sequence["GroupBatch"]) -> "GroupBatch":
        """Concatenate generated batches in generation order."""
        if not batches:
            raise ValueError("at least one group batch is required")
        return cls(
            examples=[example for batch in batches for example in batch.examples],
            completions=[group for batch in batches for group in batch.completions],
            correct=torch.cat([batch.correct for batch in batches]),
            boxed=torch.cat([batch.boxed for batch in batches]),
            penalty=torch.cat([batch.penalty for batch in batches]),
            truncated=torch.cat([batch.truncated for batch in batches]),
            rewards=torch.cat([batch.rewards for batch in batches]),
        )


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
    loss_rows: Tensor | None = None,
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
    if loss_rows is None:
        loss_rows = torch.ones((num_questions, num_samples), dtype=torch.bool)
    elif loss_rows.shape != advantages.shape:
        raise ValueError("loss_rows must have shape [Q, G]")
    else:
        loss_rows = loss_rows.to(dtype=torch.bool, device="cpu")

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

        batch_completion_mask = completion_mask(
            torch.tensor(prompt_lengths),
            torch.tensor(completion_lengths),
            sequence_length,
        )
        row_loss_flags = torch.tensor(
            [
                bool(loss_rows[divmod(row_id, num_samples)])
                for row_id in row_ids
            ],
            dtype=batch_completion_mask.dtype,
        ).unsqueeze(1)
        micro_batches.append(
            MicroBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                completion_mask=batch_completion_mask,
                advantages=torch.stack(row_advantages),
                row_ids=torch.tensor(row_ids, dtype=torch.long),
                loss_mask=batch_completion_mask * row_loss_flags,
            )
        )
    return micro_batches


def _move_micro_batch(micro_batch: MicroBatch, device: torch.device) -> None:
    for name in (
        "input_ids",
        "attention_mask",
        "completion_mask",
        "loss_mask",
        "advantages",
        "row_ids",
    ):
        value = getattr(micro_batch, name)
        if value is not None:
            setattr(micro_batch, name, value.to(device))


def generate_groups(
    policy: Any,
    examples: Sequence[Example],
    config: Config,
) -> GroupBatch:
    """Generate completion groups and compute every reward component."""
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

    correct = torch.tensor(
        [
            [compute_reward(completion.text, example.gold) for completion in group]
            for example, group in zip(examples, completions, strict=True)
        ],
        dtype=torch.float32,
    )
    extracted = [
        [extract_answer(completion.text) for completion in group]
        for group in completions
    ]
    boxed = torch.tensor(
        [[answer is not None for answer in group] for group in extracted],
        dtype=torch.float32,
    )
    prompt_ids = [group[0].prompt_token_ids for group in completions]
    longest_prompt = max(len(ids) for ids in prompt_ids)
    if longest_prompt > config.max_prompt_tokens:
        raise ValueError(
            f"prompt has {longest_prompt} tokens, more than max_prompt_tokens="
            f"{config.max_prompt_tokens}; refusing to train on a truncated question"
        )

    penalty = torch.tensor(
        [
            [
                overlong_penalty(
                    completion.num_tokens,
                    completion.truncated,
                    config.max_new_tokens,
                    config.overlong_cache,
                )
                for completion in group
            ]
            for group in completions
        ],
        dtype=torch.float32,
    )
    truncated = torch.tensor(
        [
            [completion.truncated for completion in group]
            for group in completions
        ],
        dtype=torch.bool,
    )
    # A complete box earns format_weight even when its answer is wrong.
    rewards = (
        correct
        + config.format_weight * boxed
        + config.overlong_penalty_factor * penalty
    )
    return GroupBatch(
        examples=examples,
        completions=completions,
        correct=correct,
        boxed=boxed,
        penalty=penalty,
        truncated=truncated,
        rewards=rewards,
    )


def quality_stats(groups: GroupBatch) -> dict[str, float]:
    """Compute quality and generation-cost statistics for a group batch."""
    flat_completions = [
        completion for group in groups.completions for completion in group
    ]
    flat_extracted = [
        extract_answer(completion.text) for completion in flat_completions
    ]
    correct_counts = groups.correct.sum(dim=1)
    num_samples = groups.correct.shape[1]
    zero_variance = zero_variance_groups(groups.rewards)
    lengths = [completion.num_tokens for completion in flat_completions]
    return {
        "reward_mean": float(groups.rewards.mean().item()),
        "accuracy": float(groups.correct.mean().item()),
        "format_rate": float(groups.boxed.mean().item()),
        "unparseable_rate": sum(
            answer is not None and parse_number(answer) is None
            for answer in flat_extracted
        )
        / len(flat_completions),
        "overlong_penalty_mean": float(groups.penalty.mean().item()),
        "all_correct": float(
            (correct_counts == num_samples).float().mean().item()
        ),
        "all_wrong": float((correct_counts == 0).float().mean().item()),
        "mixed": float(
            ((correct_counts > 0) & (correct_counts < num_samples))
            .float()
            .mean()
            .item()
        ),
        "nonzero_variance": float((~zero_variance).float().mean().item()),
        "mean_completion_length": statistics.fmean(lengths),
        "max_completion_length": float(max(lengths)),
        "truncation_rate": float(groups.truncated.float().mean().item()),
        "tokens_generated": float(sum(lengths)),
    }


def build_rollout(
    policy: Any,
    groups: GroupBatch,
    config: Config,
    generator: torch.Generator,
    padding: Tensor | None = None,
) -> RolloutBatch:
    """Build, score, and summarize a fixed training rollout."""
    advantages = group_advantages(groups.rewards)
    zero_variance = zero_variance_groups(groups.rewards)
    num_questions = groups.rewards.shape[0]
    if padding is not None:
        if padding.shape != (num_questions,):
            raise ValueError("padding must have shape [Q]")
        advantages = advantages.clone()
        advantages[padding.to(dtype=torch.bool, device="cpu")] = 0

    prompt_ids = [group[0].prompt_token_ids for group in groups.completions]
    completion_ids = [
        [completion.token_ids for completion in group]
        for group in groups.completions
    ]
    loss_rows = ~groups.truncated if config.overlong_filter else None
    micro_batches = build_micro_batches(
        prompt_ids,
        completion_ids,
        advantages,
        config.micro_batch_size,
        policy.tokenizer.pad_token_id,
        generator,
        loss_rows=loss_rows,
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

    entropy_sum = sum(
        float((micro_batch.entropy * micro_batch.completion_mask).sum().item())
        for micro_batch in micro_batches
        if micro_batch.entropy is not None
    )
    completion_tokens = sum(
        float(micro_batch.completion_mask.sum().item())
        for micro_batch in micro_batches
    )
    stats = quality_stats(groups)
    stats.update(
        {
            "mean_entropy": entropy_sum / completion_tokens,
            "score_seconds": score_seconds,
        }
    )
    return RolloutBatch(
        examples=groups.examples,
        completions=groups.completions,
        rewards=groups.rewards,
        advantages=advantages,
        zero_variance=zero_variance,
        micro_batches=micro_batches,
        stats=stats,
    )


def collect_rollout(
    policy: Any,
    examples: Sequence[Example],
    config: Config,
    generator: torch.Generator,
) -> RolloutBatch:
    """Generate, reward, partition, and score one training rollout."""
    return build_rollout(
        policy,
        generate_groups(policy, examples, config),
        config,
        generator,
    )
