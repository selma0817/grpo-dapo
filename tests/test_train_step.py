"""Executable contract tests for the intentionally hand-written train step."""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grpo_dapo.config import Config
from grpo_dapo.policy import Policy
from grpo_dapo.rollout import MicroBatch, RolloutBatch
from grpo_dapo.trainer import train_step


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2


class _CountingAdamW(torch.optim.AdamW):
    def __init__(self, parameters) -> None:
        super().__init__(parameters, lr=1e-3)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


def _setup():
    config = Config(
        gradient_checkpointing=False,
        questions_per_step=1,
        samples_per_question=2,
        num_minibatches=1,
        micro_batch_size=2,
        wandb_mode="disabled",
    )
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    )
    policy = Policy.from_model(model, _Tokenizer(), config)
    input_ids = torch.tensor([[3, 4, 5, 2], [3, 4, 6, 2]])
    attention_mask = torch.ones_like(input_ids)
    batch = MicroBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.float32),
        advantages=torch.tensor([1.0, -1.0]),
        row_ids=torch.tensor([0, 1]),
    )
    batch.logp_old = policy.score(batch, with_grad=False).logp
    batch.logp_ref = policy.score(batch, with_grad=False, use_adapter=False).logp
    rollout = RolloutBatch(
        examples=[],
        completions=[],
        rewards=torch.tensor([[1.0, 0.0]]),
        advantages=torch.tensor([[1.0, -1.0]]),
        zero_variance=torch.tensor([False]),
        micro_batches=[batch],
        stats={},
    )
    optimizer = _CountingAdamW(policy.trainable_parameters())
    return policy, optimizer, rollout, config


def test_train_step_returns_metrics_and_steps_each_minibatch() -> None:
    policy, optimizer, rollout, config = _setup()

    stats = train_step(policy, optimizer, rollout, config)

    assert {
        "loss",
        "grad_norm",
        "clip_low",
        "clip_high",
        "clip_fraction",
        "ratio_mean",
        "ratio_min",
        "ratio_max",
        "kl_mean",
        "first_update_max_abs_log_ratio",
        "first_update_clip_fraction",
    } <= set(stats)
    assert stats["first_update_max_abs_log_ratio"] <= 1e-7
    assert optimizer.step_count == config.num_minibatches
    assert torch.isfinite(torch.tensor(stats["loss"]))


def test_train_step_changes_lora_but_not_base_weights() -> None:
    policy, optimizer, rollout, config = _setup()
    base_before = {
        name: parameter.detach().clone()
        for name, parameter in policy.model.named_parameters()
        if "lora_" not in name
    }
    lora_before = {
        name: parameter.detach().clone()
        for name, parameter in policy.model.named_parameters()
        if "lora_" in name
    }

    train_step(policy, optimizer, rollout, config)

    for name, parameter in policy.model.named_parameters():
        if name in base_before:
            assert torch.equal(parameter, base_before[name])
    assert any(
        not torch.equal(parameter, lora_before[name])
        for name, parameter in policy.model.named_parameters()
        if name in lora_before
    )


def _tiny_policy(config: Config, seed: int = 0) -> Policy:
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    )
    return Policy.from_model(model, _Tokenizer(), config)


def _scored_micro_batch(policy: Policy, rows: list[list[int]], advantages: list[float]) -> MicroBatch:
    input_ids = torch.tensor(rows)
    batch = MicroBatch(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        completion_mask=torch.tensor([[0, 1, 1]] * len(rows), dtype=torch.float32),
        advantages=torch.tensor(advantages),
        row_ids=torch.arange(len(rows)),
    )
    batch.logp_old = policy.score(batch, with_grad=False).logp
    batch.logp_ref = policy.score(batch, with_grad=False, use_adapter=False).logp
    return batch


def _rollout(micro_batches: list[MicroBatch]) -> RolloutBatch:
    return RolloutBatch(
        examples=[],
        completions=[],
        rewards=torch.zeros(1, 1),
        advantages=torch.zeros(1, 1),
        zero_variance=torch.tensor([False]),
        micro_batches=micro_batches,
        stats={},
    )


ROWS = [[3, 4, 5, 2], [3, 4, 6, 2], [3, 7, 5, 2], [3, 7, 6, 2]]
ADVANTAGES = [1.0, -1.0, 0.5, -0.5]


def test_train_step_steps_once_per_minibatch_not_per_microbatch() -> None:
    # 4 micro-batches of 1 row, 2 mini-batches -> exactly 2 optimizer steps.
    config = Config(
        gradient_checkpointing=False,
        questions_per_step=2,
        samples_per_question=2,
        num_minibatches=2,
        micro_batch_size=1,
        wandb_mode="disabled",
    )
    policy = _tiny_policy(config)
    micro_batches = [
        _scored_micro_batch(policy, [row], [advantage])
        for row, advantage in zip(ROWS, ADVANTAGES)
    ]
    optimizer = _CountingAdamW(policy.trainable_parameters())

    train_step(policy, optimizer, _rollout(micro_batches), config)

    assert optimizer.step_count == 2


@pytest.mark.parametrize("aggregation", ["sample", "token"])
def test_microbatch_accumulation_matches_one_big_microbatch(aggregation: str) -> None:
    # Same 2 rows as one micro-batch or as two: one update must give the same weights.
    # SGD (not AdamW) so a wrong denominator shows up as a scaled update.
    updated = []
    for micro_batch_size in (2, 1):
        config = Config(
            gradient_checkpointing=False,
            questions_per_step=1,
            samples_per_question=2,
            num_minibatches=1,
            micro_batch_size=micro_batch_size,
            aggregation=aggregation,
            max_grad_norm=1e9,
            wandb_mode="disabled",
        )
        policy = _tiny_policy(config, seed=0)
        if micro_batch_size == 2:
            micro_batches = [_scored_micro_batch(policy, ROWS[:2], ADVANTAGES[:2])]
        else:
            micro_batches = [
                _scored_micro_batch(policy, [row], [advantage])
                for row, advantage in zip(ROWS[:2], ADVANTAGES[:2])
            ]
        optimizer = torch.optim.SGD(policy.trainable_parameters(), lr=1.0)
        train_step(policy, optimizer, _rollout(micro_batches), config)
        updated.append(
            {
                name: parameter.detach().clone()
                for name, parameter in policy.model.named_parameters()
                if "lora_" in name
            }
        )

    for name, value in updated[0].items():
        torch.testing.assert_close(updated[1][name], value, atol=1e-6, rtol=1e-5)
