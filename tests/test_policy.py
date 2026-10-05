"""Offline tiny-Qwen tests for LoRA policy scoring and reference behavior."""

import copy

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grpo_dapo.config import Config
from grpo_dapo.logprobs import token_log_probs
from grpo_dapo.policy import Policy
from grpo_dapo.rollout import MicroBatch


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def save_pretrained(self, path) -> None:
        (path / "tokenizer.txt").write_text("tiny", encoding="utf-8")


def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(5)
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    )


def tiny_config(**overrides) -> Config:
    values = {
        "gradient_checkpointing": False,
        "questions_per_step": 2,
        "samples_per_question": 2,
        "num_minibatches": 2,
        "micro_batch_size": 1,
        "wandb_mode": "disabled",
    }
    values.update(overrides)
    return Config(**values)


def micro_batch() -> MicroBatch:
    input_ids = torch.tensor([[3, 4, 5, 2], [6, 7, 2, 0]])
    return MicroBatch(
        input_ids=input_ids,
        attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        completion_mask=torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.float32),
        advantages=torch.tensor([1.0, -1.0]),
        row_ids=torch.tensor([0, 1]),
    )


def test_scoring_is_repeatable_and_train_eval_invariant() -> None:
    policy = Policy.from_model(tiny_model(), TinyTokenizer(), tiny_config())
    batch = micro_batch()

    first = policy.score(batch, with_grad=False).logp
    second = policy.score(batch, with_grad=False).logp
    training = policy.score(batch, with_grad=True).logp

    assert torch.max(torch.abs(first - second)).item() == 0.0
    torch.testing.assert_close(first, training)


def test_adapter_disabled_matches_base_and_differs_from_adapter() -> None:
    model = tiny_model()
    base = copy.deepcopy(model).eval()
    policy = Policy.from_model(model, TinyTokenizer(), tiny_config())
    for name, parameter in policy.model.named_parameters():
        if "lora_B" in name:
            parameter.data.fill_(0.05)
    batch = micro_batch()

    adapter = policy.score(batch, with_grad=False, use_adapter=True).logp
    reference = policy.score(batch, with_grad=False, use_adapter=False).logp
    with torch.no_grad():
        base_logp = token_log_probs(
            base(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
            ).logits,
            batch.input_ids,
        )

    torch.testing.assert_close(reference, base_logp)
    assert not torch.equal(adapter, reference)


def test_only_lora_parameters_are_trainable() -> None:
    policy = Policy.from_model(tiny_model(), TinyTokenizer(), tiny_config())
    trainable_names = [
        name for name, parameter in policy.model.named_parameters() if parameter.requires_grad
    ]

    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
