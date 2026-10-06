"""Offline tests for generation token accounting."""

from types import SimpleNamespace

import pytest
import torch

from grpo_dapo.generation import completion_length, generate, is_truncated


@pytest.mark.parametrize(
    ("token_ids", "expected_length", "expected_truncated"),
    [
        # Stop at the first EOS; trailing padding does not count.
        pytest.param([7, 8, 2, 2, 2], 3, False),
        # Using every token without EOS is truncation.
        pytest.param([7, 8, 9, 10, 11], 5, True),
        # EOS on the final allowed token is a normal stop.
        pytest.param([7, 8, 9, 10, 2], 5, False),
        # Immediate EOS is an empty one-token answer.
        pytest.param([2, 2, 2, 2, 2], 1, False),
    ],
)
def test_completion_accounting(
    token_ids: list[int],
    expected_length: int,
    expected_truncated: bool,
) -> None:
    assert completion_length(token_ids, {2}) == expected_length
    assert is_truncated(token_ids, {2}, max_new_tokens=5) is expected_truncated


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def __call__(self, texts, **kwargs):
        del texts, kwargs
        return {
            "input_ids": torch.tensor([[0, 11, 12], [21, 22, 23]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        }

    def decode(self, token_ids, **kwargs):
        del kwargs
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeModel:
    device = torch.device("cpu")
    generation_config = SimpleNamespace(eos_token_id=2)

    def __init__(self) -> None:
        self.kwargs = None

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask
        self.kwargs = kwargs
        prefixes = input_ids.repeat_interleave(2, dim=0)
        suffixes = torch.tensor(
            [[31, 2, 0], [32, 2, 0], [41, 2, 0], [42, 2, 0]]
        )
        return torch.cat((prefixes, suffixes), dim=1)


def test_generate_preserves_token_ids_and_returns_exact_kwargs() -> None:
    model = _FakeModel()
    groups, settings = generate(
        model,
        _FakeTokenizer(),
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
        num_samples=2,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=3,
        batch_size=2,
    )

    assert settings == model.kwargs
    assert groups[0][0].prompt_token_ids == (11, 12)
    assert groups[1][0].prompt_token_ids == (21, 22, 23)
    assert groups[0][0].token_ids == (31, 2)
    assert groups[0][0].num_tokens == 2
    assert groups[0][0].truncated is False
