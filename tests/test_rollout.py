"""Tests for deterministic right-padded rollout partitions."""

import torch

from grpo_dapo.rollout import build_micro_batches


def _partition(seed: int):
    return build_micro_batches(
        prompt_ids=[[10, 11], [20, 21, 22]],
        completion_ids=[[[30, 2], [31, 32, 2]], [[40, 2], [41, 42, 43, 2]]],
        advantages=torch.tensor([[0.5, -0.5], [1.0, -1.0]]),
        micro_batch_size=2,
        pad_token_id=0,
        generator=torch.Generator().manual_seed(seed),
    )


def test_build_micro_batches_padding_masks_rows_and_reproducibility() -> None:
    first = _partition(17)
    second = _partition(17)
    first_ids = torch.cat([batch.row_ids.cpu() for batch in first]).tolist()
    second_ids = torch.cat([batch.row_ids.cpu() for batch in second]).tolist()

    assert first_ids == second_ids
    assert sorted(first_ids) == [0, 1, 2, 3]

    prompts = [[10, 11], [20, 21, 22]]
    completions = [[[30, 2], [31, 32, 2]], [[40, 2], [41, 42, 43, 2]]]
    for batch in first:
        for row_index, row_id in enumerate(batch.row_ids.tolist()):
            question, sample = divmod(row_id, 2)
            expected = prompts[question] + completions[question][sample]
            length = len(expected)
            assert batch.input_ids[row_index, :length].tolist() == expected
            assert torch.all(batch.input_ids[row_index, length:] == 0)
            assert batch.attention_mask[row_index].sum().item() == length
            assert torch.all(batch.attention_mask[row_index, length:] == 0)
            assert batch.completion_mask[row_index].sum().item() == len(
                completions[question][sample]
            )
            start = len(prompts[question]) - 1
            stop = start + len(completions[question][sample])
            assert torch.all(batch.completion_mask[row_index, start:stop] == 1)


def test_collect_rollout_rejects_prompts_longer_than_the_limit() -> None:
    from fractions import Fraction

    import pytest

    from grpo_dapo.config import Config
    from grpo_dapo.data import Example
    from grpo_dapo.generation import Completion
    from grpo_dapo.rollout import collect_rollout

    class _LongPromptPolicy:
        class tokenizer:
            pad_token_id = 0

        def generate(self, prompts, *, num_samples, **kwargs):
            return [
                [
                    Completion(r"\boxed{1}", 2, False, (5, 2), (3, 4, 5, 6))
                    for _ in range(num_samples)
                ]
                for _ in prompts
            ]

        def score(self, *args, **kwargs):
            raise AssertionError("must fail before scoring")

    config = Config(
        questions_per_step=1,
        samples_per_question=2,
        num_minibatches=1,
        micro_batch_size=2,
        max_prompt_tokens=3,
    )
    examples = [Example(0, "q", "1", Fraction(1))]

    with pytest.raises(ValueError, match="max_prompt_tokens"):
        collect_rollout(_LongPromptPolicy(), examples, config, torch.Generator())
