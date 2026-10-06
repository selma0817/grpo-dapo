"""Tests for shared evaluation scoring and summaries."""

import subprocess
import sys
from fractions import Fraction

from grpo_dapo.data import Example
from grpo_dapo.evaluation import (
    score_completions,
    summarize_greedy,
    summarize_sampled,
)
from grpo_dapo.generation import Completion
from grpo_dapo.metrics import summarize


def _examples() -> list[Example]:
    return [
        Example(0, "one?", "1", Fraction(1)),
        Example(1, "two?", "2", Fraction(2)),
    ]


def _completion(text: str, length: int = 2) -> Completion:
    return Completion(text, length, False, (5, 2), (3, 4))


def test_summaries_reproduce_baseline_helpers() -> None:
    examples = _examples()
    groups = [
        [_completion(r"\boxed{1}"), _completion(r"\boxed{0}")],
        [_completion(r"\boxed{2}"), _completion("no box")],
    ]
    records = score_completions(examples, groups)

    assert summarize_greedy(records[:1]) == summarize(records[:1])
    sampled = summarize_sampled(records, examples, num_samples=2)
    assert sampled["accuracy"] == 0.5
    assert sampled["pass_at_k"] == {"1": 0.5, "2": 1.0}
    assert sampled["group_stats"]["mixed_rate"] == 1.0


def test_eval_baseline_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/eval_baseline.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--num-samples" in result.stdout


def test_evaluate_generates_in_eval_mode_and_restores_mode(monkeypatch) -> None:
    # In train mode with gradient checkpointing, Hugging Face disables the KV cache.
    import torch

    import grpo_dapo.evaluation as evaluation

    model = torch.nn.Linear(2, 2)
    model.train()
    modes_during_generation = []

    def fake_generate(model, tokenizer, prompts, *, num_samples, **kwargs):
        modes_during_generation.append(model.training)
        groups = [
            [_completion(r"\boxed{1}") for _ in range(num_samples)] for _ in prompts
        ]
        return groups, {"num_return_sequences": num_samples}

    monkeypatch.setattr(evaluation, "generate", fake_generate)
    evaluation.evaluate(
        model, None, _examples(), num_samples=2, max_new_tokens=8, batch_size=2
    )

    assert modes_during_generation == [False, False]
    assert model.training is True
