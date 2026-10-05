"""Shared completion scoring and model evaluation."""

from collections.abc import Mapping, Sequence
from typing import Any

from grpo_dapo.data import Example
from grpo_dapo.generation import Completion, generate
from grpo_dapo.metrics import group_stats, pass_at_k, summarize
from grpo_dapo.prompts import build_messages
from grpo_dapo.reward import compute_reward, extract_answer, parse_number


def score_completions(
    examples: Sequence[Example],
    groups: Sequence[Sequence[Completion]],
) -> list[dict[str, Any]]:
    """Convert grouped completions into completion-level scoring records."""
    if len(examples) != len(groups):
        raise ValueError("completion groups do not match the number of examples")

    records: list[dict[str, Any]] = []
    for example, completions in zip(examples, groups, strict=True):
        for sample_index, completion in enumerate(completions):
            extracted_answer = extract_answer(completion.text)
            parsed_answer = parse_number(extracted_answer)
            records.append(
                {
                    "question_id": example.id,
                    "sample_index": sample_index,
                    "completion": completion.text,
                    "num_tokens": completion.num_tokens,
                    "truncated": completion.truncated,
                    "extracted_answer": extracted_answer,
                    "parsed_answer": (
                        str(parsed_answer) if parsed_answer is not None else None
                    ),
                    "gold": example.gold,
                    "correct": compute_reward(completion.text, example.gold) == 1.0,
                }
            )
    return records


def summarize_greedy(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one greedy completion per question."""
    return summarize(records)


def summarize_sampled(
    records: Sequence[Mapping[str, Any]],
    examples: Sequence[Example],
    num_samples: int,
) -> dict[str, Any]:
    """Summarize sampled completions with pass-at-k and group statistics."""
    metrics = summarize(records)
    correct_by_question = {example.id: 0 for example in examples}
    for record in records:
        correct_by_question[int(record["question_id"])] += int(record["correct"])
    correct_counts = [correct_by_question[example.id] for example in examples]
    metrics["pass_at_k"] = {
        str(k): (
            sum(pass_at_k(num_samples, count, k) for count in correct_counts)
            / len(correct_counts)
            if correct_counts
            else 0.0
        )
        for k in (1, 2, 4, 8)
        if k <= num_samples
    }
    metrics["group_stats"] = group_stats(correct_counts, num_samples)
    return metrics


def evaluate(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Example],
    *,
    num_samples: int,
    max_new_tokens: int,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run greedy and temperature-one sampled evaluation.

    Generation runs in eval mode: in train mode with gradient checkpointing,
    Hugging Face disables the KV cache, which makes generation far slower.
    The model's previous mode is restored afterwards.
    """
    prompts = [build_messages(example.question) for example in examples]
    was_training = model.training
    model.eval()
    try:
        greedy_groups, greedy_settings = generate(
            model,
            tokenizer,
            prompts,
            num_samples=1,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            desc="Greedy",
        )
        sampled_groups, sampled_settings = generate(
            model,
            tokenizer,
            prompts,
            num_samples=num_samples,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            desc="Sampled",
        )
    finally:
        model.train(was_training)
    greedy_records = score_completions(examples, greedy_groups)
    sampled_records = score_completions(examples, sampled_groups)
    greedy = summarize_greedy(greedy_records)
    sampled = summarize_sampled(sampled_records, examples, num_samples)

    metrics = {
        "eval/greedy_accuracy": float(greedy["accuracy"]),
        "eval/greedy_format_rate": float(greedy["format_rate"]),
        "eval/greedy_truncation_rate": float(greedy["truncation_rate"]),
        "eval/sampled_format_rate": float(sampled["format_rate"]),
        "eval/sampled_truncation_rate": float(sampled["truncation_rate"]),
        "eval/mean_length": float(sampled["response_length"]["mean"]),
    }
    for k, value in sampled["pass_at_k"].items():
        metrics[f"eval/pass@{k}"] = float(value)

    records = {
        "greedy": greedy_records,
        "sampled": sampled_records,
        "generation_settings": {
            "greedy": greedy_settings,
            "sampled": sampled_settings,
        },
    }
    return metrics, records
