"""Command-line baseline evaluation for untrained GRPO models on GSM8K."""

import argparse
import json
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from grpo_dapo.data import Example, load_gsm8k
from grpo_dapo.generation import Completion, generate, load_model_and_tokenizer
from grpo_dapo.metrics import group_stats, pass_at_k, summarize
from grpo_dapo.prompts import SYSTEM_PROMPT, build_messages
from grpo_dapo.reward import compute_reward, extract_answer, parse_number


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "openai/gsm8k"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _default_output_dir(model: str, split: str) -> Path:
    model_name = model.rsplit("/", maxsplit=1)[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "m0" / f"{model_name}_{split}_{timestamp}"


def _generation_settings(
    *,
    num_samples: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    eos_token_id: int | list[int],
    pad_token_id: int,
) -> dict[str, Any]:
    return {
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_samples,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "batch_size": batch_size,
    }


def _score_completions(
    examples: Sequence[Example],
    grouped_completions: Sequence[Sequence[Completion]],
) -> list[dict[str, Any]]:
    if len(examples) != len(grouped_completions):
        raise ValueError("completion groups do not match the number of examples")

    records: list[dict[str, Any]] = []
    for example, completions in zip(examples, grouped_completions, strict=True):
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


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sampled_metrics(
    records: Sequence[dict[str, Any]],
    examples: Sequence[Example],
    num_samples: int,
) -> dict[str, Any]:
    metrics = summarize(records)
    correct_by_question = {example.id: 0 for example in examples}
    for record in records:
        correct_by_question[record["question_id"]] += int(record["correct"])
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


def main() -> None:
    """Run greedy and sampled GSM8K baseline evaluation."""
    args = _parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    started_at = time.perf_counter()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _select_device()
    print(f"Using device: {device}", flush=True)

    examples = load_gsm8k(args.split, args.limit)
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    prompts = [build_messages(example.question) for example in examples]
    eos_token_id = model.generation_config.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("model and tokenizer must define EOS and pad token ids")
    if not isinstance(eos_token_id, int):
        eos_token_id = list(eos_token_id)

    output_dir = args.output_dir or _default_output_dir(args.model, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {}
    settings: dict[str, Any] = {}

    if not args.skip_greedy:
        greedy_settings = _generation_settings(
            num_samples=1,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        greedy_completions = generate(
            model,
            tokenizer,
            prompts,
            num_samples=1,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        greedy_records = _score_completions(examples, greedy_completions)
        _write_jsonl(output_dir / "completions_greedy.jsonl", greedy_records)
        settings["greedy"] = greedy_settings
        metrics["greedy"] = summarize(greedy_records)

    sampled_settings = _generation_settings(
        num_samples=args.num_samples,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    sampled_completions = generate(
        model,
        tokenizer,
        prompts,
        num_samples=args.num_samples,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    sampled_records = _score_completions(examples, sampled_completions)
    _write_jsonl(output_dir / "completions_sampled.jsonl", sampled_records)
    settings["sampled"] = sampled_settings
    metrics["sampled"] = _sampled_metrics(
        sampled_records,
        examples,
        args.num_samples,
    )

    summary = {
        "model": args.model,
        "dataset": DATASET_NAME,
        "split": args.split,
        "num_questions": len(examples),
        "system_prompt": SYSTEM_PROMPT,
        "seed": args.seed,
        "generation_settings": settings,
        "metrics": metrics,
        "runtime_seconds": time.perf_counter() - started_at,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Outputs: {output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
