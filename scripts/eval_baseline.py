"""Evaluate an untrained model on GSM8K with greedy and sampled decoding."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from grpo_dapo.data import load_gsm8k
from grpo_dapo.evaluation import (
    evaluate,
    score_completions,
    summarize_greedy,
    summarize_sampled,
)
from grpo_dapo.generation import generate, load_model_and_tokenizer
from grpo_dapo.prompts import SYSTEM_PROMPT, build_messages


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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the baseline evaluation and write its original artifacts."""
    args = _parse_args()
    if args.num_samples <= 0 or args.max_new_tokens <= 0 or args.batch_size <= 0:
        raise ValueError("sample count, token limit, and batch size must be positive")
    started_at = time.perf_counter()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _select_device()
    examples = load_gsm8k(args.split, args.limit)
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    if not args.skip_greedy and args.temperature == 1.0 and args.top_p == 1.0:
        metrics, records = evaluate(
            model,
            tokenizer,
            examples,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    else:
        prompts = [build_messages(example.question) for example in examples]
        records = {"generation_settings": {}}
        metrics = {}
        if not args.skip_greedy:
            greedy_groups, greedy_settings = generate(
                model,
                tokenizer,
                prompts,
                num_samples=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                desc="Greedy",
            )
            records["greedy"] = score_completions(examples, greedy_groups)
            records["generation_settings"]["greedy"] = greedy_settings
        sampled_groups, sampled_settings = generate(
            model,
            tokenizer,
            prompts,
            num_samples=args.num_samples,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            desc="Sampled",
        )
        records["sampled"] = score_completions(examples, sampled_groups)
        records["generation_settings"]["sampled"] = sampled_settings

    output_dir = args.output_dir or _default_output_dir(args.model, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "greedy" in records:
        _write_jsonl(output_dir / "completions_greedy.jsonl", records["greedy"])
    _write_jsonl(output_dir / "completions_sampled.jsonl", records["sampled"])
    summarized_metrics: dict[str, Any] = {
        "sampled": summarize_sampled(
            records["sampled"], examples, args.num_samples
        )
    }
    if "greedy" in records:
        summarized_metrics["greedy"] = summarize_greedy(records["greedy"])
    summary = {
        "model": args.model,
        "dataset": DATASET_NAME,
        "split": args.split,
        "num_questions": len(examples),
        "system_prompt": SYSTEM_PROMPT,
        "seed": args.seed,
        "generation_settings": records["generation_settings"],
        "metrics": summarized_metrics,
        "training_metrics": metrics,
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
