"""GSM8K loading and validation for baseline evaluation."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from grpo_dapo.reward import extract_gold, parse_number


@dataclass(frozen=True)
class Example:
    """A validated GSM8K example."""

    id: int
    question: str
    gold: str
    gold_number: Fraction


def build_examples(rows: Iterable[Mapping[str, Any]]) -> list[Example]:
    """Build examples and report every row whose gold answer is invalid."""
    examples: list[Example] = []
    invalid_ids: list[int] = []

    for row_id, row in enumerate(rows):
        gold = extract_gold(row.get("answer"))
        gold_number = parse_number(gold)
        if gold is None or gold_number is None:
            invalid_ids.append(row_id)
            continue

        question = row.get("question")
        if not isinstance(question, str):
            raise ValueError(f"Row {row_id} has an invalid question")
        examples.append(
            Example(
                id=row_id,
                question=question,
                gold=gold,
                gold_number=gold_number,
            )
        )

    if invalid_ids:
        listed_ids = ", ".join(str(row_id) for row_id in invalid_ids)
        raise ValueError(f"Invalid gold answer in row ids: {listed_ids}")

    return examples


def load_gsm8k(split: str, limit: int | None = None) -> list[Example]:
    """Load and validate the requested GSM8K split."""
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        rows = rows.select(range(min(limit, len(rows))))
    return build_examples(rows)
