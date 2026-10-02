"""Offline tests for GSM8K example validation."""

from fractions import Fraction

import pytest

from grpo_dapo.data import build_examples


def test_build_examples() -> None:
    rows = [
        {"question": "What is 70 + 2?", "answer": "Work\n#### 72"},
        {"question": "What is ten hundreds?", "answer": "Work\n#### 1,000"},
    ]

    examples = build_examples(rows)

    assert len(examples) == 2
    assert examples[0].id == 0
    assert examples[0].question == "What is 70 + 2?"
    assert examples[0].gold == "72"
    assert examples[0].gold_number == Fraction(72)
    assert examples[1].gold == "1,000"
    assert examples[1].gold_number == Fraction(1000)


def test_build_examples_reports_every_invalid_gold_id() -> None:
    rows = [
        {"question": "Good", "answer": "#### 72"},
        {"question": "Missing marker", "answer": "seventy-two"},
        {"question": "Also bad", "answer": "#### twelve"},
    ]

    with pytest.raises(ValueError) as error:
        build_examples(rows)

    assert "1" in str(error.value)
    assert "2" in str(error.value)


def test_build_examples_rejects_unparseable_extracted_gold() -> None:
    rows = [{"question": "Spell it", "answer": "#### twelve"}]

    with pytest.raises(ValueError, match="0"):
        build_examples(rows)
