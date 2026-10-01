"""Tests for the rule-based GSM8K reward."""

import time
from fractions import Fraction

import pytest

from grpo_dapo.reward import (
    compute_reward,
    extract_answer,
    extract_gold,
    is_equivalent,
    parse_number,
)


BOX_SPAM = " ".join(rf"\boxed{{{number}}}" for number in range(1, 101))
ODD_UNICODE = "\x00\u200b☃️�"
LONG_GARBAGE = "garbage " * 10_000
UNCLOSED_BRACES = r"\boxed{" * 10_000


@pytest.mark.parametrize(
    ("answer_field", "expected"),
    [
        # Real GSM8K example; numbers before the marker are ignored.
        pytest.param(
            "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
            "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April "
            "and May.\n#### 72",
            "72",
        ),
        # No space after the marker.
        pytest.param("...\n####72", "72"),
        # Trailing newline and whitespace are stripped.
        pytest.param("...\n#### 72\n", "72"),
        # Raw text is returned; parse_number handles cleanup.
        pytest.param("...\n#### 1,000", "1,000"),
        # A negative gold answer is retained.
        pytest.param("...\n#### -3", "-3"),
        # A missing marker fails closed.
        pytest.param("no marker here", None),
        # Non-string input fails closed.
        pytest.param(None, None),
        # Unusual Unicode characters do not raise.
        pytest.param(ODD_UNICODE, None),
        # A very long garbage string does not raise.
        pytest.param(LONG_GARBAGE, None),
    ],
)
def test_extract_gold(answer_field: str | None, expected: str | None) -> None:
    assert extract_gold(answer_field) == expected


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        # Normal boxed answer.
        pytest.param(r"... so the answer is \boxed{72}.", "72"),
        # Surrounding whitespace is stripped.
        pytest.param(r"\boxed{ 72 }", "72"),
        # With several boxes, the last one counts.
        pytest.param(r"First \boxed{70}. Wait, recheck: \boxed{72}", "72"),
        # The last box counts even when the earlier one was right.
        pytest.param(r"\boxed{72} ... actually \boxed{70}", "70"),
        # An unboxed number is not an answer.
        pytest.param("The answer is 72.", None),
        # An empty box has no usable answer.
        pytest.param(r"\boxed{}", None),
        # A whitespace-only box has no usable answer.
        pytest.param(r"\boxed{   }", None),
        # An unclosed box represents cut-off output.
        pytest.param(r"... so the total is \boxed{7", None),
        # Nested braces are balanced.
        pytest.param(r"\boxed{\text{18}}", r"\text{18}"),
        # Text inside the box is retained for parsing.
        pytest.param(r"\boxed{18 \text{ dollars}}", r"18 \text{ dollars}"),
        # Box spam does not help because only the last box counts.
        pytest.param(BOX_SPAM, "100"),
        # Non-string input fails closed.
        pytest.param(None, None),
        # Unusual Unicode characters do not raise.
        pytest.param(ODD_UNICODE, None),
        # A very long garbage string does not raise.
        pytest.param(LONG_GARBAGE, None),
        # Thousands of unclosed braces: must stay fast and not hit the recursion limit.
        pytest.param(UNCLOSED_BRACES, None),
        # The last complete box wins when the final box was cut off.
        pytest.param(r"\boxed{70} then \boxed{7", "70"),
    ],
)
def test_extract_answer(completion: str | None, expected: str | None) -> None:
    assert extract_answer(completion) == expected


def test_extract_answer_is_linear_time() -> None:
    # A linear scan handles 10,000 unclosed braces in milliseconds; scanning
    # forward from every \boxed{ is quadratic and takes minutes.
    start = time.perf_counter()
    assert extract_answer(UNCLOSED_BRACES) is None
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Plain integer.
        pytest.param("72", 72),
        # Thousands comma.
        pytest.param("1,000", 1000),
        # Several thousands commas.
        pytest.param("1,000,000", 1000000),
        # Dollar sign.
        pytest.param("$18", 18),
        # LaTeX-escaped dollar sign.
        pytest.param(r"\$18", 18),
        # Percent sign.
        pytest.param("25%", 25),
        # A trailing-zero decimal equals the integer exactly.
        pytest.param("18.0", 18),
        # Multiple trailing decimal zeros still equal the integer exactly.
        pytest.param("18.00", 18),
        # A non-integer decimal is represented exactly.
        pytest.param("2.50", Fraction(5, 2)),
        # Negative number.
        pytest.param("-3", -3),
        # Surrounding whitespace.
        pytest.param(" 18 ", 18),
        # Trailing words are ignored.
        pytest.param("18 dollars", 18),
        # A LaTeX text wrapper is ignored.
        pytest.param(r"\text{18}", 18),
        # One number among text is accepted.
        pytest.param("x = 18", 18),
        # Two numbers are ambiguous.
        pytest.param("12 or 18", None),
        # Text without a number cannot be parsed.
        pytest.param("no idea", None),
        # Empty text cannot be parsed.
        pytest.param("", None),
        # Fractions are out of scope until MATH (R5).
        pytest.param("1/2", None),
        # None-like text does not raise.
        pytest.param("None", None),
        # Non-string input fails closed.
        pytest.param(None, None),
        # Unusual Unicode characters do not raise.
        pytest.param(ODD_UNICODE, None),
        # A very long garbage string does not raise.
        pytest.param(LONG_GARBAGE, None),
    ],
)
def test_parse_number(text: str | None, expected: int | Fraction | None) -> None:
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["72", "2.50", "1,000"])
def test_parse_number_returns_exact_fraction(text: str) -> None:
    # D4: exact arithmetic. A float implementation passes the value checks above
    # (18.0 == 18), so check the type explicitly.
    assert isinstance(parse_number(text), Fraction)


@pytest.mark.parametrize(
    ("pred", "gold", "expected"),
    [
        # Identical numbers are equivalent.
        pytest.param("72", "72", True),
        # Decimal formatting does not change the numeric value.
        pytest.param("18.00", "18", True),
        # Comma formatting does not change the numeric value.
        pytest.param("1,000", "1000", True),
        # Different numbers are not equivalent.
        pytest.param("70", "72", False),
        # An ambiguous prediction is not equivalent.
        pytest.param("12 or 72", "72", False),
        # Exact comparison: these two are equal as floats.
        pytest.param("12345678901234567891", "12345678901234567890", False),
        # A missing prediction fails closed.
        pytest.param(None, "72", False),
        # A missing gold answer fails closed.
        pytest.param("72", None, False),
        # None-like text does not raise.
        pytest.param("None", "72", False),
        # Unusual Unicode characters do not raise.
        pytest.param(ODD_UNICODE, "72", False),
        # A very long garbage string does not raise.
        pytest.param(LONG_GARBAGE, "72", False),
    ],
)
def test_is_equivalent(
    pred: str | None, gold: str | None, expected: bool
) -> None:
    assert is_equivalent(pred, gold) is expected


@pytest.mark.parametrize(
    ("completion", "gold", "expected"),
    [
        # Correct answer.
        pytest.param(r"... \boxed{72}", "72", 1.0),
        # Wrong answer.
        pytest.param(r"... \boxed{70}", "72", 0.0),
        # Different comma formatting represents the same number.
        pytest.param(r"... \boxed{1,000}", "1000", 1.0),
        # Currency and decimal formatting represent the same number.
        pytest.param(r"... \boxed{\$18.00}", "18", 1.0),
        # A comma in the gold does not change its numeric value.
        pytest.param(r"... \boxed{1000}", "1,000", 1.0),
        # A correct but unboxed number gets no reward.
        pytest.param("The answer is 72.", "72", 0.0),
        # Self-correction to the right answer gets rewarded.
        pytest.param(r"\boxed{70} wait, recheck: \boxed{72}", "72", 1.0),
        # Self-correction to a wrong answer gets no reward.
        pytest.param(r"\boxed{72} actually \boxed{70}", "72", 0.0),
        # Box spam gets no reward when its final answer is wrong.
        pytest.param(BOX_SPAM, "72", 0.0),
        # Cut-off output gets no reward.
        pytest.param(r"... so the total is \boxed{7", "72", 0.0),
        # An empty box gets no reward.
        pytest.param(r"\boxed{}", "72", 0.0),
        # An ambiguous boxed answer gets no reward.
        pytest.param(r"\boxed{12 or 72}", "72", 0.0),
        # A negative answer is rewarded end to end.
        pytest.param(r"\boxed{-3}", "-3", 1.0),
        # A missing completion fails closed.
        pytest.param(None, "72", 0.0),
        # A missing gold answer fails closed.
        pytest.param(r"\boxed{72}", None, 0.0),
        # Unusual Unicode characters do not raise.
        pytest.param(ODD_UNICODE, "72", 0.0),
        # A very long garbage string does not raise.
        pytest.param(LONG_GARBAGE, "72", 0.0),
    ],
)
def test_compute_reward(
    completion: str | None, gold: str | None, expected: float
) -> None:
    assert compute_reward(completion, gold) == expected
