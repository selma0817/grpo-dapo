"""Rule-based reward: score a model completion against the gold answer."""

import re
from fractions import Fraction


_BOX_PREFIX = r"\boxed{"
_NUMBER_PATTERN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")
_DIGIT_SEPARATING_COMMA = re.compile(r"(?<=[0-9]),(?=[0-9])")


def extract_gold(answer_field: str | None) -> str | None:
    """Return the stripped text after the final GSM8K answer marker."""
    if not isinstance(answer_field, str):
        return None

    marker_position = answer_field.rfind("####")
    if marker_position == -1:
        return None

    gold = answer_field[marker_position + len("####") :].strip()
    return gold or None


def extract_answer(completion: str | None) -> str | None:
    """Return the stripped content of the last complete boxed answer."""
    if not isinstance(completion, str):
        return None

    open_braces: list[tuple[int, int | None]] = []
    last_box_position = -1
    last_content_bounds: tuple[int, int] | None = None
    index = 0

    while index < len(completion):
        if completion.startswith(_BOX_PREFIX, index):
            brace_position = index + len(_BOX_PREFIX) - 1
            open_braces.append((brace_position, index))
            index += len(_BOX_PREFIX)
            continue

        character = completion[index]
        if character == "{":
            open_braces.append((index, None))
        elif character == "}" and open_braces:
            brace_position, box_position = open_braces.pop()
            if box_position is not None and box_position > last_box_position:
                last_box_position = box_position
                last_content_bounds = (brace_position + 1, index)
        index += 1

    if last_content_bounds is None:
        return None

    start, end = last_content_bounds
    answer = completion[start:end].strip()
    return answer or None


def parse_number(text: str | None) -> Fraction | None:
    """Return the single number in text as an exact fraction, if present."""
    if not isinstance(text, str):
        return None

    cleaned = text.replace(r",\!", ",")
    cleaned = cleaned.replace(r"\!", ",").replace(r"\,", ",")
    cleaned = cleaned.replace("{,}", ",").replace("−", "-")
    cleaned = cleaned.replace(r"\$", "").replace("$", "").replace("%", "")
    cleaned = _DIGIT_SEPARATING_COMMA.sub("", cleaned)
    matches = _NUMBER_PATTERN.findall(cleaned)
    if len(matches) != 1:
        return None

    try:
        return Fraction(matches[0])
    except ValueError:
        return None


def is_equivalent(pred: str | None, gold: str | None) -> bool:
    """Return whether prediction and gold parse to the same exact number."""
    pred_number = parse_number(pred)
    gold_number = parse_number(gold)
    return (
        pred_number is not None
        and gold_number is not None
        and pred_number == gold_number
    )


def compute_reward(completion: str | None, gold: str | None) -> float:
    """Return one for an equivalent boxed answer and zero otherwise."""
    answer = extract_answer(completion)
    return 1.0 if is_equivalent(answer, gold) else 0.0
