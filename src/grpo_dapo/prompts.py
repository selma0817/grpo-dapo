"""Prompt construction for GSM8K evaluation."""


SYSTEM_PROMPT = (
    r"Please reason step by step, and put your final answer within \boxed{}."
)


def build_messages(question: str) -> list[dict[str, str]]:
    """Build the system and user messages for one question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
