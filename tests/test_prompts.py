"""Offline tests for baseline prompt construction."""

from grpo_dapo.prompts import SYSTEM_PROMPT, build_messages


def test_build_messages() -> None:
    question = "If Natalia has 72 clips, how many does she have?"

    assert build_messages(question) == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
