"""Batched Hugging Face generation and completion accounting."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class Completion:
    """One decoded model completion and its token metadata."""

    text: str
    num_tokens: int
    truncated: bool


def _eos_set(eos_ids: int | Iterable[int]) -> set[int]:
    if isinstance(eos_ids, int):
        return {eos_ids}
    return {int(token_id) for token_id in eos_ids}


def completion_length(
    new_token_ids: Sequence[int], eos_ids: int | Iterable[int]
) -> int:
    """Count generated tokens through the first end-of-sequence token."""
    end_tokens = _eos_set(eos_ids)
    for index, token_id in enumerate(new_token_ids):
        if int(token_id) in end_tokens:
            return index + 1
    return len(new_token_ids)


def is_truncated(
    new_token_ids: Sequence[int],
    eos_ids: int | Iterable[int],
    max_new_tokens: int,
) -> bool:
    """Return whether generation exhausted its token budget without EOS."""
    end_tokens = _eos_set(eos_ids)
    considered_ids = new_token_ids[:max_new_tokens]
    return len(new_token_ids) >= max_new_tokens and not any(
        int(token_id) in end_tokens for token_id in considered_ids
    )


def load_model_and_tokenizer(name: str, device: str | torch.device) -> tuple[Any, Any]:
    """Load a causal LM and left-padding tokenizer on the requested device."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    resolved_device = torch.device(device)
    dtype = torch.bfloat16 if resolved_device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.to(resolved_device)
    model.eval()
    return model, tokenizer


def generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[list[dict[str, str]]],
    *,
    num_samples: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
) -> list[list[Completion]]:
    """Generate completions in batches, grouped by input prompt order."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    eos_token_id = model.generation_config.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("model and tokenizer have no EOS token id")
    eos_ids = _eos_set(eos_token_id)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has no pad token id")

    grouped: list[list[Completion]] = []
    for batch_start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[batch_start : batch_start + batch_size]
        rendered_prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in prompt_batch
        ]
        encoded = tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
        )
        prompt_length = encoded["input_ids"].shape[1]
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=0,
                repetition_penalty=1.0,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_samples,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

        new_output_ids = output_ids[:, prompt_length:]
        for prompt_index in range(len(prompt_batch)):
            prompt_completions: list[Completion] = []
            for sample_index in range(num_samples):
                row_index = prompt_index * num_samples + sample_index
                token_ids = new_output_ids[row_index].tolist()
                prompt_completions.append(
                    Completion(
                        text=tokenizer.decode(token_ids, skip_special_tokens=True),
                        num_tokens=completion_length(token_ids, eos_ids),
                        truncated=is_truncated(
                            token_ids,
                            eos_ids,
                            max_new_tokens,
                        ),
                    )
                )
            grouped.append(prompt_completions)

    return grouped
