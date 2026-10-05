"""LoRA policy wrapper for generation and token scoring."""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn

from grpo_dapo.config import Config
from grpo_dapo.generation import Completion, generate, load_model_and_tokenizer
from grpo_dapo.logprobs import token_entropy, token_log_probs


@dataclass(frozen=True)
class ScoreOutput:
    """Token scores optionally accompanied by diagnostic entropy."""

    logp: Tensor
    entropy: Tensor | None = None


class Policy:
    """A causal language model with a trainable LoRA adapter."""

    def __init__(self, config: Config, device: str | torch.device) -> None:
        """Load and wrap the configured base model and tokenizer."""
        model, tokenizer = load_model_and_tokenizer(config.model_name, device)
        self._initialize(model, tokenizer, config)

    @classmethod
    def from_model(cls, model: nn.Module, tokenizer: Any, config: Config) -> "Policy":
        """Wrap an already-built model, primarily for offline tests."""
        policy = cls.__new__(cls)
        policy._initialize(model, tokenizer, config)
        return policy

    def _initialize(self, model: nn.Module, tokenizer: Any, config: Config) -> None:
        self.config = config
        self.tokenizer = tokenizer
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
        )
        self.model = get_peft_model(model, lora_config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        trainable = sum(parameter.numel() for parameter in self.trainable_parameters())
        total = sum(parameter.numel() for parameter in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}", flush=True)

    @property
    def device(self) -> torch.device:
        """Return the device holding the wrapped model parameters."""
        return next(self.model.parameters()).device

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only parameters optimized by LoRA training."""
        return [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

    def generate(
        self,
        prompts: list[list[dict[str, str]]],
        *,
        num_samples: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        batch_size: int,
    ) -> list[list[Completion]]:
        """Generate grouped completions with the adapter enabled."""
        self.model.eval()
        with torch.inference_mode():
            completions, _ = generate(
                self.model,
                self.tokenizer,
                prompts,
                num_samples=num_samples,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
        return completions

    def score(
        self,
        micro_batch: Any,
        *,
        with_grad: bool,
        use_adapter: bool = True,
        return_entropy: bool = False,
    ) -> ScoreOutput:
        """Score a right-padded micro-batch with optional gradients and entropy."""
        self.model.train(mode=with_grad)
        input_ids = micro_batch.input_ids.to(self.device)
        attention_mask = micro_batch.attention_mask.to(self.device)
        grad_context = nullcontext() if with_grad else torch.no_grad()
        adapter_context = (
            nullcontext() if use_adapter else self.model.disable_adapter()
        )
        with adapter_context, grad_context:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logp = token_log_probs(output.logits, input_ids)
            entropy = None
            if return_entropy:
                with torch.no_grad():
                    entropy = token_entropy(output.logits)
        return ScoreOutput(logp=logp, entropy=entropy)

    def save_adapter(self, path: str | Path) -> None:
        """Save the LoRA adapter and tokenizer to a directory."""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
