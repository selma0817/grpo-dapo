"""Configuration and command-line parsing for GRPO training."""

import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any


@dataclass
class Config:
    """Fully resolved training configuration."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    gradient_checkpointing: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    eps_low: float = 0.2
    eps_high: float = 0.2
    beta: float = 0.0
    aggregation: str = "sample"
    format_weight: float = 0.0

    dynamic_sampling: bool = False
    dynamic_max_rounds: int = 4
    overlong_cache: int = 0
    overlong_penalty_factor: float = 0.5
    overlong_filter: bool = False

    questions_per_step: int = 32
    samples_per_question: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    max_prompt_tokens: int = 512
    generation_batch_size: int = 32

    num_minibatches: int = 4
    micro_batch_size: int = 8
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0

    max_steps: int = 100
    seed: int = 0
    run_name: str | None = None
    output_dir: str = "outputs/train"

    eval_every: int = 10
    eval_questions: int = 200
    eval_samples: int = 4
    eval_seed: int = 1234
    final_eval: bool = True
    save_every: int = 25
    log_samples_every: int = 10
    logged_sample_groups: int = 2

    wandb_mode: str = "online"
    wandb_project: str = "grpo-dapo"

    def __post_init__(self) -> None:
        """Validate related settings and fill the default run name."""
        if self.aggregation not in {"sample", "token"}:
            raise ValueError("aggregation must be 'sample' or 'token'")
        if self.samples_per_question < 2:
            raise ValueError("samples_per_question must be at least 2")
        if self.num_minibatches <= 0:
            raise ValueError("num_minibatches must be positive")
        completions = self.questions_per_step * self.samples_per_question
        if completions % self.num_minibatches != 0:
            raise ValueError("completions per step must divide into num_minibatches")
        minibatch_size = completions // self.num_minibatches
        if self.micro_batch_size <= 0 or minibatch_size % self.micro_batch_size != 0:
            raise ValueError("mini-batch size must divide into micro_batch_size")
        if self.eps_low < 0 or self.eps_high < 0 or self.beta < 0:
            raise ValueError("eps_low, eps_high, and beta must be non-negative")
        if self.format_weight < 0:
            raise ValueError("format_weight must be non-negative")
        if self.dynamic_max_rounds < 1:
            raise ValueError("dynamic_max_rounds must be at least 1")
        if not 0 <= self.overlong_cache < self.max_new_tokens:
            raise ValueError(
                "overlong_cache must be non-negative and less than max_new_tokens"
            )
        if self.overlong_penalty_factor < 0:
            raise ValueError("overlong_penalty_factor must be non-negative")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"vanilla_s{self.seed}_{timestamp}"

    def to_dict(self) -> dict[str, Any]:
        """Return every resolved configuration field as a dictionary."""
        return asdict(self)


PRESETS: dict[str, dict[str, Any]] = {
    "vanilla": {},
    "vanilla_kl": {"beta": 0.04},
    "clip_higher": {"eps_high": 0.28},
    "token_level": {"aggregation": "token"},
    "dynamic_sampling": {"dynamic_sampling": True},
    "overlong": {"overlong_cache": 64},
    "dapo": {
        "eps_high": 0.28,
        "aggregation": "token",
        "dynamic_sampling": True,
        "overlong_cache": 64,
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build a parser containing a flag for every configuration field."""
    parser = argparse.ArgumentParser(description="Train GRPO or DAPO on GSM8K.")
    parser.add_argument("--preset", choices=PRESETS, default="vanilla")
    defaults = Config.__dataclass_fields__
    for field in fields(Config):
        name = field.name
        flag = f"--{name.replace('_', '-')}"
        default = defaults[name].default
        kwargs: dict[str, Any] = {"default": argparse.SUPPRESS}
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        elif isinstance(default, tuple):
            kwargs.update(type=str, nargs="+")
        elif default is None:
            kwargs["type"] = str
        else:
            kwargs["type"] = type(default)
        parser.add_argument(flag, dest=name, **kwargs)
    return parser


def load_config(argv: list[str] | None = None) -> Config:
    """Resolve dataclass defaults, a preset, and explicit command-line flags."""
    parsed = vars(build_parser().parse_args(argv))
    preset = parsed.pop("preset")
    overrides = parsed
    if "lora_target_modules" in overrides:
        overrides["lora_target_modules"] = tuple(overrides["lora_target_modules"])

    resolved = dict(PRESETS[preset])
    resolved.update(overrides)
    if "run_name" not in resolved:
        seed = int(resolved.get("seed", Config.__dataclass_fields__["seed"].default))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved["run_name"] = f"{preset}_s{seed}_{timestamp}"
    return Config(**resolved)
