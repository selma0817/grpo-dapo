"""Tests for Step 3b configuration resolution and validation."""

from dataclasses import fields

import pytest

from grpo_dapo.config import Config, build_parser, load_config


def test_defaults_then_preset_then_command_line_precedence() -> None:
    config = load_config(
        ["--preset", "vanilla_kl", "--beta", "0.1", "--seed", "7"]
    )

    assert config.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.beta == 0.1
    assert config.seed == 7
    assert config.run_name is not None
    assert config.run_name.startswith("vanilla_kl_s7_")


def test_bool_and_tuple_flags() -> None:
    config = load_config(
        [
            "--no-gradient-checkpointing",
            "--no-final-eval",
            "--lora-target-modules",
            "q_proj",
            "v_proj",
        ]
    )

    assert config.gradient_checkpointing is False
    assert config.final_eval is False
    assert config.lora_target_modules == ("q_proj", "v_proj")


@pytest.mark.parametrize(
    "overrides",
    [
        {"aggregation": "x"},
        {"samples_per_question": 1},
        {
            "questions_per_step": 1,
            "samples_per_question": 3,
            "num_minibatches": 2,
        },
        {
            "questions_per_step": 1,
            "samples_per_question": 2,
            "num_minibatches": 1,
            "micro_batch_size": 3,
        },
        {"eps_low": -0.1},
        {"eps_high": -0.1},
        {"beta": -0.1},
        {"wandb_mode": "sometimes"},
    ],
)
def test_config_validation_errors(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Config(**overrides)


def test_to_dict_and_parser_cover_every_field() -> None:
    config_names = {field.name for field in fields(Config)}
    parser_destinations = {action.dest for action in build_parser()._actions}

    assert set(Config().to_dict()) == config_names
    assert config_names <= parser_destinations
