"""Trainer artifact smoke test with injected tiny policy and data."""

import json
from fractions import Fraction

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import grpo_dapo.trainer as trainer
from grpo_dapo.config import Config
from grpo_dapo.data import Example
from grpo_dapo.generation import Completion
from grpo_dapo.policy import Policy


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def save_pretrained(self, path) -> None:
        (path / "tokenizer.txt").write_text("tiny", encoding="utf-8")


class _Run:
    def __init__(self) -> None:
        self.logged = []

    def log(self, values, step=None) -> None:
        self.logged.append((step, values))

    def finish(self) -> None:
        pass


def _policy(config: Config) -> Policy:
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    )
    return Policy.from_model(model, _Tokenizer(), config)


def _run_training(tmp_path, monkeypatch, *, final_eval: bool):
    """Run a two-step training smoke test; return the run directory and fake W&B run."""
    config = Config(
        gradient_checkpointing=False,
        questions_per_step=2,
        samples_per_question=2,
        num_minibatches=1,
        micro_batch_size=2,
        max_steps=2,
        run_name="smoke",
        output_dir=str(tmp_path),
        eval_every=1,
        eval_questions=2,
        eval_samples=2,
        final_eval=final_eval,
        save_every=1,
        log_samples_every=1,
        logged_sample_groups=1,
        wandb_mode="disabled",
    )
    policy = _policy(config)
    run = _Run()

    def fake_generate(prompts, **kwargs):
        return [
            [
                Completion(r"\boxed{1}", 2, False, (5, 2), (3, 4))
                for _ in range(kwargs["num_samples"])
            ]
            for _ in prompts
        ]

    policy.generate = fake_generate
    monkeypatch.setattr(trainer.wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda *args, **kwargs: (
            {
                "eval/greedy_accuracy": 1.0,
                "eval/greedy_format_rate": 1.0,
                "eval/greedy_truncation_rate": 0.0,
                "eval/pass@1": 1.0,
                "eval/pass@2": 1.0,
                "eval/sampled_format_rate": 1.0,
                "eval/sampled_truncation_rate": 0.0,
                "eval/mean_length": 2.0,
            },
            {},
        ),
    )
    monkeypatch.setattr(
        trainer,
        "train_step",
        lambda *args, **kwargs: {
            "loss": 0.0,
            "grad_norm": 0.0,
            "clip_low": 0.0,
            "clip_high": 0.0,
            "clip_fraction": 0.0,
            "ratio_mean": 1.0,
            "ratio_min": 1.0,
            "ratio_max": 1.0,
            "kl_mean": 0.0,
            "first_update_max_abs_log_ratio": 0.0,
            "first_update_clip_fraction": 0.0,
        },
    )
    examples = [
        Example(index, f"question {index}", "1", Fraction(1))
        for index in range(4)
    ]

    run_directory = trainer.train(
        config,
        policy=policy,
        train_examples=examples,
        eval_examples=examples,
    )
    return run_directory, run


def test_train_two_step_smoke_writes_documented_artifacts(tmp_path, monkeypatch) -> None:
    run_directory, _ = _run_training(tmp_path, monkeypatch, final_eval=False)

    assert (run_directory / "run.json").is_file()
    metric_records = [
        json.loads(line)
        for line in (run_directory / "metrics.jsonl").read_text().splitlines()
    ]
    eval_records = [
        json.loads(line)
        for line in (run_directory / "eval.jsonl").read_text().splitlines()
    ]
    assert len(metric_records) == 2
    assert len(eval_records) == 3
    required = {
        "step",
        "epoch",
        "reward/mean",
        "reward/accuracy",
        "reward/format_rate",
        "reward/unparseable_rate",
        "groups/all_correct",
        "groups/all_wrong",
        "groups/mixed",
        "length/mean",
        "length/max",
        "length/truncation_rate",
        "policy/entropy",
        "policy/kl_mean",
        "clip/low",
        "clip/high",
        "clip/total",
        "ratio/min",
        "ratio/mean",
        "ratio/max",
        "health/first_update_max_abs_log_ratio",
        "health/first_update_clip_fraction",
        "optim/loss",
        "optim/grad_norm",
        "optim/lr",
        "cost/rollout_seconds",
        "cost/score_seconds",
        "cost/update_seconds",
        "cost/step_seconds",
        "cost/samples_generated_total",
        "cost/tokens_generated_total",
        "cost/peak_gpu_memory_gb",
    }
    assert required <= set(metric_records[0])
    assert (run_directory / "checkpoints" / "final").is_dir()
    assert (run_directory / "summary.json").is_file()


def test_final_eval_is_logged_under_separate_keys(tmp_path, monkeypatch) -> None:
    # The full-test evaluation must not overwrite the eval/ curve's last point.
    run_directory, run = _run_training(tmp_path, monkeypatch, final_eval=True)

    final_logs = [values for _, values in run.logged if "final/greedy_accuracy" in values]
    assert len(final_logs) == 1
    assert not any(key.startswith("eval/") for key in final_logs[0])
    assert (run_directory / "final_eval.json").is_file()
