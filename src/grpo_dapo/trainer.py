"""Training orchestration, artifact logging, and update-loop contract."""

import json
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import peft
import torch
import transformers
import wandb

from grpo_dapo.config import Config
from grpo_dapo.data import Example, load_gsm8k
from grpo_dapo.evaluation import evaluate
from grpo_dapo.loss import grpo_loss
from grpo_dapo.policy import Policy
from grpo_dapo.rollout import RolloutBatch, collect_rollout


def train_step(
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    config: Config,
) -> dict[str, float]:
    """Run grouped optimizer updates over a rollout.

    Split consecutive rollout micro-batches into ``config.num_minibatches``
    updates. For each update, zero gradients, compute its shared sample- or
    token-level denominator, score every micro-batch with gradients, accumulate
    ``grpo_loss`` gradients, clip trainable LoRA gradients, and step once.
    Return averaged loss, gradient, clipping, ratio, and KL metrics plus the
    first update's maximum absolute log-ratio and clipping fraction; warn when
    that first log-ratio exceeds ``1e-4``.
    """
    micro_batches = rollout.micro_batches
    num_micro_batches = len(micro_batches)
    if num_micro_batches == 0:
        raise ValueError("rollout must contain at least one micro-batch")
    if num_micro_batches % config.num_minibatches != 0:
        raise ValueError("micro-batch count must be divisible by num_minibatches")

    micro_batches_per_minibatch = num_micro_batches // config.num_minibatches
    trainable_parameters = policy.trainable_parameters()

    update_losses: list[float] = []
    grad_norms: list[float] = []
    clip_lows: list[float] = []
    clip_highs: list[float] = []
    clip_fractions: list[float] = []
    ratio_means: list[float] = []
    ratio_mins: list[float] = []
    ratio_maxs: list[float] = []
    kl_means: list[float] = []
    first_update_max_abs_log_ratio = 0.0
    first_update_clip_fractions: list[float] = []

    for minibatch_index in range(config.num_minibatches):
        start = minibatch_index * micro_batches_per_minibatch
        stop = start + micro_batches_per_minibatch
        minibatch = micro_batches[start:stop]

        optimizer.zero_grad(set_to_none=True)
        if config.aggregation == "sample":
            denominator = float(
                sum(micro_batch.advantages.numel() for micro_batch in minibatch)
            )
        else:
            denominator = sum(
                micro_batch.completion_mask.sum().item() for micro_batch in minibatch
            )

        minibatch_loss = 0.0
        for micro_batch in minibatch:
            if micro_batch.logp_old is None or micro_batch.logp_ref is None:
                raise ValueError(
                    "rollout micro-batches must include old and reference log-probabilities"
                )

            score_output = policy.score(micro_batch, with_grad=True, use_adapter=True)
            micro_batch_loss, micro_batch_stats = grpo_loss(
                logp_new=score_output.logp,
                logp_old=micro_batch.logp_old,
                advantages=micro_batch.advantages,
                mask=micro_batch.completion_mask,
                eps_low=config.eps_low,
                eps_high=config.eps_high,
                beta=config.beta,
                logp_ref=micro_batch.logp_ref,
                aggregation=config.aggregation,
                denominator=denominator,
            )

            micro_batch_loss.backward()
            minibatch_loss += micro_batch_loss.item()

            clip_low = float(micro_batch_stats["clip_low_fraction"])
            clip_high = float(micro_batch_stats["clip_high_fraction"])
            clip_fraction = float(micro_batch_stats["clip_fraction"])
            max_abs_log_ratio = float(micro_batch_stats["max_abs_log_ratio"])
            kl_mean = micro_batch_stats["kl_mean"]
            if kl_mean is None:
                raise RuntimeError("reference log-probabilities must produce a KL metric")

            clip_lows.append(clip_low)
            clip_highs.append(clip_high)
            clip_fractions.append(clip_fraction)
            ratio_means.append(float(micro_batch_stats["ratio_mean"]))
            ratio_mins.append(float(micro_batch_stats["ratio_min"]))
            ratio_maxs.append(float(micro_batch_stats["ratio_max"]))
            kl_means.append(kl_mean)

            if minibatch_index == 0:
                first_update_max_abs_log_ratio = max(
                    first_update_max_abs_log_ratio,
                    max_abs_log_ratio,
                )
                first_update_clip_fractions.append(clip_fraction)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            config.max_grad_norm,
        )
        grad_norms.append(grad_norm.item())
        optimizer.step()
        update_losses.append(minibatch_loss)

    first_update_clip_fraction = sum(first_update_clip_fractions) / len(
        first_update_clip_fractions
    )
    if first_update_max_abs_log_ratio > 1e-4:
        print(
            "Warning: first update max absolute log-ratio is "
            f"{first_update_max_abs_log_ratio:.6g}",
            file=sys.stderr,
        )

    return {
        "loss": sum(update_losses) / len(update_losses),
        "grad_norm": sum(grad_norms) / len(grad_norms),
        "clip_low": sum(clip_lows) / len(clip_lows),
        "clip_high": sum(clip_highs) / len(clip_highs),
        "clip_fraction": sum(clip_fractions) / len(clip_fractions),
        "ratio_mean": sum(ratio_means) / len(ratio_means),
        "ratio_min": min(ratio_mins),
        "ratio_max": max(ratio_maxs),
        "kl_mean": sum(kl_means) / len(kl_means),
        "first_update_max_abs_log_ratio": first_update_max_abs_log_ratio,
        "first_update_clip_fraction": first_update_clip_fraction,
    }


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False) + "\n")


class _QuestionStream:
    def __init__(self, examples: Sequence[Example], seed: int) -> None:
        if not examples:
            raise ValueError("training examples must not be empty")
        self.examples = examples
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self.order: list[int] = []
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = list(range(len(self.examples)))
        random.Random(self.seed + self.epoch).shuffle(self.order)
        self.position = 0

    def take(self, count: int) -> tuple[list[Example], int]:
        selected: list[Example] = []
        starting_epoch = self.epoch
        while len(selected) < count:
            if self.position == len(self.order):
                self.epoch += 1
                self._reshuffle()
            remaining = count - len(selected)
            stop = min(self.position + remaining, len(self.order))
            selected.extend(
                self.examples[index]
                for index in self.order[self.position : stop]
            )
            self.position = stop
        return selected, starting_epoch


def _fixed_eval_subset(
    examples: Sequence[Example], count: int, seed: int
) -> list[Example]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    return [examples[index] for index in indices[: min(count, len(indices))]]


def _evaluate_and_log(
    policy: Policy,
    examples: Sequence[Example],
    config: Config,
    step: int,
    path: Path,
    run: Any,
) -> dict[str, float]:
    metrics, _ = evaluate(
        policy.model,
        policy.tokenizer,
        examples,
        num_samples=config.eval_samples,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.generation_batch_size,
    )
    record: dict[str, Any] = {"step": step, **metrics}
    _append_jsonl(path, record)
    run.log(record, step=step)
    return metrics


def _step_metrics(
    *,
    step: int,
    epoch: int,
    rollout: RolloutBatch,
    update: dict[str, float],
    optimizer: torch.optim.Optimizer,
    rollout_seconds: float,
    update_seconds: float,
    step_seconds: float,
    samples_total: int,
    tokens_total: int,
) -> dict[str, float | int]:
    stats = rollout.stats
    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )
    return {
        "step": step,
        "epoch": epoch,
        "reward/mean": stats["reward_mean"],
        "reward/accuracy": stats["accuracy"],
        "reward/format_rate": stats["format_rate"],
        "reward/unparseable_rate": stats["unparseable_rate"],
        "groups/all_correct": stats["all_correct"],
        "groups/all_wrong": stats["all_wrong"],
        "groups/mixed": stats["mixed"],
        "groups/nonzero_variance": stats["nonzero_variance"],
        "length/mean": stats["mean_completion_length"],
        "length/max": stats["max_completion_length"],
        "length/truncation_rate": stats["truncation_rate"],
        "policy/entropy": stats["mean_entropy"],
        "policy/kl_mean": update["kl_mean"],
        "clip/low": update["clip_low"],
        "clip/high": update["clip_high"],
        "clip/total": update["clip_fraction"],
        "ratio/min": update["ratio_min"],
        "ratio/mean": update["ratio_mean"],
        "ratio/max": update["ratio_max"],
        "health/first_update_max_abs_log_ratio": update[
            "first_update_max_abs_log_ratio"
        ],
        "health/first_update_clip_fraction": update[
            "first_update_clip_fraction"
        ],
        "optim/loss": update["loss"],
        "optim/grad_norm": update["grad_norm"],
        "optim/lr": float(optimizer.param_groups[0]["lr"]),
        "cost/rollout_seconds": rollout_seconds,
        "cost/score_seconds": stats["score_seconds"],
        "cost/update_seconds": update_seconds,
        "cost/step_seconds": step_seconds,
        "cost/samples_generated_total": samples_total,
        "cost/tokens_generated_total": tokens_total,
        "cost/peak_gpu_memory_gb": peak_memory,
    }


def train(
    config: Config,
    *,
    policy: Policy | None = None,
    train_examples: Sequence[Example] | None = None,
    eval_examples: Sequence[Example] | None = None,
) -> Path:
    """Train a policy and write all run artifacts under its output directory."""
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats()

    run_directory = Path(config.output_dir) / str(config.run_name)
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoints = run_directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "config": config.to_dict(),
        "git": _git_metadata(),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "command": sys.argv,
        "start_time": started_at,
    }
    _write_json(run_directory / "run.json", metadata)
    wandb_run = wandb.init(
        project=config.wandb_project,
        name=config.run_name,
        mode=config.wandb_mode,
        config=config.to_dict(),
        dir=str(run_directory),
    )

    if policy is None:
        policy = Policy(config, _select_device())
    optimizer = torch.optim.AdamW(
        policy.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    if train_examples is None:
        train_examples = load_gsm8k("train")
    if eval_examples is None:
        eval_examples = load_gsm8k("test")
    fixed_eval = _fixed_eval_subset(
        eval_examples, config.eval_questions, config.eval_seed
    )
    eval_path = run_directory / "eval.jsonl"
    _evaluate_and_log(policy, fixed_eval, config, 0, eval_path, wandb_run)

    stream = _QuestionStream(train_examples, config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    samples_total = 0
    tokens_total = 0
    latest_metrics: dict[str, float | int] = {}
    train_started = time.perf_counter()
    for step in range(1, config.max_steps + 1):
        step_started = time.perf_counter()
        step_examples, epoch = stream.take(config.questions_per_step)
        rollout_started = time.perf_counter()
        rollout = collect_rollout(policy, step_examples, config, generator)
        rollout_seconds = time.perf_counter() - rollout_started
        update_started = time.perf_counter()
        update = train_step(policy, optimizer, rollout, config)
        update_seconds = time.perf_counter() - update_started
        samples_total += len(step_examples) * config.samples_per_question
        tokens_total += int(rollout.stats["tokens_generated"])
        latest_metrics = _step_metrics(
            step=step,
            epoch=epoch,
            rollout=rollout,
            update=update,
            optimizer=optimizer,
            rollout_seconds=rollout_seconds,
            update_seconds=update_seconds,
            step_seconds=time.perf_counter() - step_started,
            samples_total=samples_total,
            tokens_total=tokens_total,
        )
        _append_jsonl(run_directory / "metrics.jsonl", latest_metrics)
        wandb_run.log(latest_metrics, step=step)

        if step % config.log_samples_every == 0:
            for group_index in range(
                min(config.logged_sample_groups, len(step_examples))
            ):
                _append_jsonl(
                    run_directory / "samples.jsonl",
                    {
                        "step": step,
                        "question": step_examples[group_index].question,
                        "gold": step_examples[group_index].gold,
                        "completions": [
                            completion.text
                            for completion in rollout.completions[group_index]
                        ],
                        "rewards": rollout.rewards[group_index].tolist(),
                        "advantages": rollout.advantages[group_index].tolist(),
                    },
                )
        if step % config.eval_every == 0:
            _evaluate_and_log(
                policy, fixed_eval, config, step, eval_path, wandb_run
            )
        if step % config.save_every == 0:
            policy.save_adapter(checkpoints / f"step_{step:04d}")

    policy.save_adapter(checkpoints / "final")
    final_metrics: dict[str, float] | None = None
    if config.final_eval:
        final_metrics, final_records = evaluate(
            policy.model,
            policy.tokenizer,
            eval_examples,
            num_samples=8,
            max_new_tokens=config.max_new_tokens,
            batch_size=config.generation_batch_size,
        )
        _write_json(
            run_directory / "final_eval.json",
            {"metrics": final_metrics, "records": final_records},
        )
        # Separate keys: the full test set must not overwrite the step-N point
        # of the 200-question eval/ curve logged at the same step.
        wandb_run.log(
            {
                f"final/{key.split('/', 1)[-1]}": value
                for key, value in final_metrics.items()
            },
            step=config.max_steps,
        )

    summary = {
        "run_name": config.run_name,
        "steps": config.max_steps,
        "runtime_seconds": time.perf_counter() - train_started,
        "last_metrics": latest_metrics,
        "final_eval": final_metrics,
    }
    _write_json(run_directory / "summary.json", summary)
    wandb_run.finish()
    return run_directory
