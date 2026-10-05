# Step 3b spec: policy, training loop, evaluation, run tracking

Everything around the step 3a math core needed to train vanilla GRPO on the RTX 4070 Ti: the LoRA policy, rollouts, the training loop, configuration, shared evaluation, entry-point scripts, and run tracking. **`train_step` (the update loop) is written by hand**: this spec gives its contract and a stub. Everything else is implemented from this spec.

## Decisions

| | Choice |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`; bf16 on CUDA, float32 elsewhere |
| LoRA (PEFT) | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`; r = 16; α = 32; dropout 0 |
| Optimizer | AdamW over **LoRA parameters only**; learning rate 1e-5 (constant); weight decay 0; gradient-norm clip 1.0 |
| Rollout | 32 questions × 8 samples; temperature 1.0, top-p 1.0, top-k off, repetition penalty 1.0; `max_new_tokens` 512 (as the baseline) |
| Updates | 4 mini-batches of 64 completions per rollout; micro-batches of 8; `grpo_loss` with ε 0.2/0.2, β 0 (KL still logged), aggregation `sample` |
| Reference model | the same model with the adapter disabled |
| Training data | GSM8K train (7,473), validated at load; reshuffled each epoch from the seed; no question repeats within an epoch |
| Evaluation | at step 0 and every 10 steps: 200 fixed test questions (chosen once with `eval_seed`), greedy + 4 samples at T = 1; at the end: all 1,319 test questions, greedy + 8 samples (same as the step 2 baseline) |
| Checkpoints | LoRA adapter every 25 steps and at the end |
| Tracking | JSONL files always; W&B on top |
| First run | preset `vanilla`, 100 steps |

## Design rules

1. **Train on the token ids that were generated.** Completions carry their generated token ids; never rebuild them by tokenizing decoded text.
2. **One partition, reused.** After a rollout, rows are shuffled once and split into micro-batches, each right-padded once. `logp_old`, `logp_ref` and `logp_new` are all computed on these same tensors.
3. **Only LoRA parameters are trained.** The optimizer receives only parameters with `requires_grad`; base weights must be bit-identical after training (tested).
4. **One source of truth for generation settings.** Build the `generate` keyword arguments once and record that same dict (fixes the duplication noted in the step 2 review).

## `src/grpo_dapo/config.py`

A single `@dataclass Config`, fields grouped as below, with these defaults:

```text
model:        model_name="Qwen/Qwen2.5-0.5B-Instruct", gradient_checkpointing=True
lora:         lora_r=16, lora_alpha=32, lora_dropout=0.0,
              lora_target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
algorithm:    eps_low=0.2, eps_high=0.2, beta=0.0, aggregation="sample"
rollout:      questions_per_step=32, samples_per_question=8, temperature=1.0, top_p=1.0,
              max_new_tokens=512, max_prompt_tokens=512, generation_batch_size=32
optimization: num_minibatches=4, micro_batch_size=8, learning_rate=1e-5, max_grad_norm=1.0
run:          max_steps=100, seed=0, run_name=None, output_dir="outputs/train"
evaluation:   eval_every=10, eval_questions=200, eval_samples=4, eval_seed=1234, final_eval=True,
              save_every=25, log_samples_every=10, logged_sample_groups=2
tracking:     wandb_mode="online"  ("online" | "offline" | "disabled"), wandb_project="grpo-dapo"
```

- `PRESETS: dict[str, dict[str, Any]]`, each preset an explicit set of field values: `vanilla: {}`, `vanilla_kl: {"beta": 0.04}`, `clip_higher: {"eps_high": 0.28}`, `token_level: {"aggregation": "token"}`. (Step 6 adds `dynamic_sampling`, `overlong`, `dapo`.)
- `build_parser()`: generates a `--field-name` flag for **every** field from the dataclass (int, float, str; bool via `--flag/--no-flag`; tuple via `nargs="+"`), plus `--preset` (default `vanilla`).
- `load_config(argv=None) -> Config`: defaults → preset → flags explicitly given on the command line (**the command line always wins**).
- `__post_init__` validation (`ValueError`): `aggregation ∈ {sample, token}`; `samples_per_question ≥ 2`; `questions_per_step × samples_per_question` divisible by `num_minibatches`; mini-batch size divisible by `micro_batch_size`; `eps_low, eps_high, beta ≥ 0`; `wandb_mode` valid.
- `run_name` default: `"{preset}_s{seed}_{YYYYmmdd_HHMMSS}"`.
- `to_dict()` returns the fully resolved config (saved to `run.json` and W&B).

## `src/grpo_dapo/generation.py` (changes)

- `Completion` gains `token_ids: tuple[int, ...]` (the generated tokens through the first end token, inclusive) and `prompt_token_ids: tuple[int, ...]` (the prompt as tokenized for generation, without padding).
- Build the `model.generate` keyword arguments once in a helper and return them alongside the completions, so callers record exactly what was used.

## `src/grpo_dapo/policy.py`

`class Policy` wraps the PEFT model and tokenizer.

| Method | Behavior |
| --- | --- |
| `Policy(config, device)` | load tokenizer and base model (bf16 on CUDA), wrap with `get_peft_model(LoraConfig(...))`; if `gradient_checkpointing`, enable it (and `enable_input_require_grads()`); print the number of trainable parameters |
| `Policy.from_model(model, tokenizer, config)` | the same, around an already-built model (for tests with a tiny model) |
| `trainable_parameters()` | parameters with `requires_grad` (LoRA only) |
| `generate(prompts, *, num_samples, do_sample, temperature, top_p, max_new_tokens, batch_size)` | eval mode, `torch.inference_mode()`, adapter on; calls `generation.generate`; returns completions grouped per prompt with token ids |
| `score(micro_batch, *, with_grad, use_adapter=True, return_entropy=False)` | one forward pass over `input_ids`/`attention_mask` (`use_cache=False`); returns `token_log_probs` `[b, T−1]` and, if asked, `token_entropy` under `no_grad`. `with_grad=False` runs under `no_grad` in eval mode; `with_grad=True` in train mode. `use_adapter=False` runs inside `model.disable_adapter()` |
| `save_adapter(path)` | `save_pretrained` for the adapter, plus the tokenizer |

## `src/grpo_dapo/rollout.py`

```text
@dataclass MicroBatch:
    input_ids [b, T] (right-padded), attention_mask [b, T], completion_mask [b, T−1],
    advantages [b], row_ids [b],
    logp_old [b, T−1] | None, logp_ref [b, T−1] | None, entropy [b, T−1] | None

@dataclass RolloutBatch:
    examples, completions (grouped [Q][G]), rewards [Q, G], advantages [Q, G],
    zero_variance [Q], micro_batches: list[MicroBatch], stats: dict[str, float]
```

- `build_micro_batches(prompt_ids, completion_ids, advantages, micro_batch_size, pad_token_id, generator) -> list[MicroBatch]` (pure): one row per completion; shuffle rows once with `generator`; split into micro-batches; per micro-batch concatenate prompt + completion ids, right-pad to that micro-batch's longest row, set `attention_mask`, and build `completion_mask` with `logprobs.completion_mask`.
- `collect_rollout(policy, examples, config, generator) -> RolloutBatch`: generate `samples_per_question` completions per question; reward with `compute_reward(text, example.gold)`; `group_advantages` and `zero_variance_groups`; `build_micro_batches`; then for every micro-batch, without gradient: `logp_old` and `entropy` (adapter on), `logp_ref` (adapter off, even when β = 0). `stats`: reward mean, accuracy, format rate, unparseable rate, all-correct/all-wrong/mixed shares, mean/max completion length, truncation rate, mean entropy over completion tokens.

## `src/grpo_dapo/trainer.py`

### `train_step(policy, optimizer, rollout, config) -> dict[str, float]` — **stub, written by hand**

Contract:
- Split `rollout.micro_batches` into `config.num_minibatches` consecutive mini-batches.
- For each mini-batch: zero the gradients; compute `denominator` (completions in the mini-batch for `sample`, completion tokens in the mini-batch for `token`); for each micro-batch: `logp_new = policy.score(mb, with_grad=True).logp`, then `grpo_loss(logp_new, mb.logp_old, mb.advantages, mb.completion_mask, eps_low=…, eps_high=…, beta=…, logp_ref=mb.logp_ref, aggregation=…, denominator=…)`, then `loss.backward()`; after all micro-batches: clip the gradient norm of `policy.trainable_parameters()` to `max_grad_norm`, then `optimizer.step()`.
- Return: `loss` (per update, averaged over updates), `grad_norm` (mean), `clip_low`, `clip_high`, `clip_fraction`, `ratio_mean` (means over micro-batches), `ratio_min`, `ratio_max` (over all), `kl_mean`, and the health metrics from the **first** mini-batch: `first_update_max_abs_log_ratio`, `first_update_clip_fraction`. Print a warning if `first_update_max_abs_log_ratio > 1e-4`.

### `train(config, *, policy=None, train_examples=None, eval_examples=None)`

Arguments after `config` are for tests (inject a tiny model and data).

1. Seed `random`, `torch`; create `output_dir/run_name/`; write `run.json`: resolved config, git commit and whether the tree had uncommitted changes, versions of torch/transformers/peft, GPU name, command line, start time. Initialize W&B with the resolved config (per `wandb_mode`).
2. Load the policy, AdamW, GSM8K train (validated) and the fixed evaluation subset.
3. Evaluate at step 0.
4. For each step 1…`max_steps`: next `questions_per_step` questions → `collect_rollout` → `train_step` → append one line to `metrics.jsonl` and log to W&B; every `log_samples_every` steps append `logged_sample_groups` full groups (question, gold, 8 completions, rewards, advantages) to `samples.jsonl`; every `eval_every` steps evaluate (append to `eval.jsonl`, log to W&B); every `save_every` steps save the adapter to `checkpoints/step_XXXX/`.
5. At the end: save `checkpoints/final/`; if `final_eval`, evaluate on all 1,319 test questions with greedy + 8 samples and write `final_eval.json`; write `summary.json`; finish W&B.

### Metrics per step (`metrics.jsonl` and W&B)

```text
step, epoch
reward/mean, reward/accuracy, reward/format_rate, reward/unparseable_rate
groups/all_correct, groups/all_wrong, groups/mixed
length/mean, length/max, length/truncation_rate
policy/entropy, policy/kl_mean
clip/low, clip/high, clip/total, ratio/min, ratio/mean, ratio/max
health/first_update_max_abs_log_ratio, health/first_update_clip_fraction
optim/loss, optim/grad_norm, optim/lr
cost/rollout_seconds, cost/score_seconds, cost/update_seconds, cost/step_seconds,
cost/samples_generated_total, cost/tokens_generated_total, cost/peak_gpu_memory_gb
```

Evaluation keys: `eval/greedy_accuracy`, `eval/greedy_format_rate`, `eval/greedy_truncation_rate`, `eval/pass@1`, `eval/pass@4` (`pass@8` in the final evaluation), `eval/sampled_format_rate`, `eval/sampled_truncation_rate`, `eval/mean_length`.

## `src/grpo_dapo/evaluation.py` and `scripts/`

- Move scoring and summaries out of `eval_baseline.py` into public functions: `score_completions(examples, groups)`, `summarize_greedy(records)`, `summarize_sampled(records, examples, num_samples)` (pass@k for k ∈ {1, 2, 4, 8} up to `num_samples`, plus `group_stats`).
- `evaluate(model, tokenizer, examples, *, num_samples, max_new_tokens, batch_size) -> (metrics, records)`: greedy (1) + sampled (`num_samples`, T = 1, top-p 1) via `generation.generate`. Used by both the baseline script and the trainer.
- `scripts/eval_baseline.py`: same flags, behavior and output files as the current `python -m grpo_dapo.eval_baseline`, which is removed. `scripts/train.py`: `load_config()` then `train(config)`.
- Update the README commands.

## Tests (CPU, no network)

Use a **tiny randomly initialized Qwen2 model** built from `Qwen2Config` (2 layers, hidden size 32, 4 heads, 2 KV heads), wrapped with LoRA, and token ids built directly. Tests that need the real tokenizer load it with `local_files_only=True` and are skipped if it isn't cached.

| Area | Checks |
| --- | --- |
| `build_micro_batches` | right padding and `attention_mask`; `completion_mask` covers exactly each completion; every row appears exactly once; the shuffle is reproducible from the seed |
| scoring invariant | scoring the same `MicroBatch` twice gives identical log-probs (max difference 0); eval-mode and train-mode scores are identical (dropout 0) |
| reference | after making LoRA `B` non-zero, `use_adapter=False` scores equal the base model's and differ from adapter-on scores |
| trainable parameters | only LoRA parameters are trainable; after `train_step`, base weights are bit-identical and some LoRA weights changed |
| `train_step` (on a hand-made rollout) | returns all keys; `first_update_max_abs_log_ratio` ≤ 1e-7; the optimizer stepped exactly `num_minibatches` times; loss finite |
| config | defaults < preset < command line; bool and tuple flags; each validation error; `to_dict` contains every field |
| evaluation refactor | `summarize_*` reproduce the old helpers' outputs on synthetic records; `scripts/eval_baseline.py --help` runs |
| trainer smoke | `train()` for 2 steps with the tiny model and 4 injected questions writes `run.json`, `metrics.jsonl`, `eval.jsonl` with the documented keys (W&B disabled) |

Tests for `train_step` fail until it is written by hand.

## Implementation requirements

- `uv add peft wandb`. Nothing else.
- **Do not modify** `reward.py`, `logprobs.py`, `advantage.py`, `loss.py` or their tests. Leave `train_step` as a stub raising `NotImplementedError` with its contract as the docstring.
- Every run artifact goes under `outputs/` (git-ignored).

## How to run

Smoke test on the Mac (real model, tiny settings):

```bash
uv run python scripts/train.py --preset vanilla --max-steps 2 --questions-per-step 2 --samples-per-question 4 --num-minibatches 2 --micro-batch-size 4 --eval-questions 4 --eval-samples 2 --no-final-eval --wandb-mode disabled
```

First real run on the 4070 Ti (after `wandb login`):

```bash
uv run python scripts/train.py --preset vanilla
```
