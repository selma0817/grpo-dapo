# grpo-dapo

GRPO and DAPO for math reasoning on Qwen2.5-Instruct, implemented in PyTorch + Transformers + PEFT, with controlled ablations of DAPO's four techniques.

**Status:** in progress.

## Plan

1. ✅ Rule-based reward: exact answer checking (`docs/reward-spec.md`)
2. ✅ Baseline evaluation on GSM8K: accuracy, pass@k, format, truncation, group statistics (`docs/m0-spec.md`)
3. GRPO core and first training run: LoRA policy, log-probs, group-relative advantages, clipped loss, KL, training loop
4. Training diagnostics; run vanilla GRPO until it fails
5. Difficulty-graded data: per-question solve rates, MATH support
6. DAPO techniques as independent switches: Clip-Higher, Dynamic Sampling, Token-level Loss, Overlong Reward Shaping
7. Ablations across seeds, including Clip-Higher's effect as a function of updates per rollout

## Layout

Modules are added in the step that needs them. All logic lives in the package, where it can be imported and tested offline; `scripts/` holds thin entry points.

```text
src/grpo_dapo/
├── reward.py         step 1 ✅  rule-based reward: last \boxed{}, exact numeric equality
├── data.py           step 2 ✅  load and validate datasets (difficulty buckets and MATH in step 5)
├── prompts.py        step 2 ✅  system prompt and chat messages, shared by evaluation and training
├── generation.py     step 2 ✅  batched sampling (Hugging Face generate; vLLM later)
├── metrics.py        step 2 ✅  pass@k, group statistics, evaluation summaries
├── evaluation.py     step 3 ✅  scoring and metrics shared by baseline and in-training evaluation
├── policy.py         step 3 ✅  LoRA policy: generate, log-probs, reference log-probs (adapter disabled)
├── logprobs.py       step 3     per-token log-probabilities with masks
├── advantage.py      step 3     group-relative advantages (+ dynamic sampling in step 6)
├── loss.py           step 3     clipped objective, KL, loss averaging (+ Clip-Higher, token-level in step 6)
├── config.py         step 3 ✅  training hyperparameters and ablation presets
├── rollout.py        step 3 ✅  rollout → reward → advantage → fixed micro-batches
├── trainer.py        step 3     training orchestration (`train_step` is the hand-written stub)
└── diagnostics.py    step 4     entropy, KL, clip fractions, zero-variance share, lengths
scripts/
├── eval_baseline.py  step 3 ✅  baseline evaluation entry point
├── train.py          step 3 ✅  training entry point
└── run_ablations.sh  step 7     launches the ablation runs
```

Specs for each step are in `docs/`. Run artifacts (completions, checkpoints) are written to `outputs/`, which is git-ignored; result summaries worth keeping are copied into `results/`.

## Development

```bash
uv sync
uv run pytest
```

Run the baseline evaluation:

```bash
uv run python scripts/eval_baseline.py
```

Run a two-step local smoke test:

```bash
uv run python scripts/train.py --preset vanilla --max-steps 2 --questions-per-step 2 --samples-per-question 4 --num-minibatches 2 --micro-batch-size 4 --eval-questions 4 --eval-samples 2 --no-final-eval --wandb-mode disabled
```

Start the first full vanilla GRPO run:

```bash
uv run python scripts/train.py --preset vanilla
```

Available presets are `vanilla`, `vanilla_kl`, `clip_higher`, `token_level`,
`dynamic_sampling`, `overlong`, and `dapo`. The `dapo` preset combines
Clip-Higher, token-level loss, Dynamic Sampling, and overlong reward shaping.

## References

- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO), arXiv:2402.03300
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, arXiv:2503.14476
- Hugging Face TRL `GRPOTrainer` (reference implementation for numerical checks)
- volcengine/verl `core_algos.py` (DAPO reference implementation)
