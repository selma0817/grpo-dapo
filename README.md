# grpo-dapo

GRPO and DAPO for math reasoning on Qwen2.5-Instruct, implemented in PyTorch + Transformers + PEFT, with controlled ablations of DAPO's four techniques.

**Status:** in progress.

## Plan

1. Rule-based reward and evaluation (pass@k) on GSM8K
2. Vanilla GRPO: rollouts, group-relative advantages, clipped policy loss, KL
3. Training diagnostics: entropy, KL, response length, zero-variance groups, clip fraction
4. DAPO techniques as independent switches: Clip-Higher, Dynamic Sampling, Token-level Loss, Overlong Reward Shaping
5. Ablations across seeds, including Clip-Higher's effect as a function of updates per rollout

## Layout

Modules are added with the milestone that needs them. Pure logic (parsing, metrics, loss math) is kept separate from I/O so it can be tested offline.

```text
src/grpo_dapo/
├── reward.py         ✅  rule-based reward: last \boxed{}, exact numeric equality
├── data.py           ✅  load and validate datasets (difficulty buckets in M2, MATH in R5)
├── prompts.py        ✅  system prompt and chat messages, shared by evaluation and training
├── generation.py     ✅  batched sampling (Hugging Face generate; vLLM later)
├── metrics.py        ✅  pass@k, group statistics, evaluation summaries
├── eval_baseline.py  ✅  M0 baseline evaluation CLI
├── config.py         M3  training hyperparameters and ablation presets
├── logprobs.py       M3  per-token log-probabilities with masks
├── advantage.py      M3  group-relative advantages (+ dynamic sampling in M5)
├── loss.py           M3  clipped objective, KL, loss averaging (+ Clip-Higher, token-level in M5)
├── train.py          M3  rollout → reward → advantage → update loop
└── diagnostics.py    M4  entropy, KL, clip fractions, zero-variance share, lengths
```

Specs for each milestone are in `docs/`.

## Development

```bash
uv sync
uv run pytest
```

## References

- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO), arXiv:2402.03300
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, arXiv:2503.14476
- Hugging Face TRL `GRPOTrainer` (reference implementation for numerical checks)
- volcengine/verl `core_algos.py` (DAPO reference implementation)
