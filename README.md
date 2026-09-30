# grpo-dapo

GRPO and DAPO for math reasoning on Qwen2.5-Instruct, implemented in PyTorch + Transformers + PEFT, with controlled ablations of DAPO's four techniques.

**Status:** in progress.

## Plan

1. Rule-based reward and evaluation (pass@k) on GSM8K
2. Vanilla GRPO: rollouts, group-relative advantages, clipped policy loss, KL
3. Training diagnostics: entropy, KL, response length, zero-variance groups, clip fraction
4. DAPO techniques as independent switches: Clip-Higher, Dynamic Sampling, Token-level Loss, Overlong Reward Shaping
5. Ablations across seeds, including Clip-Higher's effect as a function of updates per rollout

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
