# Step 3a spec: the GRPO math core

Three small modules of pure tensor functions: per-token log-probs and masks, group-relative advantages, and the clipped policy loss with KL. No model, no I/O, so everything is tested on CPU in seconds. **The implementation is written by hand** (by me, reviewed); the tests are written first from this spec.

Conventions match TRL 1.14.1 (`GRPOTrainer`), checked in its source:
- advantage: `(R − group mean) / (group std + 1e-4)`, std with Bessel's correction (n − 1)
- per-token loss: `−min(ρ·Â, clamp(ρ, 1 − ε_low, 1 + ε_high)·Â) + β·k3`, with `k3 = exp(logp_ref − logp) − (logp_ref − logp) − 1`
- `grpo` averaging: mean over each completion's tokens, then mean over completions; `dapo` averaging: sum over all tokens divided by the number of completion tokens in the whole accumulation window

## `src/grpo_dapo/logprobs.py`

| Function | Signature | Behavior |
| --- | --- | --- |
| `token_log_probs` | `(logits: Tensor, input_ids: Tensor) -> Tensor` | `logits` `[B, T, V]` (any float dtype), `input_ids` `[B, T]` → `[B, T−1]` **float32**. Entry `j` is the log-probability of `input_ids[:, j+1]` under the distribution at position `j` (output at `t` predicts token `t+1`). Compute as `logit[token] − logsumexp(logits)` in float32; **never build the full `[B, T, V]` log-softmax**. |
| `token_entropy` | `(logits: Tensor) -> Tensor` | `[B, T, V]` → `[B, T−1]` float32: entropy of the next-token distribution at positions `0 … T−2` (aligned with `token_log_probs`): `logsumexp(z) − Σ softmax(z)·z`. Used for diagnostics, under `torch.no_grad()`. |
| `completion_mask` | `(prompt_lengths: Tensor, completion_lengths: Tensor, seq_len: int) -> Tensor` | `[B]`, `[B]` → `[B, seq_len − 1]` float32, aligned with `token_log_probs`. Rows are **right-padded** `[prompt][completion][pad]`. Scored index `j` refers to input index `j+1`, so `mask[i, j] = 1` iff `prompt_len[i] ≤ j + 1 < prompt_len[i] + completion_len[i]`. Completion length counts through the first end token (inclusive), as `generation.completion_length` does. |

## `src/grpo_dapo/advantage.py`

| Function | Signature | Behavior |
| --- | --- | --- |
| `group_advantages` | `(rewards: Tensor, eps: float = 1e-4) -> Tensor` | `[N, G]` → `[N, G]`: `(R − row mean) / (row std + eps)`, std with Bessel's correction (n − 1). Requires `G ≥ 2` (raise `ValueError` otherwise). A row whose rewards are all equal gives all zeros. |
| `zero_variance_groups` | `(rewards: Tensor) -> Tensor` | `[N, G]` → `[N]` bool: `True` where every reward in the row is equal (Â = 0, no learning signal). |

With n − 1, the values differ from the ÷n examples in the vault notes by a factor of √(G/(G−1)) (≈ 1.07 for G = 8); the direction and the "rare outcomes get large advantages" pattern are the same.

## `src/grpo_dapo/loss.py`

```text
grpo_loss(logp_new, logp_old, advantages, mask, *,
          eps_low=0.2, eps_high=0.2, beta=0.0, logp_ref=None,
          aggregation="sample", denominator=None) -> (loss: Tensor, stats: dict[str, float | None])
```

Shapes: `logp_new`, `logp_old`, `logp_ref`, `mask`: `[B, T]`; `advantages`: `[B]` (one per completion, shared by its tokens). Only `logp_new` carries gradient.

1. `ratio = exp(logp_new − logp_old)`
2. `objective = min(ratio · Â, clamp(ratio, 1 − eps_low, 1 + eps_high) · Â)`
3. `k3 = exp(logp_ref − logp_new) − (logp_ref − logp_new) − 1` (when `logp_ref` is given)
4. `per_token_loss = −objective + beta · k3` (the KL term only when `beta > 0`)
5. Aggregate over masked tokens:
   - `"sample"` (GRPO): `Σᵢ [ Σₜ ℓᵢₜ mᵢₜ / max(Σₜ mᵢₜ, 1) ] / D`, with `D = denominator` or, if `None`, the number of completions `B`
   - `"token"` (DAPO): `Σᵢₜ ℓᵢₜ mᵢₜ / D`, with `D = denominator` or, if `None`, the number of masked tokens `Σ m`

**Why `denominator` exists: gradient accumulation.** A mini-batch is processed as several micro-batches whose losses are added before one optimizer step. Each micro-batch must divide by the **mini-batch's** total (completions for `"sample"`, completion tokens for `"token"`), not its own; otherwise accumulated gradients don't equal the full-batch gradient. (A version of this bug shipped in the Hugging Face Trainer and was fixed in late 2024.)

**Stats** (Python floats, computed without gradient, over masked tokens only):

| Key | Definition |
| --- | --- |
| `clip_low_fraction` | share of tokens with `Â < 0` and `ratio < 1 − eps_low` |
| `clip_high_fraction` | share of tokens with `Â > 0` and `ratio > 1 + eps_high` |
| `clip_fraction` | the sum of the two |
| `ratio_mean`, `ratio_min`, `ratio_max` | over masked tokens |
| `max_abs_log_ratio` | `max |logp_new − logp_old|`: the invariant, ≈ 0 at the first update after a rollout |
| `kl_mean` | mean k3 over masked tokens if `logp_ref` is given (logged even when `beta = 0`), else `None` |

Raise `ValueError` if `beta > 0` and `logp_ref is None`, if `aggregation` is not `"sample"` or `"token"`, or if shapes don't match.

## Tests

Written first, in `tests/test_logprobs.py`, `tests/test_advantage.py`, `tests/test_loss.py`. Module stubs with these signatures and `raise NotImplementedError` let the tests import and fail one by one.

### `token_log_probs` / `token_entropy` / `completion_mask`

| Case | Expected |
| --- | --- |
| random logits `[2, 5, 7]` (float64), random ids | equals `log_softmax(logits[:, :-1]).gather(ids[:, 1:])` within 1e-6 |
| bfloat16 logits | output dtype float32, shape `[B, T−1]` |
| logits where position `t` puts a huge score on `input_ids[t+1]` | all log-probs ≈ 0 (alignment: an off-by-one implementation gives large negative values) |
| uniform logits over V = 7 | entropy = log 7 everywhere |
| one dominant logit per position | entropy ≈ 0 |
| `prompt_lengths=[3]`, `completion_lengths=[4]`, `seq_len=10` (the `[p1 p2 p3 a1 a2 a3 <|im_end|> pad pad pad]` example) | `[[0, 0, 1, 1, 1, 1, 0, 0, 0]]` |
| two rows with different prompt and completion lengths | each row's mask covers exactly its completion |

### `group_advantages` / `zero_variance_groups`

| Rewards | Expected |
| --- | --- |
| `[[1, 0, 0, 1]]` | `±0.5 / (0.57735 + 1e-4)` = `[0.8659, −0.8659, −0.8659, 0.8659]` (4 d.p.) |
| `[[1, 0, 0, 0]]` | `[0.75, −0.25, −0.25, −0.25] / (0.5 + 1e-4)` |
| `[[1, 1, 1, 1]]`, `[[0, 0, 0, 0]]` | all zeros; `zero_variance_groups` = `[True]` |
| random rewards `[16, 8]` | equals the TRL formula (float64 reference in the test) |
| `G = 1` | `ValueError` |

### `grpo_loss`

Worked example from the vault note *Policy Ratio and Clipping*: two completions (`is 5`, Â = +1; `is 6`, Â = −1), two tokens each, all masked; `logp_old = log[[.9, .3], [.9, .6]]`.

| Case | `logp_new` | Expected |
| --- | --- | --- |
| step 2, sample | `log[[.9, .345], [.9, .555]]` | loss = −0.05625; all clip fractions 0 |
| step 3, sample | `log[[.9, .39], [.9, .51]]` | loss = −0.0875 (`"5"` clipped at 1.2); `clip_high_fraction` = 0.25, `clip_low_fraction` = 0 |
| step 3, `eps_high = 0.28` | same | loss = −0.1075 (`"5"` clipped at 1.28) |
| ρ = 1 everywhere | `= logp_old` | `max_abs_log_ratio` = 0; gradient w.r.t. each masked `logp_new` token = `−Âᵢ / (|oᵢ| · B)` (plain policy gradient) |
| clipped tokens | step 3 | gradient of `"5"` is exactly 0; gradient of `"6"` (ρ = 0.85, not clipped) is non-zero |
| harmful-direction tokens | Â = +1 with ρ = 0.7; Â = −1 with ρ = 1.3 | non-zero gradient (never clipped) |
| KL | tokens with (π_θ, π_ref) = (0.4, 0.2), (0.2, 0.4), (0.1, 0.3) | k3 = 0.1931, 0.3069, 0.9014; `kl_mean` = their mean; `beta = 0.04` adds `0.04 × k3` per token before aggregation |
| unequal lengths | completions of 1 and 3 masked tokens | `"sample"` and `"token"` give the hand-computed (different) values |
| accumulation | 4 completions split into 2 micro-batches | sum of micro-batch losses with the mini-batch `denominator` equals the full-batch loss, values and gradients, for both aggregations |
| reference | random inputs (float64) | equals a float64 transcription of TRL 1.14.1's `grpo` (sample) and `dapo` (token) branches within 1e-10 |
| errors | `beta > 0` without `logp_ref`; `aggregation="x"` | `ValueError` |

## Implementation notes (for the hand-written code)

- Pure `torch`; no model, no file I/O. Do the math in float32 (`.float()` on logits before `logsumexp`).
- Useful operations: `torch.gather(x, dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)`, `torch.logsumexp`, `torch.softmax`, `torch.clamp`, `torch.minimum`, masked sums (`(x * mask).sum(...)`).
- No in-place operations on tensors that need gradient. Compute `stats` under `torch.no_grad()` and return Python floats.
- Type hints and a one-line docstring per function.

## Not in 3a (comes in 3b)

`policy.py` (LoRA, generation, scoring, adapter-off reference), `trainer.py`, `config.py`, `evaluation.py`, `scripts/`, run metadata, W&B logging, entropy and diagnostics wiring.
