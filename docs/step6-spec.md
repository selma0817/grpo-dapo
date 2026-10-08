# Step 6 spec: DAPO switches (Dynamic Sampling, Overlong Reward Shaping)

Adds the two DAPO techniques that don't exist yet, plus the presets for the round-2 ablations. Clip-Higher (`eps_high`) and Token-level Loss (`aggregation="token"`) already exist. **The `train_step` changes are written by hand**: this spec gives their contract. Everything else is implemented from this spec.

## Decisions

| | Choice |
| --- | --- |
| Dynamic Sampling filter | **accuracy**, as in DAPO: a group is informative iff 0 < number correct < G. Not shaped-reward variance (rayyy's choice). |
| Over-sampling | rounds of `questions_per_step` new questions, at most `dynamic_max_rounds = 4` rounds per step; stop as soon as the batch has `questions_per_step` informative groups |
| Too many informative groups | keep the first `questions_per_step` in generation order, discard the rest |
| Too few after the last round | top up with filtered groups (in generation order) so batch shapes stay fixed; **their advantages are set to 0**, so they are pure padding and never train, even when a shaping term varies within them; log how many |
| Overlong penalty | soft punishment over the last `overlong_cache` tokens; `overlong_cache = 64` in the presets (penalty starts at token 192 of 256) |
| Penalty scale | `overlong_penalty_factor = 0.5`: with 0/1 correctness this makes our reward exactly (R_DAPO + 1) / 2, so advantages equal DAPO's (±1 correctness + full penalty) |
| Overlong filtering | a separate switch, `overlong_filter`, off in every preset: truncated answers are masked out of the loss |
| Training metrics | quality metrics (accuracy, format, length, group shares) over **all generated groups**; kept-batch stats under `dynamic/` |

## Design rules

1. **Switches off = current behavior, exactly.** With `dynamic_sampling=False`, `overlong_cache=0` and `overlong_filter=False`, a run must produce the same rewards, advantages, partition and losses as the current code for the same seed (tested).
2. **Score only what you train on.** Filtering happens before `logp_old` / `logp_ref` scoring; discarded groups are never scored.
3. **Every generated sample is counted.** `cost/samples_generated_total` and `cost/tokens_generated_total` include discarded and filtered groups, so Dynamic Sampling can be compared at equal generation cost.
4. **The policy doesn't change between rounds,** so every round's samples are on-policy and `logp_old` is still the behavior policy.
5. **Advantages are computed from the shaped reward over the full group** (truncated answers included). Overlong filtering only masks the loss; it doesn't remove answers from the group mean and std.
6. **A filtered group never affects the gradient.** The filter judges accuracy but advantages follow the shaped reward, so top-up groups get their advantages zeroed explicitly.

## `src/grpo_dapo/config.py`

New fields (group "dapo"), with defaults that keep current behavior:

```text
dynamic_sampling: bool = False
dynamic_max_rounds: int = 4
overlong_cache: int = 0              # L_cache in tokens; 0 = no length penalty
overlong_penalty_factor: float = 0.5
overlong_filter: bool = False
```

Validation (`ValueError`): `dynamic_max_rounds >= 1`; `0 <= overlong_cache < max_new_tokens`; `overlong_penalty_factor >= 0`.

New presets:

```text
dynamic_sampling: {"dynamic_sampling": True}
overlong:         {"overlong_cache": 64}
dapo:             {"eps_high": 0.28, "aggregation": "token", "dynamic_sampling": True, "overlong_cache": 64}
```

`overlong_cache = 64` assumes `max_new_tokens = 256`. Update the README's preset list.

## `src/grpo_dapo/reward.py`

```python
def overlong_penalty(num_tokens: int, truncated: bool, max_new_tokens: int, cache: int) -> float:
```

- `cache == 0` → `0.0`
- `truncated` → `-1.0`
- `num_tokens <= max_new_tokens - cache` → `0.0`
- otherwise → `-(num_tokens - (max_new_tokens - cache)) / cache`

Range [−1, 0]. `num_tokens` is `Completion.num_tokens` (generated tokens through the end token). An answer that ends exactly at `max_new_tokens` is not truncated and gets −1 from the formula, so the penalty is continuous at the limit. `ValueError` for `cache < 0`, `cache >= max_new_tokens`, or `num_tokens < 0`.

## `src/grpo_dapo/rollout.py`

Split `collect_rollout` into two stages so the trainer can loop between them.

**`generate_groups(policy, examples, config) -> GroupBatch`**: generate, check prompt lengths (existing `max_prompt_tokens` error), and compute per answer, all `[Q, G]` tensors:
- `correct` (0/1), `boxed` (0/1), `penalty` (`overlong_penalty`, raw, in [−1, 0]), `truncated` (bool)
- `rewards = correct + format_weight * boxed + overlong_penalty_factor * penalty`

`GroupBatch` holds `examples`, `completions` and these tensors, and supports selecting a subset of groups by index and concatenating batches.

**`build_rollout(policy, groups, config, generator, padding=None) -> RolloutBatch`**: everything `collect_rollout` does after rewards: advantages and zero-variance flags from `groups.rewards`, `build_micro_batches`, scoring `logp_old` / `logp_ref` / entropy, and the kept-batch stats. `padding` is an optional `[Q]` bool tensor marking top-up groups; their advantages are set to 0 after `group_advantages` and before partitioning.

**`collect_rollout`** stays as `build_rollout(policy, generate_groups(policy, examples, config), config, generator)`, so existing callers and tests keep working.

**Overlong filtering** (only when `config.overlong_filter`): `build_micro_batches` takes an optional `[Q, G]` bool `loss_rows` (default all True). `MicroBatch` gains `loss_mask = completion_mask * loss_rows[row]`. `completion_mask` is unchanged and still used for scoring, entropy and diagnostics. With the switch off, `loss_mask` equals `completion_mask`.

## `src/grpo_dapo/trainer.py`

Replace the single `stream.take` + `collect_rollout` with:

```text
generated = []                       # every GroupBatch generated this step
for round in 1..(dynamic_max_rounds if dynamic_sampling else 1):
    examples, epoch_of_round = stream.take(questions_per_step)
    generated.append(generate_groups(policy, examples, config))
    if not dynamic_sampling: break
    if informative groups across generated >= questions_per_step: break
kept = first questions_per_step informative groups (generation order)
       + filtered groups in generation order if short (topped_up = how many)
       (without dynamic_sampling: kept = the single round's groups, unfiltered)
rollout = build_rollout(policy, kept, config, generator, padding=<True for top-up groups>)
update  = train_step(policy, optimizer, rollout, config)
```

`epoch` in the metrics is the epoch of the first round. Discarded questions are consumed from the stream and not reused.

`samples.jsonl` logs groups from the kept batch, as now.

## `train_step` (written by hand)

Only when the loss mask differs from the completion mask, i.e. `overlong_filter` on:
- pass `mask=micro_batch.loss_mask` to `grpo_loss`
- token aggregation: denominator = sum of `loss_mask` over the mini-batch
- sample aggregation: denominator = number of rows in the mini-batch with at least one `loss_mask` token
- if a mini-batch's denominator is 0 (every answer in it was filtered), skip its backward and optimizer step and count it in a new returned stat `skipped_minibatches`

## Metrics

Computed over **all generated groups this step** (all rounds, kept and discarded):
`reward/mean`, `reward/accuracy`, `reward/format_rate`, `reward/unparseable_rate`, `groups/all_correct`, `groups/all_wrong`, `groups/mixed`, `groups/nonzero_variance`, `length/mean`, `length/max`, `length/truncation_rate`, and new `reward/overlong_penalty_mean` (raw penalty, before the factor).

New, every step:

| Key | Meaning |
| --- | --- |
| `dynamic/rounds` | generation rounds used (1 when off) |
| `dynamic/questions_generated` | questions generated this step |
| `dynamic/keep_rate` | informative groups ÷ generated groups |
| `dynamic/topped_up_groups` | filtered groups used to fill the batch |
| `dynamic/kept_accuracy` | accuracy over the trained batch (biased toward the middle by construction; not a quality metric) |
| `overlong/masked_rate` | share of trained answers masked by `overlong_filter` |
| `optim/skipped_minibatches` | from `train_step` |
| `cost/questions_used_total` | cumulative questions drawn from the stream |

`policy/entropy` stays as now: it comes from scoring, so it covers **the trained batch only**. With Dynamic Sampling on, that batch is enriched in harder, mixed questions, so its entropy isn't directly comparable with vanilla's. Note this in the metric's docstring rather than paying to score discarded groups.

`cost/samples_generated_total` and `cost/tokens_generated_total` count every generated sample (rule 3).

## Tests

1. `overlong_penalty` (L_max 256, cache 64): 100 → 0; 192 → 0; 224 → −0.5; 256 finished → −1; truncated → −1; cache 0 → 0; invalid arguments raise.
2. **The DAPO-scale identity:** for one group with answers {correct 100 tokens, correct 224, wrong 150, wrong 224, truncated}, our rewards with factor 0.5 equal `(R_DAPO + 1) / 2` (R_DAPO = ±1 + full penalty), and `group_advantages` gives the same advantages for both.
3. **Off = unchanged:** with every new switch off, `collect_rollout` and one `train_step` give exactly the same rewards, advantages, partition, loss and updated weights as before, for a fake policy and a fixed seed.
4. **Dynamic Sampling, normal case** (fake policy with scripted correctness, Q = 2, G = 2): round 1 has one all-correct and one mixed group, round 2 one mixed and one all-wrong. Two rounds; the kept batch is exactly the two mixed groups; the fake policy's `score` sees only kept rows; `dynamic/keep_rate` = 0.5; `dynamic/questions_generated` = 4; `topped_up` = 0.
5. **Excess:** a round with more informative groups than needed keeps the first ones in generation order.
6. **Shortfall:** with `overlong_cache = 64`, every group is all-wrong with varied lengths (so its shaped rewards differ) for `dynamic_max_rounds` rounds; the batch is topped up to Q with these groups; `topped_up_groups` = Q; their advantages are all 0 despite the reward variance; the update leaves the weights unchanged.
7. **Accounting:** `cost/samples_generated_total` after one step equals all generated samples, discarded groups included.
8. **Overlong filter:** truncated rows have `loss_mask` 0 and an unchanged `completion_mask`; a hand-computed `train_step` example matches; an all-masked mini-batch is skipped and counted.
9. Config: the three presets resolve to the fields above; the validation errors fire.

## Out of scope

Filtering on reward variance; entropy regularization (rayyy's extra preset); MATH; any change to evaluation.
