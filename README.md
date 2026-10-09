# grpo-dapo

GRPO and DAPO for math reasoning on **Qwen2.5-0.5B-Instruct** with LoRA, implemented from scratch in PyTorch + Transformers + PEFT, with a controlled ablation of DAPO's four techniques on GSM8K. Everything runs on a single 16 GB consumer GPU.

- **Every DAPO technique is an independent switch** (a config field and a preset), so each run adds exactly one change to vanilla GRPO.
- **Faithful where it matters:** several updates per rollout (so Clip-Higher can act at all), DAPO's accuracy filter for Dynamic Sampling, and a length penalty scaled so our advantages are *exactly* DAPO's.
- **Measured carefully:** final evaluation on all 1,319 GSM8K test questions, paired standard errors between runs, pass@k with the unbiased estimator, and predictions written down before every run.
- **208 tests**, including hand-computed loss values, gradient-accumulation equivalence, and a bit-for-bit check that new switches leave old runs unchanged.

## Results

### DAPO ablation (256-token budget, seed 0)

The table is generated from the run directories, never typed by hand:

```bash
uv run python scripts/ablation_table.py results/train/{vanilla,clip_higher,dynamic_sampling,token_level,overlong,dapo}_256_s0 --baseline results/train/vanilla_256_s0
```

| Run | Clip-Higher | Dynamic Sampling | Token-level | Overlong | Greedy acc. | Δ vs vanilla_256_s0 | pass@1 | pass@8 | Greedy format | Greedy trunc. | Mean length | Samples generated | Runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla_256_s0 |  |  |  |  | 49.7% | — | 46.5% | 74.8% | 89.5% | 11.4% | 175 | 25,600 | 75 min |
| clip_higher_256_s0 | ✓ |  |  |  | 51.6% | +1.9 ± 1.9† | 46.8% | 75.9% | 90.8% | 9.5% | 173 | 25,600 | 75 min |
| dynamic_sampling_256_s0 |  | ✓ |  |  | 51.2% | +1.4 ± 1.9† | 47.7% | 74.1% | 86.1% | 14.7% | 186 | 56,832 | 98 min |
| token_level_256_s0 |  |  | ✓ |  | 49.7% | +0.0 ± 1.9† | 47.8% | 75.1% | 89.8% | 10.5% | 174 | 25,600 | 75 min |
| overlong_256_s0 |  |  |  | ✓ | 48.1% | -1.7 ± 1.9† | 46.5% | 72.7% | 97.9% | 2.0% | 130 | 25,600 | 71 min |
| dapo_256_s0 | ✓ | ✓ | ✓ | ✓ | 53.4% | +3.6 ± 1.9† | 49.3% | 76.5% | 95.8% | 4.3% | 155 | 57,856 | 98 min |

Final evaluation on all GSM8K test questions (greedy, plus 8 samples at T = 1 for pass@k). Δ is in percentage points with one standard error, paired over the same questions; † marks an unpaired standard error (per-question records not available).

### Earlier runs (full GSM8K test set)

| Run | Budget | Greedy acc. | pass@1 | pass@8 | Format (sampled) | Truncation (sampled) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct, untrained | 512 | 47.5% | 31.5% | 68.1% | 77.9% | 9.9% |
| Vanilla GRPO, 100 steps (`vanilla_s0`) | 512 | 51.6% | 49.8% | 77.3% | 96.7% | 3.5% |
| Untrained | 256 | 23.0% | 16.1% | 39.5% | — | — |

At 512 tokens most of GRPO's gain is *sharpening*: sampled pass@1 rose 18 points while greedy rose 4, and entropy fell 76%. At 256 tokens two thirds of the untrained model's answers are cut off before they reach `\boxed{}`, so learning to finish in time is the main signal, which is why the ablation uses this budget.

## Setup

| | |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, bf16 |
| LoRA | all 7 projections (`q k v o gate up down`), r = 16, α = 32, dropout 0 |
| Data | GSM8K train (7,473) for training, test (1,319) for evaluation |
| Reward | 1 if the last `\boxed{}` equals the gold answer exactly (as a fraction), else 0 |
| Rollout | 32 questions × 8 samples per step; temperature 1, top-p 1, top-k off, repetition penalty 1 (all set explicitly) |
| Update | 4 mini-batches of 64 per rollout (4 optimizer steps), micro-batches of 8; AdamW, learning rate 1e-5, gradient clip 1.0 |
| Regularization | β = 0 (KL to the reference is still logged); the reference model is the same model with the adapter disabled |
| Evaluation | every 10 steps on 200 fixed test questions; at the end on all 1,319 (greedy + 8 samples) |
| Hardware | one RTX 4070 Ti SUPER (16 GB) |

## Method

### GRPO

For each question, sample a group of $G$ answers from the current policy, score them, and use each answer's reward relative to its group as the advantage of every one of its tokens:

$$
\hat{A}_i = \frac{R_i - \operatorname{mean}(R_1, \dots, R_G)}{\operatorname{std}(R_1, \dots, R_G) + 10^{-4}},
\qquad
\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}
$$

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\Big(\rho_{i,t}\hat{A}_i,\ \operatorname{clip}\big(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon\big)\hat{A}_i\Big) - \beta\, \hat{\mathbb{D}}_{\text{KL}}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)\right]
$$

| Piece | Code |
| --- | --- |
| Per-token log-probs (shift by one, fp32 log-softmax), completion mask, entropy | [`logprobs.py`](src/grpo_dapo/logprobs.py): `token_log_probs`, `completion_mask`, `token_entropy` |
| Group-relative advantage (Bessel-corrected std) | [`advantage.py`](src/grpo_dapo/advantage.py): `group_advantages` |
| Clipped objective, k3 KL estimator, sample/token aggregation | [`loss.py`](src/grpo_dapo/loss.py): `grpo_loss` |
| Mini-batch updates with gradient accumulation | [`trainer.py`](src/grpo_dapo/trainer.py): `train_step` |
| Reference log-probs with the LoRA adapter disabled | [`policy.py`](src/grpo_dapo/policy.py): `Policy.score(use_adapter=False)` |

Two invariants are checked during training: the model trains on the token ids it generated (never re-tokenized text), and at the first update after each rollout $\rho = 1$ exactly (the logged health metric $\max|\log\rho|$ is 0 with dropout off).

### DAPO

DAPO keeps the group-relative advantage, drops the KL term, and changes four things:

$$
\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\Big(\rho_{i,t}\hat{A}_i,\ \operatorname{clip}\big(\rho_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}\big)\hat{A}_i\Big)\right]
\quad \text{s.t.} \quad 0 < \big|\lbrace\, i : o_i \text{ is correct} \,\rbrace\big| < G
$$

#### 1. Clip-Higher (`--preset clip_higher`)

**Problem.** The ratio limits *relative* change, so it binds almost only on rare tokens: with $\varepsilon = 0.2$ a token at $p = 0.01$ can rise to 0.012 per rollout, while one at $p = 0.9$ is never limited. Rare tokens in good answers (the model's alternatives) can't grow, and entropy collapses.

**Change.** Decouple the bounds: $\varepsilon_{\text{low}} = 0.2$, $\varepsilon_{\text{high}} = 0.28$. Only the upper bound widens; a wider lower bound would let one bad answer push a rare token toward zero.

**Code.** `eps_high` in [`config.py`](src/grpo_dapo/config.py), used by `grpo_loss`. Clipping only acts from the second optimizer step on a rollout (at the first, $\rho = 1$), which is why every run takes 4 updates per rollout.

#### 2. Dynamic Sampling (`--preset dynamic_sampling`)

**Problem.** A group whose answers are all correct or all wrong has $\hat{A} = 0$ and contributes no gradient. Their share grows as the model improves (about half the groups late in our 512-token run), so each update averages over fewer informative answers.

**Change.** Keep generating groups for new questions, drop those whose accuracy is 0 or 1, and stop when the batch has 32 informative groups.

**Code.** The round loop in `train` with `_informative` and `_select_training_groups` ([`trainer.py`](src/grpo_dapo/trainer.py)); generation and rewards are split from scoring (`generate_groups` / `build_rollout` in [`rollout.py`](src/grpo_dapo/rollout.py)) so dropped groups are never scored. Rounds of 32 questions, at most `dynamic_max_rounds = 4`; if the batch is still short, it is padded with dropped groups whose advantages are set to 0. Cost counters include every discarded sample, so runs can be compared at equal generation cost, not only equal steps.

#### 3. Token-level loss (`--preset token_level`)

**Problem.** GRPO averages within each answer first, so every answer has the same total weight regardless of length: a token in a 250-token answer gets 5× less weight than one in a 50-token answer. Long rambling or repetitive answers are under-penalized, and long correct reasoning is under-learned.

**Change.** Average over every token in the batch:

$$
\underbrace{\frac{1}{G}\sum_{i}\frac{1}{|o_i|}\sum_{t}\ell_{i,t}}_{\text{sample-level (GRPO)}}
\quad\longrightarrow\quad
\underbrace{\frac{1}{\sum_i |o_i|}\sum_{i}\sum_{t}\ell_{i,t}}_{\text{token-level (DAPO)}}
$$

**Code.** `aggregation="token"` in `grpo_loss`. Under gradient accumulation, `train_step` divides every micro-batch by the *whole mini-batch's* token count, so the accumulated gradient is exactly the mini-batch token mean (tested against a single large micro-batch).

#### 4. Overlong Reward Shaping (`--preset overlong`)

**Problem.** An answer cut off at the token limit never reaches its `\boxed{}`, so it scores the same as a wrong answer, and sound reasoning gets pushed down.

**Change.** A soft length penalty over the last $L_{\text{cache}}$ tokens before the limit $L_{\max}$:

$$
R_{\text{length}}(y) =
\begin{cases}
0, & |y| \le L_{\max} - L_{\text{cache}} \\
\dfrac{(L_{\max} - L_{\text{cache}}) - |y|}{L_{\text{cache}}}, & L_{\max} - L_{\text{cache}} < |y| \le L_{\max} \\
-1, & \text{truncated}
\end{cases}
$$

We use $L_{\max} = 256$, $L_{\text{cache}} = 64$. DAPO adds this to a ±1 correctness reward; with our 0/1 reward the penalty is halved, which makes our reward an exact affine transform of DAPO's, so the advantages are identical (group normalization removes shifts and scales):

$$
R = \mathbb{1}[\text{correct}] + \tfrac{1}{2}R_{\text{length}} = \tfrac{1}{2}\big(R_{\text{DAPO}} + 1\big)
$$

**Code.** `overlong_penalty` in [`reward.py`](src/grpo_dapo/reward.py); `overlong_cache` and `overlong_penalty_factor` in the config. DAPO's other variant, masking truncated answers out of the loss (`overlong_filter`), is built into the rollout but refused by `train` until `train_step` supports it; no preset uses it.

#### All four (`--preset dapo`)

`eps_high = 0.28`, `aggregation = "token"`, `dynamic_sampling = True`, `overlong_cache = 64`.

## Design decisions

The ablation design and the 256-token budget follow [rayyy032/qwen-math-grpo-dapo](https://github.com/rayyy032/qwen-math-grpo-dapo), used as a reference for which experiments to run; this code base is written independently. Where the setups differ:

| | This repo | Reference | Why |
| --- | --- | --- | --- |
| Updates per rollout | 4 | 1 | with one update $\rho = 1$, so Clip-Higher cannot change anything |
| Rollout | 32 questions × 8 | 2 × 8 | enough data per step for differences to exceed noise |
| Reward | 0/1 correctness | correctness + 0.2 for any box | a format bonus pays for boxing wrong answers and overlaps with length shaping |
| Length penalty | window 64, factor 0.5 | window 128, factor 1 | ours equals DAPO's relative strength; theirs is twice it and starts at half the budget |
| Dynamic Sampling filter | accuracy (as in DAPO) | shaped-reward variance | with shaping terms, almost every group has some variance |
| Sampling | full distribution, set explicitly | the model's `generation_config` (top-p 0.8, repetition penalty 1.1) still applies | samples should come from the policy whose log-probs are trained |
| LoRA dropout | 0 | 0.05 | dropout makes $\rho \ne 1$ at the first update |
| Evaluation | 1,319 test questions, paired standard errors | 100 questions | ±5 points of noise on 100 questions |

With the reference's own settings, this code reproduces its GRPO accuracy (33% vs 32% on 100 questions). The remaining format-rate gap is mostly the format bonus: adding it moved our format rate from 47% to 61% (theirs: 76%).

## Reproduce

```bash
uv sync
uv run pytest
```

Baseline evaluation (all test questions, greedy + 8 samples):

```bash
uv run python scripts/eval_baseline.py --max-new-tokens 256
```

The six ablation runs (roughly 9 h on one RTX 4070 Ti SUPER):

```bash
for p in vanilla clip_higher token_level dynamic_sampling overlong dapo; do uv run python scripts/train.py --preset $p --run-name ${p}_256_s0 --max-new-tokens 256 --micro-batch-size 8 || break; done
```

Every config field is also a command-line flag (`--eps-high 0.28`, `--dynamic-sampling`, …); the command line overrides the preset. Runs write to `outputs/train/<run>/`: `run.json` (resolved config, git commit, versions), `metrics.jsonl` (every step), `eval.jsonl` (every 10 steps), `final_eval.json` (per-question records), `summary.json`, and LoRA checkpoints. Summaries worth keeping are copied to `results/`.

Plots and tables:

```bash
uv run python scripts/plot_runs.py results/train/*_256_s0 --out results/plots/ablation_256
uv run python scripts/ablation_table.py results/train/*_256_s0 --baseline results/train/vanilla_256_s0
```

A quick local smoke test (CPU or Apple silicon):

```bash
uv run python scripts/train.py --preset vanilla --max-steps 2 --questions-per-step 2 --samples-per-question 4 --num-minibatches 2 --micro-batch-size 4 --eval-questions 4 --eval-samples 2 --no-final-eval --wandb-mode disabled
```

## Layout

```text
src/grpo_dapo/
├── reward.py       last \boxed{} answer, exact fraction equality, overlong penalty
├── data.py         GSM8K loading with gold answers validated at load time
├── prompts.py      system prompt and chat messages
├── generation.py   batched sampling; completions keep their generated token ids
├── metrics.py      pass@k (unbiased), group statistics
├── evaluation.py   scoring shared by the baseline and in-training evaluation
├── policy.py       LoRA policy: generate, score, reference via disabled adapter
├── logprobs.py     per-token log-probs, completion mask, entropy
├── advantage.py    group-relative advantages, zero-variance groups
├── loss.py         clipped objective, KL (k3), sample/token aggregation
├── rollout.py      generate + reward, fixed micro-batch partition, scoring
├── config.py       Config dataclass, presets, command-line flags
└── trainer.py      train_step and the training loop (dynamic sampling, logging, evaluation)
scripts/            eval_baseline.py, train.py, plot_runs.py, ablation_table.py
docs/               the spec written before each step
results/            kept run summaries and plots
```

## References

- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO), arXiv:2402.03300
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, arXiv:2503.14476
- Hugging Face TRL `GRPOTrainer` (numerical checks) and volcengine/verl `core_algos.py`
- [rayyy032/qwen-math-grpo-dapo](https://github.com/rayyy032/qwen-math-grpo-dapo) (ablation design and replication target)
