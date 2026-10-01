# M0 spec: baseline evaluation on GSM8K

M0 measures the **untrained** model before any GRPO training: how accurate it is, whether it uses the `\boxed{}` format, how often it runs out of tokens, and, most importantly, **how many questions would give GRPO a learning signal** (groups whose answers are neither all correct nor all wrong). The per-question success rates computed here are reused later for difficulty buckets (M2).

## Settings

| Setting | Value | Why |
| --- | --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (CLI flag; 1.5B later) | |
| Data | GSM8K (`openai/gsm8k`, config `main`), **test** split, all 1,319 questions | full test set, so the result is reportable |
| System prompt | `Please reason step by step, and put your final answer within \boxed{}.` | Qwen's recommended math prompt; matches the reward's required format |
| Greedy run | 1 completion per question, `do_sample=False` | reproducible number to compare with papers |
| Sampled run | **8** completions per question, temperature **1.0**, top-p **1.0**, top-k **disabled**, repetition penalty **1.0** | exactly what GRPO training will sample |
| `max_new_tokens` | 512 (CLI flag) | raise it if the truncation rate is above 2% |
| Generation | Hugging Face `generate`, batched, **left padding** | simple; vLLM comes later with the training loop |
| Seed | 0 (CLI flag) | |

> **Override every sampling parameter explicitly.** `Qwen2.5-Instruct`'s `generation_config.json` defaults to temperature 0.7, top-p 0.8, top-k 20 and repetition penalty 1.05. Relying on defaults would silently measure a different distribution from the one GRPO trains on. Record the settings actually passed to `generate` in `summary.json`.

## Metrics

Reward functions come from `grpo_dapo.reward`: a completion is **correct** if `compute_reward(completion, gold) == 1.0`.

| Metric | Definition | Runs |
| --- | --- | --- |
| Accuracy | share of completions that are correct | greedy |
| pass@k, k ∈ {1, 2, 4, 8} | per question `1 − C(n−c, k) / C(n, k)` (n = 8 samples, c correct), averaged over questions; `1.0` when `n − c < k` | sampled |
| Format rate | share of completions where `extract_answer` is not `None` | both |
| Unparseable rate | share of completions where `extract_answer` is not `None` but `parse_number` of it is `None` (boxed, but not a single number) | both |
| Truncation rate | share of completions that used all `max_new_tokens` without producing an end-of-sequence token | both |
| Response length | completion tokens (up to and including the first end-of-sequence token): mean, median, 90th percentile, max | both |
| **Group statistics** | per question c = number correct out of 8: share of questions with **c = 8** (all correct), **c = 0** (all wrong), **0 < c < 8** (mixed); histogram of c (0–8); mean of p(1−p) with p = c/8 | sampled |

Mixed groups are the only ones with a non-zero GRPO advantage at step 0. p(1−p) is the variance of a group's 0/1 rewards; it is largest at p = 0.5.

## Modules

| File | Contents |
| --- | --- |
| `src/grpo_dapo/data.py` | `Example` dataclass: `id: int`, `question: str`, `gold: str`, `gold_number: Fraction`. `build_examples(rows) -> list[Example]`: pure function, no network; for each row (`question`, `answer` fields) calls `extract_gold` and `parse_number`, and **raises `ValueError` listing every row id whose gold fails to extract or parse**. `load_gsm8k(split, limit=None) -> list[Example]`: loads with `datasets`, then calls `build_examples` |
| `src/grpo_dapo/prompts.py` | `SYSTEM_PROMPT` constant. `build_messages(question) -> list[dict]`: `[{"role": "system", ...}, {"role": "user", "content": question}]` |
| `src/grpo_dapo/generation.py` | `load_model_and_tokenizer(name, device)`: tokenizer with `padding_side="left"`; bf16 on CUDA, float32 otherwise. `generate(model, tokenizer, prompts, *, num_samples, do_sample, temperature, top_p, max_new_tokens, batch_size) -> list[list[Completion]]`, grouped per prompt in input order. Prompts are built with `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`. `Completion`: `text` (decoded new tokens, special tokens skipped), `num_tokens`, `truncated`. Pure helpers (testable without a model): `completion_length(new_token_ids, eos_ids) -> int` and `is_truncated(new_token_ids, eos_ids, max_new_tokens) -> bool` |
| `src/grpo_dapo/metrics.py` | `pass_at_k(n, c, k) -> float`; `group_stats(correct_counts, n) -> dict`; `summarize(records) -> dict` for format / unparseable / truncation rates and length statistics |
| `src/grpo_dapo/eval_baseline.py` | CLI (`python -m grpo_dapo.eval_baseline`): loads data and model, runs greedy then sampled generation, scores every completion, writes outputs, prints the summary |

**End-of-sequence ids** come from the model's generation config (for Qwen2.5-Instruct: `<|im_end|>` and `<|endoftext|>`); the pad token `<|endoftext|>` is one of them, so the length of a completion is "up to and including the first EOS id".

**New tokens** are `output_ids[:, prompt_length:]` (with left padding, every prompt in a batch ends at the same column). With `num_return_sequences=8`, outputs come back as 8 consecutive rows per prompt.

## CLI

```text
--model            default Qwen/Qwen2.5-0.5B-Instruct
--split            default test
--limit            default None (all questions); e.g. 8 for a smoke test
--num-samples      default 8
--temperature      default 1.0
--top-p            default 1.0
--max-new-tokens   default 512
--batch-size       default 32 (prompts per generate call)
--seed             default 0
--skip-greedy      flag
--output-dir       default outputs/m0/<model-name>_<split>_<timestamp>
```

## Outputs (in `--output-dir`, which is git-ignored under `outputs/`)

- `summary.json`: model, dataset and split, number of questions, system prompt, the exact generation settings of each run, all metrics above, runtime in seconds, and versions of `torch` and `transformers`.
- `completions_greedy.jsonl` and `completions_sampled.jsonl`: one line per completion: `question_id`, `sample_index`, `completion`, `num_tokens`, `truncated`, `extracted_answer`, `parsed_answer` (string, or `null`), `gold`, `correct`.

## Tests (CPU only, no network, no model download)

### `pass_at_k`

| n, c, k | Expected | Checks |
| --- | --- | --- |
| 8, 3, 1 | `0.375` | pass@1 = c/n |
| 8, 3, 4 | `1 − 5/70` | standard estimator |
| 8, 3, 2 | `1 − 10/28` | standard estimator |
| 8, 0, 4 | `0.0` | never correct |
| 8, 8, 1 | `1.0` | always correct |
| 8, 5, 4 | `1.0` | n − c < k: every subset of 4 contains a correct answer |

### `group_stats`

| correct_counts (n = 8) | Expected | Checks |
| --- | --- | --- |
| `[8, 0, 4, 1]` | all-correct 0.25, all-wrong 0.25, mixed 0.5 | classification |
| `[4]` | mean p(1−p) = `0.25` | maximal variance at p = 0.5 |
| `[0, 8]` | mean p(1−p) = `0.0` | no signal from uniform groups |
| `[8, 0, 4, 1]` | histogram has `{0: 1, 1: 1, 4: 1, 8: 1}` and zeros elsewhere | histogram over 0–8 |

### `completion_length` / `is_truncated` (EOS ids `{2}`, `max_new_tokens = 5`)

| new_token_ids | Length | Truncated | Checks |
| --- | --- | --- | --- |
| `[7, 8, 2, 2, 2]` | 3 | False | stops at the first EOS; trailing padding not counted |
| `[7, 8, 9, 10, 11]` | 5 | True | used every token without EOS |
| `[7, 8, 9, 10, 2]` | 5 | False | EOS on the last allowed token is a normal stop |
| `[2, 2, 2, 2, 2]` | 1 | False | empty answer: immediate EOS |

### `build_examples`

| Rows | Expected | Checks |
| --- | --- | --- |
| two rows with `#### 72` and `#### 1,000` | two `Example`s with `gold_number` 72 and 1000 | normal case |
| one good row and one row with no `####` | raises `ValueError` naming the bad row id | gold validated at load time |
| a row whose gold is `#### twelve` | raises `ValueError` | gold extracts but doesn't parse |

### `summarize`

Synthetic records covering: a correct boxed answer, a wrong boxed answer, an unboxed answer, a boxed unparseable answer (`\boxed{12 or 72}`), and a truncated completion. Check format rate, unparseable rate, truncation rate and accuracy against hand-computed values.

### `build_messages`

Returns exactly two messages, system first with `SYSTEM_PROMPT`, then the user's question unchanged.

## Implementation requirements

- **Dependencies:** add `torch`, `transformers` and `datasets` with `uv add`. Nothing else.
- **No network or model in tests.** Keep pure logic (`build_examples`, `pass_at_k`, `group_stats`, `summarize`, `completion_length`, `is_truncated`) separate from I/O so it can be tested offline.
- **Set every sampling parameter explicitly** on each `generate` call (`do_sample`, `temperature`, `top_p`, `top_k=0`, `repetition_penalty=1.0`, `max_new_tokens`, `num_return_sequences`, `eos_token_id`, `pad_token_id`). Never rely on the model's `generation_config` defaults.
- **Left padding** for generation; compute new tokens as `output_ids[:, prompt_length:]`.
- Scoring uses only `grpo_dapo.reward`; don't reimplement answer parsing.
- Device: CUDA if available, then Apple MPS, then CPU. Print the device used.
- Do not modify `reward.py` or its tests.

## How to run

Smoke test (any machine; downloads the ~1 GB model on first run):

```bash
uv run python -m grpo_dapo.eval_baseline --limit 8 --num-samples 4
```

Full baseline (on the 4070 Ti):

```bash
uv run python -m grpo_dapo.eval_baseline
```
