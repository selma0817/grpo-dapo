# Reward spec (GSM8K)

The rule-based reward scores one model completion against the gold answer: `1.0` if the model's final answer equals the gold answer, otherwise `0.0`. Every advantage in GRPO comes from this number, so anything the checker gets wrong, the model learns.

## Design decisions

| # | Decision | Choice | Why |
| --- | --- | --- | --- |
| D1 | Required answer format | The final answer goes in `\boxed{...}` (the prompt asks for it) | Standard for math RL; unambiguous to extract; also works for MATH's LaTeX answers later |
| D2 | Correct number, but no `\boxed{}` | Reward `0.0` (strict) | Guessing which number is "the answer" gives noisy rewards and invites hacking. The model learns the format quickly; the format rate is tracked as a metric |
| D3 | Several `\boxed{}` in one completion | **The last one** counts | The last box is the model's final decision. Spamming boxes doesn't help, since only the last one counts |
| D4 | When are two answers equal? | **Same number**, compared exactly (`fractions.Fraction`) after cleanup | `18 = 18.0 = 18.00`, `1,000 = 1000`, with no floating-point surprises |

Smaller defaults (change any of them, but update the tests to match):

- **Bad data crashes, bad model output scores 0.** A gold answer without `####` is a dataset bug, so `extract_gold` raises `ValueError`. A model completion without a usable answer just gets `0.0`.
- `extract_answer` returns the box content **stripped of surrounding whitespace**, and returns `None` for an empty box, no box, or an unclosed box (output cut off at the length limit).
- `extract_answer` matches **balanced braces**, so `\boxed{\text{18}}` gives `\text{18}`, not `\text{18`.
- `parse_number` ignores `$`, `\$`, `%`, thousands commas, whitespace and non-numeric text, then requires **exactly one number**: zero numbers or two or more numbers give `None`.
- Fractions (`1/2`, `\frac{1}{2}`) and scientific notation are **out of scope** until the MATH dataset is added; `parse_number` returns `None` for them for now.

## Functions (all in `src/grpo_dapo/reward.py`)

```text
GSM8K "answer" field ──extract_gold──▶ "72" ──────────┐
                                                      ├─▶ parse_number on both ─▶ equal? ─▶ 1.0 / 0.0
model completion ──extract_answer──▶ "72" or None ────┘
                  └──────────────────── compute_reward (ties the steps together) ────────────┘
```

| Function | Signature | Job |
| --- | --- | --- |
| `extract_gold` | `(answer_field: str) -> str` | Return the text after `####`, stripped. Raise `ValueError` if there is no `####`. |
| `extract_answer` | `(completion: str) -> str \| None` | Return the content of the **last** complete `\boxed{...}`, stripped. `None` if there is none, or it is empty. |
| `parse_number` | `(text: str) -> Fraction \| None` | Clean up the text and return its single number exactly. `None` if it doesn't contain exactly one number. |
| `compute_reward` | `(completion: str, gold: str) -> float` | `gold` is the output of `extract_gold`. Return `1.0` if both parse to the same number, else `0.0`. |

## Test cases

### `extract_gold`

| Input | Expected | Checks |
| --- | --- | --- |
| `"Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"` | `"72"` | real GSM8K example; numbers before `####` are ignored |
| `"...\n####72"` | `"72"` | no space after `####` |
| `"...\n#### 72\n"` | `"72"` | trailing newline / whitespace |
| `"...\n#### 1,000"` | `"1,000"` | returns raw text; cleanup is `parse_number`'s job |
| `"...\n#### -3"` | `"-3"` | negative gold |
| `"no marker here"` | raises `ValueError` | dataset bug fails loudly |

### `extract_answer`

| Input | Expected | Checks |
| --- | --- | --- |
| `"... so the answer is \boxed{72}."` | `"72"` | normal case |
| `"\boxed{ 72 }"` | `"72"` | surrounding whitespace stripped |
| `"First \boxed{70}. Wait, recheck: \boxed{72}"` | `"72"` | several boxes: last one (D3) |
| `"\boxed{72} ... actually \boxed{70}"` | `"70"` | last one, even if the earlier box was right |
| `"The answer is 72."` | `None` | no box (D2) |
| `"\boxed{}"` | `None` | empty box |
| `"\boxed{   }"` | `None` | whitespace-only box |
| `"... so the total is \boxed{7"` | `None` | unclosed box: output cut off |
| `"\boxed{\text{18}}"` | `"\text{18}"` | balanced braces |
| `"\boxed{18 \text{ dollars}}"` | `"18 \text{ dollars}"` | text inside the box is kept; parsing handles it |
| `"\boxed{1} \boxed{2} \boxed{3} ... \boxed{100}"` | `"100"` | box spam: only the last counts |

### `parse_number`

| Input | Expected | Checks |
| --- | --- | --- |
| `"72"` | `72` | plain integer |
| `"1,000"` | `1000` | thousands comma |
| `"1,000,000"` | `1000000` | several commas |
| `"$18"` | `18` | dollar sign |
| `"\$18"` | `18` | LaTeX-escaped dollar |
| `"25%"` | `25` | percent sign |
| `"18.0"` | `18` | trailing zero decimal (D4) |
| `"18.00"` | `18` | D4 |
| `"2.50"` | `Fraction(5, 2)` | decimal, exact |
| `"-3"` | `-3` | negative |
| `" 18 "` | `18` | whitespace |
| `"18 dollars"` | `18` | trailing words |
| `"\text{18}"` | `18` | LaTeX wrapper |
| `"x = 18"` | `18` | one number among text |
| `"12 or 18"` | `None` | two numbers: ambiguous |
| `"no idea"` | `None` | no number |
| `""` | `None` | empty |
| `"1/2"` | `None` | fractions out of scope for now |

### `compute_reward`

| Completion | Gold | Expected | Checks |
| --- | --- | --- | --- |
| `"... \boxed{72}"` | `"72"` | `1.0` | correct |
| `"... \boxed{70}"` | `"72"` | `0.0` | wrong |
| `"... \boxed{1,000}"` | `"1000"` | `1.0` | formatting differs, same number |
| `"... \boxed{\$18.00}"` | `"18"` | `1.0` | formatting differs, same number |
| `"... \boxed{1000}"` | `"1,000"` | `1.0` | comma in the gold |
| `"The answer is 72."` | `"72"` | `0.0` | correct number but no box (D2) |
| `"\boxed{70} wait, recheck: \boxed{72}"` | `"72"` | `1.0` | self-correction to the right answer (D3) |
| `"\boxed{72} actually \boxed{70}"` | `"72"` | `0.0` | self-correction to a wrong answer (D3) |
| `"\boxed{1} \boxed{2} ... \boxed{100}"` | `"72"` | `0.0` | box spam |
| `"... so the total is \boxed{7"` | `"72"` | `0.0` | cut off |
| `"\boxed{}"` | `"72"` | `0.0` | empty box |
| `"\boxed{12 or 72}"` | `"72"` | `0.0` | ambiguous box |

## Instructions for writing the tests

- Write `tests/test_reward.py` only. **Do not implement `reward.py`**: it is written by hand, against these tests.
- One test function per function above, each using `@pytest.mark.parametrize` over the cases in its table, with a comment on every case saying what it checks.
- Import from `grpo_dapo.reward`. Compare `parse_number` results to ints or `Fraction`s (`Fraction(18) == 18` is true).
- Use raw strings (`r"..."`) for anything containing backslashes, such as `\boxed` and `\text`.
- Until `reward.py` is implemented, the tests are expected to fail at import.
