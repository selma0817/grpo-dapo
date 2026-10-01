# Reward spec (GSM8K)

The rule-based reward scores one model completion against the gold answer: `1.0` if the model's final answer equals the gold answer, otherwise `0.0`. Every advantage in GRPO comes from this number, so anything the checker gets wrong, the model learns.

## Design decisions

| # | Decision | Choice | Why |
| --- | --- | --- | --- |
| D1 | Required answer format | The final answer goes in `\boxed{...}` (the prompt asks for it) | Standard for math RL; unambiguous to extract; also works for MATH's LaTeX answers later |
| D2 | Correct number, but no `\boxed{}` | Reward `0.0` (strict) | Guessing which number is "the answer" gives noisy rewards and invites hacking. The model learns the format quickly; the format rate is tracked as a metric |
| D3 | Several `\boxed{}` in one completion | **The last one** counts | The last box is the model's final decision. Spamming boxes doesn't help, since only the last one counts |
| D4 | When are two answers equal? | **Same number**, compared exactly (`fractions.Fraction`) after cleanup | `18 = 18.0 = 18.00`, `1,000 = 1000`, with no floating-point surprises |
| D5 | Malformed or unexpected input | **Fail closed**, returning `None`, `False`, or `0.0` | Reward code runs inside the training loop; malformed model output must never stop a run |

Smaller defaults (change any of them, but update the tests to match):

- **Garbage input never crashes the reward.** Missing markers, very long strings, unusual Unicode, `None`, and `None`-like text fail closed with `None`, `False`, or `0.0`. **Gold answers are validated once at load time instead:** `data.py` must check that `extract_gold` and `parse_number` succeed for every example, and crash or report if any fail. Otherwise a malformed gold answer would silently score every completion 0, giving that question Â = 0 with no warning.
- `extract_answer` returns the box content **stripped of surrounding whitespace**, and returns `None` for an empty box, no box, or an unclosed box (output cut off at the length limit).
- `extract_answer` matches **balanced braces**, so `\boxed{\text{18}}` gives `\text{18}`, not `\text{18`.
- `parse_number` ignores `$`, `\$`, `%`, thousands commas, whitespace and non-numeric text, then requires **exactly one number**: zero numbers or two or more numbers give `None`.
- Fractions (`1/2`, `\frac{1}{2}`) and scientific notation are **out of scope until the MATH dataset is added (R5)**; `parse_number` returns `None` for them for now. GSM8K gold answers are integers by design, so this only affects rare cases like `\frac{144}{2}`. For MATH, `is_equivalent` will try the exact numeric check first and fall back to `math-verify`.
- `parse_number` returns a `fractions.Fraction`, never a `float`, so equality is exact (D4). A test checks the type, since `18.0 == 18` would let a float implementation pass every value check.

## Functions (all in `src/grpo_dapo/reward.py`)

```text
GSM8K "answer" field ──extract_gold──▶ "72" or None ──────────────┐
                                                                  ├─▶ is_equivalent ─▶ bool
model completion ──extract_answer──▶ "72" or None ────────────────┘         │
                  └──────────── compute_reward (extract, compare, score) ───┴─▶ 1.0 / 0.0

is_equivalent ──parse_number(pred) + parse_number(gold)──▶ exact equality
```

| Function | Signature | Job |
| --- | --- | --- |
| `extract_gold` | `(answer_field: str \| None) -> str \| None` | Return the text after `####`, stripped. Return `None` for missing, malformed, or non-string input. |
| `extract_answer` | `(completion: str \| None) -> str \| None` | Return the content of the **last** complete `\boxed{...}`, stripped. `None` if there is none, it is empty, or the input is invalid. |
| `parse_number` | `(text: str \| None) -> Fraction \| None` | Clean up the text and return its single number exactly. `None` if it doesn't contain exactly one number or the input is invalid. |
| `is_equivalent` | `(pred: str \| None, gold: str \| None) -> bool` | Parse both answers and compare them exactly. Return `False` if either answer cannot be parsed. This is the only function that defines answer equivalence. |
| `compute_reward` | `(completion: str \| None, gold: str \| None) -> float` | Extract the completion's answer, delegate comparison to `is_equivalent`, and return `1.0` for equivalent answers or `0.0` otherwise. Never raise for malformed inputs. |

## Test cases

### `extract_gold`

| Input | Expected | Checks |
| --- | --- | --- |
| `"Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"` | `"72"` | real GSM8K example; numbers before `####` are ignored |
| `"...\n####72"` | `"72"` | no space after `####` |
| `"...\n#### 72\n"` | `"72"` | trailing newline / whitespace |
| `"...\n#### 1,000"` | `"1,000"` | returns raw text; cleanup is `parse_number`'s job |
| `"...\n#### -3"` | `"-3"` | negative gold |
| `"no marker here"` | `None` | missing marker fails closed |
| `None` | `None` | non-string input fails closed |
| odd Unicode text | `None` | unusual characters do not raise |
| a very long garbage string | `None` | pathological model/data text does not raise |

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
| `None` | `None` | non-string input fails closed |
| odd Unicode text | `None` | unusual characters do not raise |
| a very long garbage string | `None` | pathological model output does not raise |
| `"\boxed{"` repeated 10,000 times | `None` | thousands of unclosed braces: must stay fast and must not hit the recursion limit |
| `"\boxed{70} then \boxed{7"` | `"70"` | the last *complete* box wins when the final box was cut off |

Timing check: `extract_answer` on the 10,000 unclosed braces finishes in under 1 second (a linear scan takes milliseconds; a quadratic one takes minutes).

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
| `"1/2"` | `None` | fractions out of scope until MATH (R5) |
| `"None"` | `None` | `None`-like text does not raise |
| `None` | `None` | non-string input fails closed |
| odd Unicode text | `None` | unusual characters do not raise |
| a very long garbage string | `None` | pathological text does not raise |

Type check: `parse_number("72")`, `parse_number("2.50")` and `parse_number("1,000")` all return a `Fraction` (D4).

### `is_equivalent`

| Pred | Gold | Expected | Checks |
| --- | --- | --- | --- |
| `"72"` | `"72"` | `True` | identical numbers |
| `"18.00"` | `"18"` | `True` | formatting differs, same number |
| `"1,000"` | `"1000"` | `True` | comma formatting differs |
| `"70"` | `"72"` | `False` | different numbers |
| `"12 or 72"` | `"72"` | `False` | ambiguous prediction |
| `"12345678901234567891"` | `"12345678901234567890"` | `False` | exact comparison: these are equal as floats (D4) |
| `None` | `"72"` | `False` | missing prediction fails closed |
| `"72"` | `None` | `False` | missing gold fails closed |
| `"None"` | `"72"` | `False` | `None`-like text does not raise |
| odd Unicode text | `"72"` | `False` | unusual characters do not raise |
| a very long garbage string | `"72"` | `False` | pathological text does not raise |

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
| `"\boxed{-3}"` | `"-3"` | `1.0` | negative answer end to end |
| `None` | `"72"` | `0.0` | missing completion fails closed |
| `"\boxed{72}"` | `None` | `0.0` | missing gold fails closed |
| odd Unicode text | `"72"` | `0.0` | unusual characters do not raise |
| a very long garbage string | `"72"` | `0.0` | pathological model output does not raise |

## Implementation requirements

For whoever implements `src/grpo_dapo/reward.py`:

- **Standard library only** (`re`, `fractions`). No `math-verify` until MATH (R5).
- **Explicit checks, not a catch-all.** Fail closed (D5) with `isinstance` / `None` checks. Don't wrap functions in `try/except Exception: return 0.0`: that would also hide bugs in the reward itself (a typo would silently make every reward 0.0). If a last-resort catch-all is added anywhere, it must log what it caught.
- **`extract_answer` runs in linear time without recursion.** Use one left-to-right pass with a stack of open-brace positions. Scanning forward from every `\boxed{` is quadratic and far too slow for 10,000 unclosed braces (a test enforces a time limit).
- **The last *complete* box wins.** In `\boxed{70} then \boxed{7` (the final box was cut off), the answer is `"70"`. Only a completion with no complete box at all gives `None`.
- **`parse_number`:** remove `\$`, `$`, `%`, and commas *between digits* (so `1,000` → `1000`, while `12, 18` stays two numbers). Match numbers with `[0-9]`, not `\d` (which also matches non-ASCII digits like `٣`). Build the result with `Fraction(string)`, never `float`.
- **`is_equivalent` is the only place equality is defined.** `compute_reward` calls `extract_answer`, then `is_equivalent`, and returns `1.0` or `0.0`.
- Type hints and a one-line docstring per function. Do not modify the tests to make them pass.

## Instructions for writing the tests

- The tests in `tests/test_reward.py` were written first, from the tables above; `reward.py` is implemented against them.
- One test function per function above, each using `@pytest.mark.parametrize` over the cases in its table, with a comment on every case saying what it checks.
- Import from `grpo_dapo.reward`. Compare `parse_number` results to ints or `Fraction`s (`Fraction(18) == 18` is true).
- Use raw strings (`r"..."`) for anything containing backslashes, such as `\boxed` and `\text`.
- Until `reward.py` is implemented, the tests are expected to fail at import.
