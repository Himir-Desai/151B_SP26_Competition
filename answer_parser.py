#Answer extraction and normalization helpers.

import re
from math import isclose


# Box extraction / replacement 
def _find_all_boxes(text: str) -> list[tuple[int, int, str]]:
    """Return list of (start_of_boxed, end_after_closing_brace, content)."""
    out = []
    needle = r'\boxed{'
    i = 0
    while True:
        idx = text.find(needle, i)
        if idx < 0: break
        start = idx + len(needle)
        depth = 1; j = start
        while j < len(text) and depth > 0:
            if text[j] == '{': depth += 1
            elif text[j] == '}': depth -= 1
            j += 1
        if depth == 0:
            out.append((idx, j, text[start:j-1]))
            i = j
        else:
            i = start  # unterminated — skip past opening
    return out


def extract_box(pred: str) -> str | None:
    """Return the LAST \\boxed{...} content from pred, or None."""
    tail = pred.split('</think>')[-1] if '</think>' in pred else pred
    boxes = _find_all_boxes(tail)
    return boxes[-1][2].strip() if boxes else None


def replace_box(pred: str, new_box_content: str) -> str:
    """Replace the last \\boxed{...} in pred with given content."""
    tail = pred.split('</think>')[-1] if '</think>' in pred else pred
    head_len = len(pred) - len(tail)
    boxes = _find_all_boxes(tail)
    if not boxes:
        return pred
    start, end, _ = boxes[-1]
    new_tail = tail[:start] + r'\boxed{' + new_box_content + '}' + tail[end:]
    return pred[:head_len] + new_tail


# ---------------------------------------------------------------------------
# Splitting boxed content by top-level comma (respecting brace depth)
# ---------------------------------------------------------------------------
def split_top_level(s: str) -> list[str]:
    depth_brace = depth_paren = depth_bracket = 0
    cur, parts = '', []
    for ch in s:
        if ch == '{':
            depth_brace += 1; cur += ch
        elif ch == '}':
            depth_brace -= 1; cur += ch
        elif ch == '(':
            depth_paren += 1; cur += ch
        elif ch == ')':
            depth_paren -= 1; cur += ch
        elif ch == '[':
            depth_bracket += 1; cur += ch
        elif ch == ']':
            depth_bracket -= 1; cur += ch
        elif ch == ',' and depth_brace == 0 and depth_paren == 0 and depth_bracket == 0:
            parts.append(cur.strip()); cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts



def _replace_simple_sqrt_calls(s: str) -> str:
    """Convert Python-style sqrt(x) to LaTeX \\sqrt{x} for simple arguments."""
    # Repeat so "sqrt(sqrt(2))" converges from the inside on simple cases.
    for _ in range(4):
        new = re.sub(r'(?<!\\)sqrt\s*\(([^()]+)\)', r'\\sqrt{\1}', s)
        if new == s:
            break
        s = new
    s = re.sub(r'(?<!\\)sqrt\s*([A-Za-z0-9]+)', r'\\sqrt{\1}', s)
    return s


def _normalize_exp_notation(s: str) -> str:
    """Rewrite Euler e powers into exp(...) when the expression has variables."""
    def repl_braced(m: re.Match) -> str:
        inner = m.group(1).strip()
        inner = re.sub(r'(\d(?:\.\d+)?)\s*([A-Za-z])', r'\1*\2', inner)
        return f'exp({inner})'

    s = re.sub(r'(?<![A-Za-z\\])e\s*\^\s*\{([^{}]+)\}', repl_braced, s)
    s = re.sub(
        r'(?<![A-Za-z\\])e\s*\^\s*([A-Za-z0-9.]+)',
        lambda m: f'exp({m.group(1)})',
        s,
    )
    s = re.sub(r'(?<=[0-9A-Za-z\)])\s+(?=exp\()', '*', s)
    s = re.sub(r'(?<=[0-9A-Za-z\)])(?=exp\()', '*', s)
    return s


def _normalize_reject_phrases(s: str) -> str:
    """Public gold uses uppercase REJECT/DO NOT REJECT; keep this deterministic."""
    if not re.search(r'\breject\b', s, re.I):
        return s
    s = re.sub(r'\bdo\s+not\s+reject\b', 'DO NOT REJECT', s, flags=re.I)
    s = re.sub(r'(?<!NOT\s)\breject\b', 'REJECT', s, flags=re.I)
    return s


def _normalize_answer_style(s: str) -> str:
    """Cosmetic answer-token rewrites that do not require gold."""
    s = _replace_simple_sqrt_calls(s)
    s = _normalize_exp_notation(s)
    for fn in ('arctan', 'arcsin', 'arccos', 'arccot', 'arcsec', 'arccsc'):
        s = s.replace('\\\\' + fn, fn)
        s = s.replace('\\' + fn, fn)
    s = s.replace('**', '^')
    s = s.replace(r'\times', '*').replace(r'\cdot', '*')
    s = s.replace('×', '*').replace('·', '*')
    s = re.sub(r'\binfinity\b', r'\\infty', s, flags=re.I)
    s = re.sub(r'(?<!\\)\binfty\b', r'\\infty', s, flags=re.I)
    s = _normalize_reject_phrases(s)
    return s


# ---------------------------------------------------------------------------
# Cosmetic normalizations
# ---------------------------------------------------------------------------
def clean_text(s: str) -> str:
    """Apply universal cosmetic fixes. Safe to run on whole box OR one part."""
    # 1. Strip \text{X} wrappers — keep X
    s = re.sub(r'\\text\{([^{}]*)\}', r'\1', s)
    # 2. \dfrac → \frac, \tfrac → \frac
    s = s.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')
    # 3. Strip LaTeX whitespace markers (BEFORE split so commas in \, don't fool us)
    s = s.replace(r'\,', '').replace(r'\!', '').replace(r'\;', '').replace(r'\:', '')
    # 4. Escaped-space-after-comma like "A,\ C" → "A, C"
    s = re.sub(r'\\\s+', ' ', s)
    # 5. Collapse multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()
    # 6. Normalize common non-LaTeX / parser-fragile answer notation.
    s = _normalize_answer_style(s)
    return s


# Backwards-compat alias
clean_part = clean_text


# ---------------------------------------------------------------------------
# Polynomial / expression notation: ^ <-> ** + drop spaces around operators
# ---------------------------------------------------------------------------
def to_sympy_style(s: str) -> str:
    """Convert LaTeX-ish polynomial notation to sympy-parseable form."""
    # x^4 → x**4   (don't disturb \frac, \sqrt, etc.)
    s = re.sub(r'(?<!\\)\^', '**', s)
    # 7x → 7*x  (digit immediately followed by letter)
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
    # )( → )*(
    s = re.sub(r'\)\(', r')*(', s)
    # remove spaces in expressions
    s = s.replace(' ', '')
    return s


def to_latex_style(s: str) -> str:
    """Convert Python ** to LaTeX ^."""
    return s.replace('**', '^')


# ---------------------------------------------------------------------------
# Precision matching: judger uses ABSOLUTE 1e-8 tol against gold's printed value.
# gold "2.2892" → pred "2.28920183054" FAILS (diff = 1.8e-6 > 1e-8).
# Truncate pred's numeric value to gold's printed precision when possible.
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$')


def _decimals(s: str) -> int | None:
    """Number of decimal places in a numeric string, or None if not numeric."""
    s = s.strip()
    if not _NUM_RE.match(s):
        return None
    if '.' in s:
        frac = s.split('.')[1].split('e')[0].split('E')[0]
        return len(frac)
    return 0


def match_precision(pred: str, gold: str) -> str:
    """
    Truncate pred to gold's decimal precision IF both are plain numerics.

    Examples:
      match_precision("2.28920183054", "2.2892")  → "2.2892"
      match_precision("19.7746",       "19.7745967126229")  → "19.7746"  (no extra precision to add)
      match_precision("43",            "43.12") → "43.00"                (pad to match)
      match_precision("\\sqrt{2}",     "1.4142") → "\\sqrt{2}"           (skip — pred not numeric)
    """
    g_dp = _decimals(gold)
    p_dp = _decimals(pred)
    if g_dp is None or p_dp is None:
        return pred  # non-numeric, leave alone
    try:
        val = float(pred)
        return f'{val:.{g_dp}f}'
    except ValueError:
        return pred


# ---------------------------------------------------------------------------
# Tuple re-wrap: if gold expects 1 entry like "(a, b)", glue split parts
# ---------------------------------------------------------------------------
def maybe_rewrap_tuple(parts: list[str], gold_list: list) -> list[str]:
    """
    If gold has exactly 1 entry that looks like a tuple `(a, b, ...)` and
    pred has N>1 parts that exactly fit (split a single tuple), join them.

    Examples:
      gold=['(2.35, 5.49)'], parts=['2.35', '5.49']  →  ['(2.35, 5.49)']
      gold=['(0, -1.375)'],  parts=['0', '-\\dfrac{11}{8}']  →  ['(0, -\\dfrac{11}{8})']
    """
    if len(gold_list) != 1 or len(parts) <= 1:
        return parts
    g0 = str(gold_list[0]).strip()
    if not (g0.startswith('(') and g0.endswith(')')):
        return parts
    # count commas inside the gold tuple — must match parts split
    inner = g0[1:-1]
    gold_inner_parts = split_top_level(inner)
    if len(gold_inner_parts) != len(parts):
        return parts
    return ['(' + ', '.join(parts) + ')']


# ---------------------------------------------------------------------------
# Top-level normalize: apply everything to a full pred string
# ---------------------------------------------------------------------------
def normalize_pred(pred: str, gold_list: list) -> str:
    """
    Take Opus's full response, rewrite its \\boxed{} content into a form
    the judger is more likely to canonicalize correctly.

    Returns the full response with the boxed content replaced.
    Pred unchanged if no \\boxed{} found.
    """
    box = extract_box(pred)
    if box is None:
        return pred

    # 1. Whole-box cleanup FIRST (so `\,` doesn't confuse comma-split)
    box = clean_text(box)
    # 2. Split into top-level parts
    parts = split_top_level(box)
    # 3. Per-part second pass (rare leftover)
    parts = [clean_text(p) for p in parts]
    # 4. Re-wrap tuples if gold structure asks
    parts = maybe_rewrap_tuple(parts, gold_list)
    # 5. Precision match: truncate each numeric pred-part to gold's precision
    if len(parts) == len(gold_list):
        parts = [match_precision(p, str(g)) for p, g in zip(parts, gold_list)]
    # 6. Rejoin with canonical separator
    new_box = ', '.join(parts)
    return replace_box(pred, new_box)


# ---------------------------------------------------------------------------
# Diagnostic helpers (for debugging — not part of the judge pipeline)
# ---------------------------------------------------------------------------
def normalize_pred_no_gold(pred: str) -> str:
    """
    Submission-time normalizer (no gold available).
    Applies: \\text{} strip, \\dfrac→\\frac, \\,/\\!/\\; strip, and UNSAFE
    compound symbolic → decimal conversion. Skips tuple-rewrap and
    precision-match (those need gold).
    """
    box = extract_box(pred)
    if box is None:
        inferred = infer_unboxed_answer(pred)
        if inferred is None:
            return pred
        return pred.rstrip() + '\n\n' + r'\boxed{' + inferred + '}'
    box = clean_text(box)
    parts = split_top_level(box)
    parts = [clean_text(p) for p in parts]
    new_box = ', '.join(parts)
    return replace_box(pred, new_box)


def normalize_pred_for_row(pred: str, row: dict | None = None) -> str:
    """Submission-time normalizer with safe question-aware repairs.

    Unlike ``normalize_pred`` this still does not use gold answers. The row is
    only used for surface cues such as the number of [ANS] slots or a stated
    percentage growth rate.
    """
    out = normalize_pred_no_gold(pred)
    if not row:
        return out
    box = extract_box(out)
    if box is None:
        return out
    question = str(row.get("question", ""))
    n_ans = question.count("[ANS]") or 1
    if n_ans == 1:
        out = _repair_single_slot_terms(out, question)
        out = _repair_single_slot_letter_list(out)
    out = _repair_currency_rounding(out, question)
    out = _repair_percent_growth_formula(out, question)
    return out


def _join_additive_terms(parts: list[str]) -> str:
    out = parts[0].strip()
    for part in parts[1:]:
        p = part.strip()
        if p.startswith("-"):
            out += " - " + p[1:].strip()
        elif p.startswith("+"):
            out += " + " + p[1:].strip()
        else:
            out += " + " + p
    return out


def _repair_single_slot_terms(pred: str, question: str) -> str:
    if not re.search(r'\b(first|next)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+terms\b|binomial expansion', question, re.I):
        return pred
    box = extract_box(pred)
    if box is None:
        return pred
    parts = split_top_level(box)
    if len(parts) <= 1:
        return pred
    if not any(re.search(r'[A-Za-z\\^]', p) for p in parts):
        return pred
    if all(re.fullmatch(r'[A-Ja-j]', p.strip()) for p in parts):
        return pred
    return replace_box(pred, _join_additive_terms(parts))


def _repair_single_slot_letter_list(pred: str) -> str:
    """Kaggle gold for check-all MCQ often stores CD, not C, D."""
    box = extract_box(pred)
    if box is None:
        return pred
    parts = split_top_level(box)
    if len(parts) <= 1:
        return pred
    cleaned = [p.strip().upper() for p in parts]
    if not all(re.fullmatch(r"[A-J]", p) for p in cleaned):
        return pred
    return replace_box(pred, "".join(cleaned))


def _fmt_rate(rate: float, raw: str) -> str:
    return raw if "." in raw else str(int(rate))


def _repair_percent_growth_formula(pred: str, question: str) -> str:
    rate_m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(?:\\%|%|percent)\s+per\s+(?:year|day|month|week)', question, re.I)
    if not rate_m:
        return pred
    rate_raw = rate_m.group(1)
    rate = float(rate_raw)
    expected_base = 1.0 + rate / 100.0
    box = extract_box(pred)
    if box is None:
        return pred
    parts = split_top_level(box)
    if len(parts) < 2:
        return pred

    rate_expr = f"(1 + {_fmt_rate(rate, rate_raw)}/100)"
    changed = False
    new_parts: list[str] = []
    first_initial: float | None = None
    first_var: str | None = None

    first_pat = re.compile(
        r'^\s*([0-9]+(?:\.[0-9]+)?)\s*\*\s*\(?([0-9]+(?:\.[0-9]+)?)\)?\s*\^\s*\{?([A-Za-z])\}?\s*$'
    )
    m0 = first_pat.match(parts[0])
    if m0 and isclose(float(m0.group(2)), expected_base, rel_tol=0, abs_tol=5e-10):
        first_initial = float(m0.group(1))
        first_var = m0.group(3)
        new_parts.append(f"{m0.group(1)}*{rate_expr}^{first_var}")
        changed = True
    else:
        new_parts.append(parts[0])

    delta_pat = re.compile(
        r'^\s*([0-9]+(?:\.[0-9]+)?)\s*\*\s*\(?([0-9]+(?:\.[0-9]+)?)\)?\s*\^\s*\{?([A-Za-z])\s*-\s*1\}?\s*$'
    )
    for part in parts[1:]:
        m = delta_pat.match(part)
        if (
            first_initial is not None
            and first_var is not None
            and m
            and m.group(3) == first_var
            and isclose(float(m.group(2)), expected_base, rel_tol=0, abs_tol=5e-10)
            and isclose(float(m.group(1)), first_initial * rate / 100.0, rel_tol=0, abs_tol=5e-8)
        ):
            new_parts.append(f"{first_initial:g}*({rate_expr}^({first_var}-1))*({_fmt_rate(rate, rate_raw)}/100)")
            changed = True
        else:
            new_parts.append(part)

    if not changed:
        return pred
    return replace_box(pred, ', '.join(new_parts))


def _repair_currency_rounding(pred: str, question: str) -> str:
    if not re.search(r'\b(?:premium|charged\s+a\s+monthly\s+premium)\b', question, re.I):
        return pred
    box = extract_box(pred)
    if box is None:
        return pred
    parts = split_top_level(box)
    changed = False
    new_parts: list[str] = []
    for part in parts:
        p = part.strip()
        if re.fullmatch(r'-?\d+\.\d{3,}', p):
            new_parts.append(f"{float(p):.2f}")
            changed = True
        else:
            new_parts.append(part)
    return replace_box(pred, ', '.join(new_parts)) if changed else pred


def infer_unboxed_answer(pred: str) -> str | None:
    """Best-effort rescue for truncated/no-box outputs with explicit final text.

    This intentionally avoids "last number" guessing. It only fires on strong
    final-answer phrases, so it is much less risky than scraping arbitrary
    reasoning.
    """
    text = pred.split('</think>')[-1] if '</think>' in pred else pred
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return None

    # MCQ-style explicit phrases.
    mcq_patterns = [
        r'(?:final\s+)?answer\s*(?:is|:)\s*\(?([A-J])(?![A-Za-z])\)?',
        r'(?:correct\s+)?(?:choice|option|letter)\s*(?:is|:)\s*\(?([A-J])(?![A-Za-z])\)?',
    ]
    for pat in mcq_patterns:
        matches = list(re.finditer(pat, text, re.I))
        if matches:
            return matches[-1].group(1).upper()

    # Free-form explicit phrases. Stop at a sentence boundary unless the value
    # is parenthesized/bracketed. Keep it short to avoid swallowing reasoning.
    value = None
    ff_patterns = [
        r'(?:final\s+answer|answer|result)\s*(?:is|:)\s*(?:approximately|about)?\s*([^;\n]{1,160})',
        r'(?:therefore|thus),?\s*([^;\n]{1,160})\s+is\s+the\s+(?:answer|result)',
    ]
    for pat in ff_patterns:
        matches = list(re.finditer(pat, text, re.I))
        if matches:
            value = matches[-1].group(1).strip()
    if value is None:
        return None

    value = re.split(r'(?<=\D)\.\s+|(?<=\d)\.\s+(?=[A-Z])', value, 1)[0]
    value = value.strip(' $.')
    value = re.sub(r'^(?:\\boxed\{)?', '', value).strip()
    value = value[:-1] if value.endswith('}') and value.count('{') < value.count('}') else value
    value = re.sub(r'\s*(?:miles|feet|dollars|degrees|mph|percent|%)\s*$', '', value, flags=re.I)
    value = clean_text(value.strip())

    # Reject obvious prose fragments.
    if not value or len(value.split()) > 12:
        return None
    if re.search(r'\b(the|this|that|problem|option|choice|likely|probably)\b', value, re.I):
        return None
    if not re.search(r'[0-9A-Za-z\\]', value):
        return None
    return value


def normalize_box_only(box: str, gold_list: list) -> str:
    """Lightweight: normalize just the boxed content (string in, string out)."""
    box = clean_text(box)
    parts = split_top_level(box)
    parts = [clean_text(p) for p in parts]
    parts = maybe_rewrap_tuple(parts, gold_list)
    return ', '.join(parts)


if __name__ == '__main__':
    # Quick smoke test
    tests = [
        # (pred_box, gold, expected_normalized)
        (r'\boxed{\text{decreasing}, 4339145}', ['DECREASING', '4339144.29'],
         'decreasing, 4339145'),
        (r'\boxed{0, -\dfrac{11}{8}}', ['(0, -1.375)'],
         '(0, -\\frac{11}{8})'),
        (r'\boxed{A,\ C}', ['A', 'C'],
         'A, C'),
        (r'\boxed{-10,\, 100,\, -9}', ['-10', '100', '-9'],
         '-10, 100, -9'),
        (r'\boxed{\sqrt{2}, \pi}', ['\\sqrt{2}', '\\pi'],
         '\\sqrt{2}, \\pi'),
    ]
    for raw, gold, want in tests:
        got = normalize_box_only(extract_box(raw), gold)
        mark = '✓' if got == want else '✗'
        print(f'{mark} got={got!r}  want={want!r}')
