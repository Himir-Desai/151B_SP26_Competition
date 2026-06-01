import re

# ── System prompts ─────────────────────────────────────────────────────────


CONTRACT_PRECISION_HINT = (
    "Numeric answers must have at least 15 significant digits. "
    "Pay special attention to: confidence intervals, p-values, test statistics, critical "
    "values, regression coefficients, roots, and numeric endpoints. "
    "Do not round intermediate computations."
)


SYSTEM_FF_TYPED_V1 = ( "You are a Nobel Prize winning mathematician tasked to solve math problems easy or difficult. " 
"You should answer the questions like a professional mathematician who is careful and meticulous in solving problems. " 
"Formatting Instructions:\n" 
"1. If the question provides options, answer with the option letter, not the value.\n" 
"2. Never round your final answer, even if the question explicitly asks for rounding or decimal approximation.\n" 
"3. Never output the final answer in decimal form under any circumstances. Do not use decimal approximations, rounded decimals, or evaluated decimal constants.\n" 
"4. Always prefer exact symbolic form. Leave complicated answers unevaluated, e.g. leave 325*(1+325) as is instead of evaluating it, and leave (1/2)^[(1999-1963)/31] as is.\n" 
"5. Leave constants and functions exact. Leave pi as pi, e as e, ln(...) as ln(...), sqrt(...) as sqrt(...), atan(...) as atan(...), arctan(...) as arctan(...), sin(...), cos(...), and similar expressions unevaluated no matter what the question says.\n" 
"6. If the problem contains decimal numbers, convert terminating decimals to fractions before using them in the final answer. For example, use 119/25 instead of 4.76, and use exact fractional form instead of decimal form whenever possible.\n" 
"7. If an exact numerical value is not simplifiable, leave it as an unevaluated expression. Never replace it with a decimal approximation.\n" 
"8. For questions that demand fractional answers like 5/8, output \\boxed{5/8} as it is and not \\boxed{frac{5}{8}}.\n"
"9. For answers that include expressions, like 2x^2+3x+4, answer like 2*x^2+3*x+4 instead of 2x^2+3x+4.\n" 
"10. If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, e.g. if the answer is 3 and 7, output \\boxed{3, 7}.\n" 
"11. Never add the characters { } inside the \\boxed{} answer.\n" 
"12. In equation-type questions, your final answer should use the variables used and suggested by the question.\n" 
"13. Make sure your equations and answers are properly bracketed.\n" "The most important thing is to put your final answer inside \\boxed{} without failure using the formatting instructions above. " 
"Only output one final \\boxed{} answer at the end with answers to all parts of the question if there are any, or multiple answers for the same question if there are multiple. " 
"Do not output any other \\boxed{} that is not the final answer to the question." ) 
SYSTEM_MC_TYPED_V1 = ( 
"You are a world-class mathematician solving a multiple-choice problem.\n" 
"All reasoning, working, and deliberation must happen entirely inside your thinking. " 
"Your response outside thinking must contain ONE thing only: \\boxed{X} where X is your chosen letter(s).\n\n" "Inside your thinking, use this strict two-phase approach:\n\n" 
"PHASE 1 — SOLVE INDEPENDENTLY:\n" "Completely ignore the answer options. Solve the problem from scratch as if it were open-ended, " 
"working step by step to a precise answer. Do not look at or mention the options during this phase.\n" 
"PHASE 2 — SELECT:\n" "1. EXACT or EQUIVALENT: Does any option equal your answer through algebra or simplification " 
"(e.g., 2/3·ln9 = 4/3·ln3)? If your answer involves pi or e, substitute pi≈3.14159 or e≈2.71828 " 
"and check for a close match. If two options are mathematically equivalent, prefer the simplified form.\n" 
"2. QUALITATIVE: If your Phase 1 result is divergent, undefined, or 'no solution' (including any " 
"improper integral that diverges), look for the option that captures that property — it may be worded " "as 'diverges', 'DNE', '∞', '-∞', or a descriptive phrase. Do NOT fall back to CLOSEST in this case.\n" 
"3. CLOSEST: Only if neither of the above apply, pick the numerically closest option.\n" 
"If nothing matches, briefly reconsider whether Phase 1 made an error, then commit to one letter.\n\n" 
"MANDATORY OUTPUT RULE:\n" 
"After your thinking block ends, write NOTHING except \\boxed{X}. " 
"For single answer: \\boxed{C}. " 
"For multiple answers (select all that apply): concatenate letters with NO commas or spaces: \\boxed{BCEG} not \\boxed{B,C,E,G}. " 
"No explanation, no summary, no 'therefore', no 'the answer is' — just \\boxed{...} and nothing else. " 
"Violating this rule is not permitted under any circumstance." 
)


# SYSTEM_MC_TYPED_V1 = (
#     "You are a world-class mathematician solving a multiple-choice problem.\n"
#     "Think step-by-step inside <think>...</think>.\n\n"
#     "REQUIRED PROCESS:\n"
#     "1. Compute the answer completely independently — do not look at the options yet.\n"
#     "2. Compare your result against EVERY listed option, including approximate numeric "
#     "matches (substitute π≈3.14159, e≈2.71828 if needed).\n"
#     "3. If multiple options look numerically close, choose the one that best matches "
#     "your derived value.\n"
#     "4. Commit to exactly one letter.\n\n"
#     "After your thinking block, output exactly \\boxed{LETTER}. Nothing else."
# )

# SYSTEM_FF_TYPED_V1 = (
#     "You are a world-class mathematician solving a math problem.\n"
#     "Think step-by-step inside <think>...</think>.\n\n"
#     "Before writing the final answer, verify inside your thinking:\n"
#     "1. Slot count — your answer has the same number of entries as [ANS] placeholders.\n"
#     "2. Slot order — entries appear in the same order as the placeholders.\n"
#     "3. Answer form — each entry matches the form the blank requests.\n\n"
#     "Then output exactly one \\boxed{...} containing all answers comma-separated. "
#     "Include a unit or label only when the blank explicitly asks for it. "
#     "Do NOT repeat units printed outside the blank."
# )

# ── Compiled question-text classifiers ────────────────────────────────────

_EMBEDDED_CHOICE_RE = re.compile(
    # Require at least TWO consecutive uppercase option labels so that a single
    # sub-question label (e.g. "B. Repeat…" after an [ANS]) does not fire.
    # Branch 1: options appear AFTER an [ANS] slot (most common).
    # Branch 2: "match the column" format where options are listed BEFORE [ANS] slots.
    r'\[ANS\](?:(?!\[ANS\]).){0,400}?[A-E][.)]\s(?:(?!\[ANS\]).){0,300}?[A-E][.)]\s'
    r'|(?:\b[A-E][.)]\s.{3,200}?){3}',
    re.DOTALL,
)

_MULTI_SELECT_RE = re.compile(
    r'check all|select all|all that apply|choose all|mark all',
    re.IGNORECASE,
)

_INTERVAL_LIST_RE = re.compile(
    # Structural patterns: [ANS] appearing inside vector/tuple notation, or explicit
    # ordered-pair/tuple language.  Avoids firing on "confidence interval" when the
    # answer is a single number (e.g. required sample size for a CI).
    r'<[^>]*\[ANS\]'                 # vector  <ANS, ANS>
    r'|\([^)]*\[ANS\][^)]*\)'        # tuple   (ANS, ANS)
    r'|\bordered pair\b'
    r'|\btuple\b',
    re.IGNORECASE | re.DOTALL,
)

_HIGH_PRECISION_RE = re.compile(
    r'p[- ]?value|test statistic|regression|anova|chi[- ]?square'
    r'|z[- ]?test|t[- ]?test|correlation coefficient|standard error'
    r'|f[- ]?statistic|degrees of freedom',
    re.IGNORECASE,
)

_EXACT_SYMBOLIC_RE = re.compile(
    r'exact (?:form|value|answer)|fraction|\\sqrt|\bsqrt\b'
    r'|\bpi\b|\\pi\b|\bln\b|\blog\b|do not approximate|in terms of',
    re.IGNORECASE,
)

_UNIT_WORD_RE = re.compile(
    r'\b(?:feet|foot|meters?|miles?|km|dollars?|cents?|percent|direction'
    r'|label|days?|hours?|minutes?|seconds?|kg|lbs?|pounds?|inches?'
    r'|gallons?|liters?|years?|months?)\b',
    re.IGNORECASE,
)

_ROUNDING_RE = re.compile(
    r'round(?:ed)? to|nearest|decimal place|significant figure|sig fig',
    re.IGNORECASE,
)

_PERCENT_RE = re.compile(
    r'percent(?:age)?|\b%\b',
    re.IGNORECASE,
)

_CONCLUSION_CHOICE_RE = re.compile(
    r'\breject\b|fail to reject|sufficient evidence|not sufficient evidence'
    r'|type [iI] error|type [iI][iI] error|null hypothesis|alternative hypothesis',
    re.IGNORECASE,
)

_TRUE_FALSE_RE = re.compile(
    r'\benter\s+[TF]\s+or\s+[FT]\b'
    r'|\btrue\s+or\s+false\b'
    r'|\b[TF]\s+for\s+(?:true|false)\b'
    r'|\bselect\s+true\s+or\s+false\b',
    re.IGNORECASE,
)

# ── Helpers ────────────────────────────────────────────────────────────────

def is_mc(ex):
    return bool(ex.get("options"))


def format_mcq_user(q, opts):
    letters = "ABCDEFGHIJ"
    opts_text = "\n".join(f"({letters[i]}) {opt}" for i, opt in enumerate(opts))
    return f"{q}\n\n{opts_text}\n\nAnswer with exactly one letter in \\boxed{{}}."


def _count_slots(ex):
    return max(1, ex.get("question", "").count("[ANS]"))


def _contract_requirement_lines(ex, hiprec=True):
    q = ex.get("question", "")
    n = _count_slots(ex)
    lines = [
        f"Put all {n} answer(s) in a single \\boxed{{}} separated by commas.",
        "Do not convert between symbolic and decimal forms unless the problem requires it.",
    ]
    if _EMBEDDED_CHOICE_RE.search(q):
        lines.append("Select only the embedded option letter(s) — do not copy the option text.")
    if _MULTI_SELECT_RE.search(q):
        lines.append("Concatenate all selected letters without spaces, e.g. \\boxed{ACE}.")
    if _INTERVAL_LIST_RE.search(q):
        lines.append("Group interval or tuple entries inside parentheses, e.g. (a, b).")
    if hiprec and _HIGH_PRECISION_RE.search(q):
        lines.append(CONTRACT_PRECISION_HINT)
    if _EXACT_SYMBOLIC_RE.search(q):
        lines.append("Give the exact symbolic form; do not approximate unless instructed.")
    if _UNIT_WORD_RE.search(q):
        lines.append("Include the unit or label if the blank explicitly requests it.")
    return lines


def _format_contract_requirements(ex, hiprec=True):
    return "\n".join(f"- {line}" for line in _contract_requirement_lines(ex, hiprec))


def _detect_typed_traits(ex, hiprec=True):
    q = ex.get("question", "")
    if is_mc(ex):
        traits = ["mcq_option_match"]
        if _HIGH_PRECISION_RE.search(q) or _ROUNDING_RE.search(q) or _PERCENT_RE.search(q):
            traits.append("mcq_numeric_match")
        return traits

    n = _count_slots(ex)
    traits = ["ff_single_slot" if n == 1 else "ff_multi_slot"]
    if hiprec and _HIGH_PRECISION_RE.search(q):
        traits.append("high_precision_numeric")
    if _CONCLUSION_CHOICE_RE.search(q):
        traits.append("stats_conclusion_letters")
    if _EMBEDDED_CHOICE_RE.search(q) or _TRUE_FALSE_RE.search(q):
        traits.append("embedded_choice_letters")
    if _MULTI_SELECT_RE.search(q):
        traits.append("multi_select_letters")
    if _INTERVAL_LIST_RE.search(q):
        traits.append("interval_or_grouped_list")
    if _EXACT_SYMBOLIC_RE.search(q):
        traits.append("exact_symbolic")
    if _UNIT_WORD_RE.search(q):
        traits.append("unit_or_word_blank")
    if _PERCENT_RE.search(q):
        traits.append("percent_context")
    if _ROUNDING_RE.search(q):
        traits.append("explicit_rounding")
    return traits


def _typed_requirement_lines(ex, hiprec=True):
    traits = set(_detect_typed_traits(ex, hiprec))
    n = _count_slots(ex)
    lines = []

    if "ff_multi_slot" in traits:
        lines.append(f"Provide exactly {n} comma-separated answers in \\boxed{{}}.")
        if "interval_or_grouped_list" in traits:
            lines.append(
                "Wrap interval/tuple entries in parentheses to protect internal commas, "
                "e.g. (a, b)."
            )
    else:
        lines.append("Provide exactly one answer in \\boxed{}.")

    if "high_precision_numeric" in traits:
        lines.append(
            "Give at least 15 significant digits; do not round intermediate computations."
        )
    if "stats_conclusion_letters" in traits:
        lines.append(
            "State the conclusion as the problem requests (reject / fail to reject / etc.)."
        )
    if "embedded_choice_letters" in traits:
        lines.append("Answer with the embedded option letter only, not the option text.")
    if "multi_select_letters" in traits:
        lines.append("Concatenate all selected letters alphabetically without spaces, e.g. ACE.")
    if "exact_symbolic" in traits:
        lines.append("Keep the answer in exact symbolic form; do not convert to decimal.")
    if "unit_or_word_blank" in traits:
        lines.append("Include the unit or label only when the blank explicitly requests it.")
    if "percent_context" in traits:
        lines.append(
            "Express percentages as a number (e.g. 42.5, not 0.425) unless otherwise specified."
        )
    if "explicit_rounding" in traits:
        lines.append(
            "Obey the problem's requested precision exactly; do not over- or under-round."
        )

    lines.append("Do not change the answer form simply because an equivalent form exists.")
    return lines


def _format_typed_requirements(ex, hiprec=True):
    traits = _detect_typed_traits(ex, hiprec)
    trait_str = f"Detected answer profile: {', '.join(traits)}"
    rules_str = "\n".join(f"- {line}" for line in _typed_requirement_lines(ex, hiprec))
    return f"{trait_str}\n{rules_str}"


# ── Prompt builders ────────────────────────────────────────────────────────

_MCQ_CONTRACT_BULLETS = (
    "- Compute the answer completely independently before looking at the options.\n"
    "- Compare your result against every listed option, including approximate numeric matches.\n"
    "- If multiple options look numerically close, pick the one that best matches your derived value.\n"
    "- Output exactly one boxed uppercase letter."
)


def build_prompt(ex, tokenizer, hiprec=True):
    q = ex.get("question", "")
    if is_mc(ex):
        user_content = f"{format_mcq_user(q, ex['options'])}\n\n{_MCQ_CONTRACT_BULLETS}"
        system = SYSTEM_MC_TYPED_V1
    else:
        requirements = _format_typed_requirements(ex, hiprec)
        precision = f"\n\n{CONTRACT_PRECISION_HINT}" if hiprec and _HIGH_PRECISION_RE.search(q) else ""
        user_content = (
            f"{q}\n\n{requirements}{precision}\n\nPut all answers in a single \\boxed{{}}."
        )
        system = SYSTEM_FF_TYPED_V1

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
