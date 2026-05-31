import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = "sft-3-merged"
GPU_ID        = "3"                    # CUDA_VISIBLE_DEVICES
DATA_PATH     = "data/public.jsonl"
OUTPUT_PATH   = "results/starter_results.jsonl"
MAX_TOKENS    = 32768
NUM_QUESTIONS = 300               # Number of questions to evaluate (None = all)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# ── Prompt Construction ───────────────────────────────────────────────────────
# SYSTEM_PROMPT_MATH = (
#     "You are an expert mathematician. Solve the problem step-by-step. "
#     "Put your final answer inside \\boxed{}. "
#     "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
#     "e.g. \\boxed{3, 7}."
# )
# SYSTEM_PROMPT_MCQ = (
#     "You are an expert mathematician. "
#     "Read the problem and the answer choices below, then select the single best answer. "
#     "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
# )

SYSTEM_PROMPT_MATH = (
    "You are a world-class mathematical reasoning system that solves problems using precise step-by-step symbolic reasoning.\n\n"

    "Your primary goal is EXACTNESS, not approximation. A calculator will evaluate expressions afterward.\n\n"

    "After thinking, just give final answer in boxed, dont give justification"
    "In confusion prefer these rules and try to think question is correct rather than be oversmart"

    "Follow these rules strictly in priority order:\n\n"

    "1. Put ALL final answers inside EXACTLY ONE SINGLE \\boxed{} at the VERY END.\n"
    "   If there are multiple answers or subparts, include ALL of them in the SAME boxed expression separated by commas.\n"
    "   Never use \\boxed{} more than once.\n\n"

    "2. Never round answers unless the question explicitly asks for rounding.\n"
    "   Preserve exact values whenever possible.\n\n"

    "3. Do NOT numerically evaluate irrational, non-terminating, logarithmic, trigonometric, exponential, or symbolic power expressions.\n"
    "   Keep exact forms such as:\n"
    "   - sqrt(2)\n"
    "   - pi\n"
    "   - e\n"
    "   - ln(3)\n"
    "   - arctan(3)\n"
    "   - tan(1)\n"
    "   - (1/2)^((1999-1963)/31)\n"
    "   - 2^(sqrt(3))\n\n"

    "4. Never decimalize fractions that appear inside exponents, radicals, logarithms, trigonometric functions, symbolic expressions, or larger algebraic structures.\n"
    "   Examples:\n"
    "   - Keep (1/2)^x unchanged\n"
    "   - Keep sqrt(1/2) unchanged\n"
    "   - Keep ln(1/2) unchanged\n"
    "   - Keep (3/5)pi unchanged\n\n"

    "5. Prefer LaTeX formatting such as \\frac{}{}, \\sqrt{}, and exponents.\n\n"

    "6. Ensure every sub-question has been answered inside the FINAL SINGLE boxed expression. Every [ANS], ___ and question mark should be accounted for an answer.\n"

    "7. Make sure to use the variables as asked in question, and maintain proper bracketing. While sin theta might seem conventional, if the questions explicitly asks to use t, answer with sin(t)"
)

SYSTEM_PROMPT_MCQ = (
    "You are a world-class mathematician solving a multiple-choice problem.\n"
    "All reasoning, working, and deliberation must happen entirely inside your thinking. "
    "Your response outside thinking must contain ONE thing only: \\boxed{X} where X is your chosen letter.\n\n"
    "Inside your thinking, use this strict two-phase approach:\n\n"
    "PHASE 1 — SOLVE INDEPENDENTLY:\n"
    "Completely ignore the answer options. Solve the problem from scratch as if it were open-ended, "
    "working step by step to a precise answer. Do not look at or mention the options during this phase.\n"
    "PHASE 2 — SELECT:\n"
    "1. EXACT or EQUIVALENT: Does any option equal your answer through algebra or simplification "
    "(e.g., 2/3·ln9 = 4/3·ln3)? If your answer involves pi or e, substitute pi≈3.14159 or e≈2.71828 "
    "and check for a close match. If two options are mathematically equivalent, prefer the simplified form.\n"
    "2. QUALITATIVE: If your Phase 1 result is divergent, undefined, or 'no solution' (including any "
    "improper integral that diverges), look for the option that captures that property — it may be worded "
    "as 'diverges', 'DNE', '∞', '-∞', or a descriptive phrase. Do NOT fall back to CLOSEST in this case.\n"
    "3. CLOSEST: Only if neither of the above apply, pick the numerically closest option.\n"
    "If nothing matches, briefly reconsider whether Phase 1 made an error, then commit to one letter.\n\n"
    "MANDATORY OUTPUT RULE:\n"
    "After your thinking block ends, write NOTHING except \\boxed{X}. "
    "No explanation, no summary, no 'therefore', no 'the answer is' — just \\boxed{X} and nothing else. "
    "Violating this rule is not permitted under any circumstance."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def postprocess_response(response: str, gold_list: list = None) -> str:
    think_end = response.rfind("</think>")
    if think_end >= 0:
        prefix = response[:think_end + len("</think>")]
        answer = response[think_end + len("</think>"):]
    else:
        prefix = ""
        answer = response

    answer = re.sub(r'(\\boxed\{)<([^>]*)>(\})', r'\1\2\3', answer)

    if gold_list and len(gold_list) == 1 and re.fullmatch(r"[A-Z]{2,}", str(gold_list[0])):
        def _join_letters(m: re.Match) -> str:
            inner = m.group(1)
            parts = [p.strip() for p in inner.split(",")]
            if all(re.fullmatch(r"[A-Z]", p) for p in parts) and len(parts) > 1:
                return "\\boxed{" + "".join(parts) + "}"
            return m.group(0)
        answer = re.sub(r"\\boxed\{([^}]+)\}", _join_letters, answer)

    if gold_list:
        gold_str = " ".join(str(g) for g in gold_list)
        if "infinity" in gold_str and "\\infty" not in gold_str:
            answer = answer.replace("\\infty", "infinity")

    return prefix + answer


def extract_letter(text: str) -> str:
    # Search post-</think> first — that is the model's committed answer
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text
    m = re.search(r"\\boxed\{([A-Za-z])\}", search_text)
    if m:
        return m.group(1).upper()
    # Fall back to the last single-letter \boxed{} anywhere in the response
    all_matches = list(re.finditer(r"\\boxed\{([A-Za-z])\}", text))
    if all_matches:
        return all_matches[-1].group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from tqdm import tqdm

    sys.path.insert(0, ".")
    from judger import Judger

    # ── Load Dataset ──────────────────────────────────────────────────────────
    data = [json.loads(line) for line in open(DATA_PATH)]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # ── Load Model ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
        max_model_len=32768,
        trust_remote_code=True,
        max_num_seqs=192,
        max_num_batched_tokens=32768,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )
    print("Model loaded.")

    # ── Generate Responses ────────────────────────────────────────────────────
    subset = data[:NUM_QUESTIONS] if NUM_QUESTIONS is not None else data
    prompts = []
    for item in subset:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    responses = [out.outputs[0].text.strip() for out in outputs]

    # ── Score Responses ───────────────────────────────────────────────────────
    judger = Judger(strict_extract=False)
    results = []
    for item, response in tqdm(zip(subset, responses), total=len(responses), desc="Scoring"):
        is_mcq    = bool(item.get("options"))
        gold      = item["answer"]
        gold_list = (gold if isinstance(gold, list) else [gold]) if not is_mcq else None
        response  = postprocess_response(response, gold_list=gold_list)

        if is_mcq:
            correct = score_mcq(response, str(gold))
        else:
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        results.append({
            "id":       item.get("id"),
            "is_mcq":   is_mcq,
            "gold":     gold,
            "response": response,
            "correct":  correct,
        })

    print(f"Scoring complete. {len(results)} results.")

    # ── Summary ───────────────────────────────────────────────────────────────
    mcq_res  = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def acc(subset):
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 50)

    # ── Save Results ──────────────────────────────────────────────────────────
    SAVE_EVAL = True   # Set to False when running on the private test set

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for r in results:
            if SAVE_EVAL:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                          "response": r["response"], "correct": r["correct"]}
            else:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(results)} records to {out_path}")
