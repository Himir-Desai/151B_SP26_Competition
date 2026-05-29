import json
import re
import sys
from pathlib import Path

from judger import Judger
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_ID         = "qwen-math-sft2/merged"
DATA_PATH        = "data/public.jsonl"
OUTPUT_PATH      = "results-himir/starter_results.jsonl"
THINKING_PATH    = "results-himir/thinking.jsonl"
OUTPUT_TEXT_PATH = "results-himir/thinking.txt"

MCQ_MAX_TOKENS      = 16384
FRQ_MAX_TOKENS      = 16384
MCQ_THINKING_BUDGET = 14336
FRQ_THINKING_BUDGET = 16384-2048

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


def frq_postprocessing(output):
    r = output
    if "infty" in r.lower():
        print("Infty seen")
        r = re.sub("infty", "infinity", r, flags=re.IGNORECASE)
    return '\n'.join(r.split('\n')[:-4]).replace("oxed{\"\"","")+'\n'.join(r.split('\n')[-4:])
    



# ── Main ───────────────────────────────────────────────────────────────────

def main():
    data = [json.loads(line) for line in open(DATA_PATH)]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    budget_tokens = {
        "mcq_max_tokens":      MCQ_MAX_TOKENS,
        "frq_max_tokens":      FRQ_MAX_TOKENS,
        "mcq_thinking_budget": MCQ_THINKING_BUDGET,
        "frq_thinking_budget": FRQ_THINKING_BUDGET,
    }

    llm = LLM(
        model=MODEL_ID,
        enforce_eager=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        max_model_len=max(budget_tokens["mcq_max_tokens"], budget_tokens["frq_max_tokens"]) + 2048,
        trust_remote_code=True,
        max_num_seqs=96,
        max_num_batched_tokens=16384,
        reasoning_config=ReasoningConfig(
            reasoning_start_str="<think>",
            reasoning_end_str="</think>",
        ),
    )
    print("Model loaded.")

    mcq_sampling_params = SamplingParams(
        max_tokens=MCQ_MAX_TOKENS,
        temperature=0.1,
        top_p=0.95,
        top_k=20,
        min_p=0.05,
        presence_penalty=0.0,
        repetition_penalty=1,
        thinking_token_budget=MCQ_THINKING_BUDGET,
    )
    frq_sampling_params = SamplingParams(
    max_tokens=FRQ_MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    thinking_token_budget=FRQ_THINKING_BUDGET,
)

    # Replicate test_himir.py: run first 100 MCQ questions only
    # mcq_only = [d for d in data if d.get("options")]
    run_pipeline(
        model=llm,
        dataset=data,
        n=100,
        budget_tokens=budget_tokens,
        mcq_sampling_params=mcq_sampling_params,
        frq_sampling_params=frq_sampling_params,
        tokenizer=tokenizer,
        output_paths={
            "results": OUTPUT_PATH,
            "thinking": THINKING_PATH,
            "text":     OUTPUT_TEXT_PATH,
        },
    )

# ── Pipeline ───────────────────────────────────────────────────────────────

def run_pipeline(
    model,
    dataset,
    n,
    budget_tokens,
    mcq_sampling_params,
    frq_sampling_params,
    tokenizer,
    mcq_system_prompt=None,
    frq_system_prompt=None,
    output_paths=None,
):
    """
    Run complete MCQ + FRQ inference on the first n items of dataset.

    budget_tokens keys: mcq_max_tokens, frq_max_tokens,
                        mcq_thinking_budget, frq_thinking_budget
    output_paths keys:  results, thinking, text  (all optional, have defaults)
    mcq_system_prompt / frq_system_prompt default to module-level constants.
    """
    mcq_prompt = mcq_system_prompt or SYSTEM_PROMPT_MCQ
    frq_prompt = frq_system_prompt or SYSTEM_PROMPT_MATH

    subset    = dataset[:n]
    mcq_items = [(i, d) for i, d in enumerate(subset) if     d.get("options")]
    frq_items = [(i, d) for i, d in enumerate(subset) if not d.get("options")]
    print(f"Pipeline: {len(mcq_items)} MCQ + {len(frq_items)} FRQ (of {len(subset)} total)")

    responses = [None] * len(subset)

    if mcq_items:
        mcq_responses = run_mcq_questions(
            model=model,
            mcq_system_prompt=mcq_prompt,
            input_prompts=[d for _, d in mcq_items],
            tokenizer=tokenizer,
            sampling_params=mcq_sampling_params,
        )
        for (i, _), resp in zip(mcq_items, mcq_responses):
            responses[i] = resp

    if frq_items:
        frq_responses = run_frq_questions(
            model=model,
            frq_system_prompt=frq_prompt,
            input_prompts=[d for _, d in frq_items],
            tokenizer=tokenizer,
            sampling_params=frq_sampling_params,
        )
        for (i, _), resp in zip(frq_items, frq_responses):
            responses[i] = resp

    judger  = Judger(strict_extract=False)
    results = _score_and_collect(subset, responses, judger)
    _print_and_save(results, output_paths or {})


# ── Question runners ───────────────────────────────────────────────────────

def run_mcq_questions(
    model,
    mcq_system_prompt,
    input_prompts,
    tokenizer,
    sampling_params,
    postprocess_fn=lambda x: x,
):
    """
    Run MCQ inference.

    input_prompts: list of question dicts with 'question' and 'options' keys.
    postprocess_fn: applied to each raw response string; defaults to identity.
    Returns list of response strings.
    """
    prompts = [
        build_prompt(d["question"], mcq_system_prompt, tokenizer, d.get("options"))
        for d in input_prompts
    ]
    print(f"Generating {len(prompts)} MCQ responses...")
    outputs = model.generate(prompts, sampling_params)
    return [postprocess_fn(out.outputs[0].text.strip()) for out in outputs]


def run_frq_questions(
    model,
    frq_system_prompt,
    input_prompts,
    tokenizer,
    sampling_params,
    postprocess_fn=frq_postprocessing,
):
    """
    Run FRQ inference.

    input_prompts: list of question dicts with a 'question' key.
    postprocess_fn: applied to each raw response string; defaults to identity.
    Returns list of response strings.
    """
    prompts = [
        build_prompt(d["question"], frq_system_prompt, tokenizer)
        for d in input_prompts
    ]
    print(f"Generating {len(prompts)} FRQ responses...")
    outputs = model.generate(prompts, sampling_params)
    return [postprocess_fn(out.outputs[0].text.strip()) for out in outputs]


# ── Helpers ────────────────────────────────────────────────────────────────

def build_prompt(question, system_prompt, tokenizer, options=None):
    if options:
        letters      = "ABCDEFGHIJ"
        options_text = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
        user_content = f"{question}\n\nOptions:\n{options_text}"
    else:
        user_content = question
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def extract_letter(text):
    # Use last match to avoid being tricked by intermediate \boxed{x} variable references
    m = re.findall(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m[-1].upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def _split_thinking(response):
    if "</think>" in response:
        thinking, resp = response.split("</think>", 1)
        return thinking.strip(), resp.strip()
    return "", response.strip()


def _score_and_collect(items, responses, judger):
    results = []
    for item, response in tqdm(zip(items, responses), total=len(items), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        gold   = item["answer"]
        thinking, clean_response = _split_thinking(response)

        if is_mcq:
            extracted = extract_letter(clean_response) or extract_letter(response)
            correct   = (extracted == str(gold).strip().upper())
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            m         = re.findall(r"\\boxed\{([^}]{0,80})\}", clean_response)
            extracted = m[-1].strip() if m else ""
            try:
                correct = judger.auto_judge(
                    pred=response, gold=gold_list, options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        results.append({
            "id":            item.get("id"),
            "is_mcq":        is_mcq,
            "gold":          gold,
            "extracted":     extracted,
            "correct":       correct,
            "thinking":      thinking,
            "response":      clean_response,
            "full_response": response,
            "question":      item["question"],
        })
    return results


def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0


def _print_and_save(results, output_paths, save_eval=True):
    mcq_res  = [r for r in results if     r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]
    print(f"Scoring complete. {len(results)} results.")
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 50)

    out_path = Path(output_paths.get("results", "results/starter_results.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            record = {
                "id":       r["id"],
                "is_mcq":   r["is_mcq"],
                "gold":     r["gold"],
                "extracted": r["extracted"],
                "correct":  r["correct"],
            }
            if save_eval:
                record["response"]    = r["response"]
                record["think_words"] = len(r["thinking"].split())
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(results)} records to {out_path}")

    think_path = Path(output_paths.get("thinking", "results/thinking.jsonl"))
    think_path.parent.mkdir(parents=True, exist_ok=True)
    with open(think_path, "w") as f:
        for r in results:
            f.write(json.dumps({"id": r["id"], "thinking": r["thinking"]}) + "\n")
    print(f"Saved thinking to {think_path}")

    text_path = Path(output_paths.get("text", "results/thinking.txt"))
    text_path.parent.mkdir(parents=True, exist_ok=True)
    with open(text_path, "w") as f:
        for i, r in enumerate(results):
            status = "CORRECT" if r["correct"] else "WRONG"
            qtype  = "MCQ"     if r["is_mcq"]  else "FRQ"
            tw = len(r["thinking"].split())
            rw = len(r["response"].split())
            f.write("=" * 70 + "\n")
            f.write(
                f"Q{i:03d} | ID:{r['id']} | {qtype} | {status} | "
                f"gold={r['gold']} | got={r['extracted'] or 'NONE'}\n"
            )
            f.write(f"       think={tw}w  response={rw}w\n")
            f.write("=" * 70 + "\n")
            f.write(f"QUESTION:\n{r['question']}\n")
            f.write(f"\nTHINKING ({tw} words):\n{r['thinking']}\n")
            f.write(f"\nRESPONSE ({rw} words):\n{r['response']}\n\n")
    print(f"Saved text output to {text_path}")


if __name__ == "__main__":
    main()

