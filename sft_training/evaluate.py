"""
Step 3 — Evaluate the fine-tuned model.

Loads base model + LoRA adapter, runs inference on public.jsonl,
scores with Judger, and prints accuracy breakdown vs baseline.

Run after train.py.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

# ── Load config ───────────────────────────────────────────────────────────────
cfg_path = Path(__file__).parent / "config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["hardware"]["gpu_id"]

sys.path.insert(0, str(Path(__file__).parent.parent))

eval_cfg = cfg["evaluation"]
model_cfg = cfg["model"]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_MATH = (
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
    "7. Make sure to use the variables as asked in question, and maintain proper bracketing. While sin theta might seem conventional, if the questions explicitly asks to use t, answer with sin(t)\n\n"
    "8. When a function (atan, arctan, ln, log, sin, cos, etc.) is applied to a specific value, preserve the function form exactly — do NOT evaluate to a decimal. "
    "E.g., keep atan(4.76) not 1.360, keep ln(0.5) not -0.693, keep arctan(3) not 1.249.\n\n"
    "9. When the answer IS a decimal number (i.e. the problem asks you to compute a numerical value), "
    "preserve full precision — give at least 4 decimal places and do not round. "
    "E.g., give 143.2242 not 143.2, give 2.2892 not 2.3, give 18.8105 not 18.81."
)
SYSTEM_MCQ = (
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


def build_prompt(question, options: Optional[list]):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
        return SYSTEM_MCQ, f"{question}\n\nOptions:\n{opts}"
    return SYSTEM_MATH, question


def extract_letter(text):
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from judger import Judger

    # ── Load data ─────────────────────────────────────────────────────────────
    data = [json.loads(l) for l in open(eval_cfg["data_path"])]
    n = eval_cfg.get("num_questions")
    if n:
        data = data[:n]
    print(f"Evaluating on {len(data)} questions")

    # ── Load model with LoRA adapter via vLLM ─────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["adapter_dir"], trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_cfg["model_id"],
        enable_lora=True,
        max_lora_rank=cfg["lora"]["r"],
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=eval_cfg["gpu_memory_utilization"],
        max_model_len=eval_cfg["max_model_len"],
        trust_remote_code=True,
        max_num_seqs=eval_cfg["max_num_seqs"],
        max_num_batched_tokens=eval_cfg["max_model_len"],
    )

    lora_request = LoRARequest("sft_adapter", 1, model_cfg["adapter_dir"])

    sampling_params = SamplingParams(
        max_tokens=eval_cfg["max_new_tokens"],
        temperature=eval_cfg["temperature"],
        top_p=eval_cfg["top_p"],
        top_k=eval_cfg["top_k"],
        min_p=eval_cfg.get("min_p", 0.0),
    )

    # ── Build prompts — no thinking_budget, matches starter.py ───────────────
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    responses = [out.outputs[0].text.strip() for out in outputs]

    # ── Score ─────────────────────────────────────────────────────────────────
    judger = Judger(strict_extract=False)
    results = []

    for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        gold = item["answer"]

        if is_mcq:
            correct = extract_letter(response) == str(gold).strip().upper()
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        results.append({
            "id": item.get("id"),
            "is_mcq": is_mcq,
            "gold": gold,
            "response": response,
            "correct": correct,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    mcq_res  = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def acc(subset):
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    print("=" * 50)
    print("SFT EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 50)
    print("Baseline (starter.py): MCQ 78.9% | Free-form 51.6% | Overall 62.0%")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(eval_cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved → {out_path}")
