import json
import os
from judger import Judger

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "qwen-math-sft2/merged"
# GPU_ID      = "3"
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH      = "results-himir/starter_results.jsonl"
THINKING_PATH    = "results-himir/thinking.jsonl"
OUTPUT_TEXT_PATH = "results-himir/thinking.txt"

MCQ_MAX_TOKENS      = 16384
FRQ_MAX_TOKENS      = 16384
MCQ_THINKING_BUDGET = 14336
FRQ_THINKING_BUDGET = 8192

# os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import re
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig
from tqdm import tqdm


data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

""" Accuracy = 60%
SYSTEM_PROMPT_MATH = (
    "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult."
    "You should answer the questions like a professional mathematician who is careful and meticulous in solving problems. "
    "Formatting Instructions:\n"
    "1. Do not round your final answer. Leave complicated answers unevaluated, e.g. leave 325*(1+325) as is instead of evaluating to 105950 or leave (1/2)^[(1999-1963)/31] as is.\n" 
    "2. Prefer answering in decimal form and round only if necessary. For example these are some expected rounded answers [\"62.7777777777778\", \"335.927777777778\", \"604.67\"].\n"
    "3. For questions that demand fractional answers like 5/8 output \\boxed{5/8} as it is and not \\boxed{frac{5}{8}}\n"
    "4. For answers that include expressions, like 2x^2+3x+4 answer like the following 2*x^2+3*x+4 instead of 2x^2+3x+4\n"
    "5. If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, e.g. if and only if the answer is 3 and/or 7 then you will output \\boxed{3, 7}.\n"
    "6. Never add the characters { } in the \\boxed{} answer\n"
    "The most important thing is to put your final answer inside \\boxed{} without failure using the formatting instructions above."
    "Only and only output one final \\boxed{} answer at the end with answers to all the parts of the question if there are any, or multiple answers for the same question if there are multiple. Do not output any other \\boxed{} that is not the final answer to the question. "
)

"""

""" Accuracy sft 2 = 67% Accuracy grpo = 63%
SYSTEM_PROMPT_MATH = (
    "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult."
    "You should answer the questions like a professional mathematician who is careful and meticulous in solving problems. "
    "The most important thing is to put your final answer inside \\boxed{} without failure using the formatting instructions above."
    "Follow the following instructions in given priority order"
    "Formatting Instructions:\n"
    "1. Put your final answer at the end inside \\boxed{}. You are NOT PERMITTED TO USE BOXED MORE THAN ONCE SO BE CAREFUL AND USE IT AT THE END."
    "2. Never add the characters { } inside the \\boxed{} answer. The only allowed { } are for writting out the \\boxed{} answer.\n"
    "3. Do not round your final answer. Leave complicated answers unevaluated, e.g. leave 325*(1+325) as is or leave (1/2)^[(1999-1963)/31] as is.\n" 
    "4. Prefer answering in decimal form and round only if necessary. For example these are some expected rounded answers [\"62.7777777777778\", \"335.927777777778\", \"604.67\"].\n"
    "5. For questions that demand fractional answers like 5/8 output 5/8 as it is and not frac{5}{8}\n"
    "6. For answers that include algaebraic expressions, like 2x^2+3x+4 answer like the following 2*x^2+3*x+4 instead of 2x^2+3x+4\n"
    "7. If the problem has multiple sub-answers, separate them by commas inside a single boxed, e.g. if and only if the answer is 3 and/or 7 then you will output 3, 7.\n"
    "8. The formatting inside the boxed should not be in latek, so dont use stuff like {}"
)
"""

""" frq accuracy = 71-73%"""
SYSTEM_PROMPT_MATH = (
    "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult using a meticulous step by step process. "
    "You also give weightage to thought to how the answer should be formatted based on the instructions\n"
    "Follow the following instructions in given priority order"
    "Formatting Instructions:\n"
    "1. Put ALL your final answers at the end inside A SINGLE \\boxed{}. If the problem has multiple sub-answers, separate them by commas inside a SINGLE BOXED, e.g. if and only if the answer is 3 and/or 7 then you will output 3, 7.\n. You are NOT PERMITTED TO USE BOXED MORE THAN ONCE SO BE CAREFUL AND USE IT AT THE END.\n"
    "2. Try to follow formatting instructions and aesthetic of the question. If the below rules contradict with how the question has been presented or how the question requests a response, prefer the rules in question\n"
    "3. Never add the characters { } inside the \\boxed{} answer. The only allowed { } are for writting out the \\boxed{} answer.\n"
    "4. Never use LaTex format for the final answer. Write it as a number of an expression or whatever the question"
    "4. Do NOT ROUND your final answer. Leave complicated answers unevaluated, e.g. leave 325*(1+325) as is or leave (1/2)^[(1999-1963)/31] as is. Do not put in the values of pi and e. Leave them as the words pi and e\n" 
    "5. Prefer answering in DECIMAL FORM and round only if necessary. For example these are some expected rounded answers [\"62.7777777777778\", \"335.927777777778\", \"604.67\"].\n"
    "6. For questions that demand fractional answers like 5/8 output 5/8 as it is and not frac{5}{8}\n"
    "7. For answers that include algaebraic expressions, multiplication should be written explicitly using *. 2x should be written as 2*x. Example: 2x^2+3x+4 should be written as 2*x^2+3*x+4\n"
    "9. We write infinity as infinity and not $\infty\n"
)
# """

""" Accuracy = 62%
SYSTEM_PROMPT_MATH = (
    "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult."
    "You should answer the questions like a professional mathematician who is careful and meticulous in solving problems."
    "Follow these steps:"
    "STEP 1: First read the entire question and evaluate if its a multipart question or a single part question."
    "STEP 2A: Only do this step if the question is a single part question. Solve the question carefully and follow the formatting instructions below skipping STEP 2B. After solving the question go to STEP 3A."
    "STEP 2B: Only do this step if the question is a multipart question. Solve each part of the question individually and follow the formatting instructions below for each part. After solving all parts go to STEP 3B."
    "Formatting Instructions:\n"
    "1. Do not round your final answer. Leave complicated answers unevaluated, e.g. leave 325*(1+325) as is instead of evaluating to 105950 or leave (1/2)^[(1999-1963)/31] as is.\n" 
    "2. Prefer answering in decimal form and round only if necessary. For example these are some expected rounded answers [\"62.7777777777778\", \"335.927777777778\", \"604.67\"].\n"
    "3. For questions that demand fractional answers like 5/8 output 5/8 as it is and not \\frac{5}{8}\n"
    "4. For answers that include algaebraic expressions, like 2x^2+3x+4 answer like the following 2*x^2+3*x+4 instead of 2x^2+3x+4\n"
    "5. Once you have a final answer put write it in the following format \\box{answer}."
    "6. Never add the characters { } inside the \\box{} answer. The only allowed { } are for writting out the \\box{} answer.\n"
    "STEP 3A: Only do this step if the question is a single part question. For your final \\box{answer} output \\boxed{answer} at the end outside the thinking block."
    "STEP 3B: Only do this step if the question is a multipart question. You have answers to each of the part as \\box{answer1}, \\box{answer2}, etc. Now output a final answer at the end in the following format \\boxed{answer1, answer2, ...} with all the answers to all the parts of the entire question separated by commas inside a single \\boxed{} at the end outside the thinking block."
)
"""
"""
SYSTEM_PROMPT_MCQ = (
    "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult."
    "As you are an AI Model you should not fall into the common problems faced by AI models such as being overconfident in wrong answers or being misled by irrelevant information."
    "You should answer the questions like a professional mathematician who is careful and meticulous in solving problems."
    "Read the problem and the answer choices below, then select the single best answer from the options only."
    "If your answer does not match any of the options, you should choose the option that is closest to your answer."
    "For answers that include constants like pi or e, if none of the options match you can input the values of such constants (e.g. 3.14159 for pi) and choose the closest option."
    "The most important thing is to put your final answer inside \\boxed{} without failure."
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)
"""

SYSTEM_PROMPT_MCQ = (
    "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult. Solve meticulously and step by step"
    "Read the problem and the answer choices below, then select the single best answer from the options only."
    "If your answer does not match any of the options, you should choose the option that is closest to your answer."
    "For answers that include constants like pi or e, if none of the options match you can input the values of such constants (e.g. 3.14159 for pi) and choose the closest option."
    "Double check your answer by testing your solution in some form"
    "The most important thing is to put your final answer inside a single \\boxed{} without failure."
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
    "If there are multiple parts in the question, put all of your answers inside a single \\boxed{} separated by comma, example \\boxed{C,F}"
)

"""
SYSTEM_PROMPT_MCQ = (
    "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult. Solve meticulously and step by step"
    "Read the problem and the answer choices below, then select the single best answer from the options only."
    "If your answer does not match any of the options, you should choose the option that is closest to your answer."
    "For answers that include constants like pi or e, if none of the options match you can input the values of such constants (e.g. 3.14159 for pi) and choose the closest option."
    "Double check your answer by testing your solution in some form"
    "The most important thing is to put your final answer inside a single \\boxed{} without failure."
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
    "If there are multiple parts in the question, put all of your answers inside a single \\boxed{} separated by comma, example \\boxed{C,F}"
)
"""
"""
SYSTEM_PROMPT_MCQ = (
    "You are a world-class mathematician solving a multiple-choice problem.\n"
    "Use this strict two-phase approach:\n\n"
    "PHASE 1 — SOLVE INDEPENDENTLY:\n"
    "Completely ignore the answer options. Solve the problem from scratch as if it were open-ended, "
    "working step by step to a precise answer. Do not look at or mention the options during this phase.\n\n"
    "PHASE 2 — SELECT:\n"
    "Once you have your answer, check the options in this order:\n"
    "1. EXACT or EQUIVALENT: Does any option equal your answer through algebra or simplification (e.g., 2/3·ln9 = 4/3·ln3)? If your answer involves pi or e, substitute pi≈3.14159 or e≈2.71828 and check for a close match.\n"
    "2. QUALITATIVE: If your Phase 1 result is 'diverges', 'undefined', 'no solution', 'does not exist', or similar, look for the option that captures that property — it may be worded as 'diverges', 'DNE', '∞', or a descriptive phrase rather than a number.\n"
    "3. CLOSEST: Only if neither of the above apply, pick the numerically closest option.\n"
    "If nothing matches any of the above, briefly reconsider whether Phase 1 made an error, then commit.\n"
    "Commit to one letter — do not re-solve.\n\n"
    "MANDATORY OUTPUT RULE:\n"
    "The absolute last thing you write must be \\boxed{X} where X is the chosen letter, e.g. \\boxed{C}. "
    "You may only write \\boxed{} once, and it must be the final line of your response. "
    "No matter what, always commit to one letter and box it."
)
"""

"""
SYSTEM_PROMPT_MCQ = (
    "You are a world-class mathematician solving a multiple-choice problem.\n"
    "Use this strict two-phase approach:\n\n"
    "PHASE 1 — SOLVE INDEPENDENTLY:\n"
    "Ignore the answer options. Solve the problem from scratch as if it were open-ended, "
    "working step by step to a precise answer. Do not look at or mention the options during this phase.\n\n"
    "PHASE 2 — SELECT:\n"
    "Once you have your answer, check the options in this order:\n"
    "1. EXACT or EQUIVALENT: Does any option equal your answer through algebra or simplification (e.g., 2/3·ln9 = 4/3·ln3)? If your answer involves pi or e, substitute pi≈3.14159 or e≈2.71828 and check for a close match.\n"
    "2. QUALITATIVE: If your Phase 1 result is 'diverges', 'undefined', 'no solution', 'does not exist', or similar, look for the option that captures that property — it may be worded as 'diverges', 'DNE', '∞', or a descriptive phrase rather than a number.\n"
    "3. CLOSEST: Only if neither of the above apply, pick the numerically closest option.\n"
    "If nothing matches any of the above, briefly reconsider whether Phase 1 made an error, then commit.\n"
    "Commit to one letter — do not re-solve.\n\n"
    "MANDATORY OUTPUT RULE:\n"
    "The absolute last thing you write must be \\boxed{X} where X is the chosen option letter and not the letter X itself"
    "You may only write \\boxed{} once, and it must be the final line of your response. "
    "No matter what, always commit to one letter and box it."
    "After thinking your response should only and only have the final answer in \\boxed{} no need to give any reasoning for your answer outside thinking."
)
"""

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



tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token


def build_prompt(question, options=None):
    if options:
        letters = "ABCDEFGHIJ"
        options_text = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
        user_content  = f"{question}\n\nOptions:\n{options_text}"
        system_prompt = SYSTEM_PROMPT_MCQ
    else:
        user_content  = question
        system_prompt = SYSTEM_PROMPT_MATH

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


llm = LLM(
    model=MODEL_ID,
    enforce_eager=True,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.9,
    max_model_len=max(MCQ_MAX_TOKENS, FRQ_MAX_TOKENS) + 2048,
    trust_remote_code=True,
    max_num_seqs=96,
    max_num_batched_tokens=16384,
    reasoning_config=ReasoningConfig(
        reasoning_start_str="<think>",
        reasoning_end_str="</think>",
    )
)

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

print("Model loaded.")

sys.path.insert(0, ".")
judger = Judger(strict_extract=False)
SAVE_EVAL = True


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0


def _split_thinking(response: str):
    """Split vLLM output into (thinking, response) on </think>."""
    if "</think>" in response:
        thinking, resp = response.split("</think>", 1)
        return thinking.strip(), resp.strip()
    return "", response.strip()


def _score_and_collect(items, responses, all_thinking):
    results = []
    for i, (item, response) in enumerate(tqdm(zip(items, responses), total=len(items), desc="Scoring")):
        is_mcq    = bool(item.get("options"))
        gold      = item["answer"]
        thinking, clean_response = _split_thinking(response)

        if is_mcq:
            extracted = extract_letter(clean_response) or extract_letter(response)
            correct   = (extracted == str(gold).strip().upper())
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            m = re.findall(r"\\boxed\{([^}]{0,80})\}", clean_response)
            extracted = m[-1].strip() if m else ""
            try:
                correct = judger.auto_judge(
                    pred=response, gold=gold_list, options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        results.append({
            "id": item.get("id"), "is_mcq": is_mcq, "gold": gold,
            "extracted": extracted, "correct": correct,
            "thinking": thinking, "response": clean_response,
            "full_response": response, "question": item["question"],
        })
    return results


def _print_and_save(results):
    mcq_res  = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]
    print(f"Scoring complete. {len(results)} results.")
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 50)

    # ── JSONL results ────────────────────────────────────────────
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                      "extracted": r["extracted"], "correct": r["correct"]}
            if SAVE_EVAL:
                record["response"]  = r["response"]
                record["think_words"] = len(r["thinking"].split())
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(results)} records to {out_path}")

    # ── Thinking JSONL (raw, for debugging) ──────────────────────
    think_path = Path(THINKING_PATH)
    think_path.parent.mkdir(parents=True, exist_ok=True)
    with open(think_path, "w") as f:
        for r in results:
            f.write(json.dumps({"id": r["id"], "thinking": r["thinking"]}) + "\n")
    print(f"Saved thinking to {think_path}")

    # ── Human-readable text output ───────────────────────────────
    text_path = Path(OUTPUT_TEXT_PATH)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    with open(text_path, "w") as f:
        for i, r in enumerate(results):
            status   = "CORRECT" if r["correct"] else "WRONG"
            qtype    = "MCQ" if r["is_mcq"] else "FRQ"
            tw       = len(r["thinking"].split())
            rw       = len(r["response"].split())
            f.write("=" * 70 + "\n")
            f.write(f"Q{i:03d} | ID:{r['id']} | {qtype} | {status} | "
                    f"gold={r['gold']} | got={r['extracted'] or 'NONE'}\n")
            f.write(f"       think={tw}w  response={rw}w\n")
            f.write("=" * 70 + "\n")
            f.write(f"QUESTION:\n{r['question']}\n")
            f.write(f"\nTHINKING ({tw} words):\n{r['thinking']}\n")
            f.write(f"\nRESPONSE ({rw} words):\n{r['response']}\n\n")
    print(f"Saved text output to {text_path}")


def run_inference(n):
    """Run on the first n questions (mixed MCQ + FRQ)."""
    subset       = data[:n]
    mcq_indices  = [i for i, d in enumerate(subset) if d.get("options")]
    free_indices = [i for i, d in enumerate(subset) if not d.get("options")]

    mcq_prompts  = [build_prompt(subset[i]["question"], subset[i].get("options")) for i in mcq_indices]
    free_prompts = [build_prompt(subset[i]["question"]) for i in free_indices]

    print(f"Generating responses for {len(mcq_prompts)} MCQ and {len(free_prompts)} FRQ questions...")
    mcq_outputs  = llm.generate(mcq_prompts, mcq_sampling_params)
    free_outputs = llm.generate(free_prompts, frq_sampling_params)

    raw_outputs = [None] * n
    for idx, out in zip(mcq_indices, mcq_outputs):
        raw_outputs[idx] = out
    for idx, out in zip(free_indices, free_outputs):
        raw_outputs[idx] = out

    responses    = [out.outputs[0].text.strip() for out in raw_outputs]
    all_thinking = [[getattr(c, "reasoning_content", None) or "" for c in out.outputs] for out in raw_outputs]

    results = _score_and_collect(subset, responses, all_thinking)
    _print_and_save(results)


def run_mcq(n):
    """Run on the first n MCQ questions only."""
    mcq_data = [d for d in data if d.get("options")][:n]
    prompts  = [build_prompt(d["question"], d.get("options")) for d in mcq_data]

    print(f"Generating responses for {len(prompts)} MCQ questions...")
    raw_outputs = llm.generate(prompts, mcq_sampling_params)

    responses    = [out.outputs[0].text.strip() for out in raw_outputs]
    all_thinking = [[getattr(c, "reasoning_content", None) or "" for c in out.outputs] for out in raw_outputs]

    results = _score_and_collect(mcq_data, responses, all_thinking)
    _print_and_save(results)


def run_frq(n):
    """Run on the first n FRQ questions only."""
    frq_data = [d for d in data if not d.get("options")][:n]
    prompts  = [build_prompt(d["question"]) for d in frq_data]

    print(f"Generating responses for {len(prompts)} FRQ questions...")
    raw_outputs = llm.generate(prompts, frq_sampling_params)

    responses    = [out.outputs[0].text.strip() for out in raw_outputs]
    all_thinking = [[getattr(c, "reasoning_content", None) or "" for c in out.outputs] for out in raw_outputs]

    results = _score_and_collect(frq_data, responses, all_thinking)
    _print_and_save(results)


run_mcq(100)



