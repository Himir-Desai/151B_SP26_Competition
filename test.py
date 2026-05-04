import json
import os
from judger import Judger

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "1"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH = "results/starter_results.jsonl"
MAX_TOKENS  = 81920

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import re
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from vllm.config import ReasoningConfig


data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

# Preview one MCQ and one free-form item
mcq_sample  = next(d for d in data if d.get("options"))
free_sample = next(d for d in data if not d.get("options"))

print("\n── MCQ sample ──")
print(json.dumps(mcq_sample, indent=2))
print("\n── Free-form sample ──")
print(json.dumps(free_sample, indent=2))

SYSTEM_PROMPT_MATH = (
    "You are a nobel prize winning mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are a nobel prize winning mathematician."
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# Verify with samples
for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
    sys_p, usr_p = build_prompt(item["question"], item.get("options"))
    print(f"── {label} user prompt (first 200 chars) ──")
    print(usr_p[:200], "...\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token


llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.9,
    max_model_len=MAX_TOKENS+2048,
    trust_remote_code=True,
    max_num_seqs=96,
    max_num_batched_tokens=16224,
    reasoning_config=ReasoningConfig(
        reasoning_start_str="<think>",
        reasoning_end_str="I have to now answer based on my thinking</think>",
    )
)

sampling_params = SamplingParams(
    # n=12,
    max_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    thinking_token_budget=1024
)

print("Model loaded.")

# Build prompts for first 5 entries
# Build prompts for first 5 entries 
NUM_INPUTS = 64
responses = []


prompts = []
for item in data[:NUM_INPUTS]:
    system, user = build_prompt(item["question"], item.get("options"))
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
        {"role": "user",   "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts.append(prompt_text)

# Generate
print(f"Generating responses for {len(prompts)} questions...")
outputs = llm.generate(prompts, sampling_params=sampling_params)

responses = [out.outputs[0].text.strip() for out in outputs]

# Preview first 3
# for i in range(min(3, len(responses))):
#     print(f"\n── Response {i} (id={data[i].get('id')}) ──")
#     print(responses[i][:800], "..." if len(responses[i]) > 800 else "")
def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


# Load Judger for free-form scoring
sys.path.insert(0, ".")
judger = Judger(strict_extract=False)

from collections import Counter

def majority_vote(responses: list[str], is_mcq: bool, gold, judger) -> tuple[bool, str]:
    """Extract answers from all candidates and return (correct, best_answer)."""
    if is_mcq:
        votes = [extract_letter(r) for r in responses]
        votes = [v for v in votes if v]  # drop blanks
        best = Counter(votes).most_common(1)[0][0] if votes else ""
        return best == str(gold).strip().upper(), best
    else:
        # For free-form, collect all extracted boxed answers and majority vote
        extracted = []
        for r in responses:
            m = re.search(r"\\boxed\{([^}]+)\}", r)
            if m:
                extracted.append(m.group(1).strip())
        
        if not extracted:
            return False, ""
        
        best = Counter(extracted).most_common(1)[0][0]
        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            correct = judger.auto_judge(
                pred=f"\\boxed{{{best}}}",
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        except Exception:
            correct = False
        return correct, best


# Extract responses — now outputs[i].outputs has n=8 candidates
all_responses = [
    [candidate.text.strip() for candidate in out.outputs]
    for out in outputs
]

results = []
for item, candidates in tqdm(zip(data, all_responses), total=len(data), desc="Scoring"):
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]

    correct, best_answer = majority_vote(candidates, is_mcq, gold, judger)

    results.append({
        "id":       item.get("id"),
        "is_mcq":   is_mcq,
        "gold":     gold,
        "response": best_answer,   # save the winning answer
        "correct":  correct,
    })

# results = []
# for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
#     is_mcq = bool(item.get("options"))
#     gold   = item["answer"]

#     if is_mcq:
#         correct = score_mcq(response, str(gold))
#     else:
#         gold_list = gold if isinstance(gold, list) else [gold]
#         try:
#             correct = judger.auto_judge(
#                 pred=response,
#                 gold=gold_list,
#                 options=[[]] * len(gold_list),
#             )
#         except Exception:
#             correct = False

#     results.append({
#         "id":       item.get("id"),
#         "is_mcq":   is_mcq,
#         "gold":     gold,
#         "response": response,
#         "correct":  correct,
#     })

print(f"Scoring complete. {len(results)} results.")

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