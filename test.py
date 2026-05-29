import json
import os
from judger import Judger

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = "qwen-math-sft3/merged"
TOKENIZER_ID  = "qwen-math-sft3/final"
# GPU_ID      = "3"
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH      = "results/starter_results.jsonl"
THINKING_PATH    = "results/thinking.jsonl"
OUTPUT_TEXT_PATH = "results/thinking.txt"
MAX_TOKENS  = 16384

# os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import re
import sys
from pathlib import Path
from collections import Counter

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
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
""" Accuracy = 63%
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
"""
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

SYSTEM_PROMPT_MCQ = (
    "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult."
    "As you are an AI Model you should not fall into the common problems faced by AI models such as being overconfident in wrong answers or being misled by irrelevant information."
    "You should answer the questions like a professional mathematician who is careful and meticulous in solving problems."
    "Read the problem and the answer choices below, then select the single best answer from the options only."
    "If your answer does not match any of the options, you should choose the option that is closest to your answer."
    "For answers that include constants like pi or e, if none of the options match you can input the values of such constants (e.g. 3.14159 for pi) and choose the closest option."
    "The most important thing is to put your final answer inside \\boxed{} without failure."
    "Output ONLY THE LETTER, NOT VALUE of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
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
    tokenizer=TOKENIZER_ID,
    enforce_eager=True,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.9,
    max_model_len=MAX_TOKENS + 2048,
    trust_remote_code=True,
    max_num_seqs=96,
    max_num_batched_tokens=16384,
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
)

print("Model loaded.")
NUM_INPUTS=100
prompts = [build_prompt(d["question"], d.get("options")) for d in data[:NUM_INPUTS]]

print(f"Generating responses for {len(prompts)} questions...")
outputs = llm.generate(prompts, sampling_params)


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


sys.path.insert(0, ".")
judger = Judger(strict_extract=False)


def majority_vote(responses: list[str], is_mcq: bool, gold, judger) -> tuple[bool, str]:
    if is_mcq:
        votes = [extract_letter(r) for r in responses]
        votes = [v for v in votes if v]
        best = Counter(votes).most_common(1)[0][0] if votes else ""
        return best == str(gold).strip().upper(), best
    else:
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


all_responses = [
    [candidate.text.strip() for candidate in out.outputs]
    for out in outputs
]

all_thinking = [
    [getattr(candidate, "reasoning_content", None) or "" for candidate in out.outputs]
    for out in outputs
]

results = []
for item, candidates, thinking in tqdm(zip(data[:NUM_INPUTS], all_responses, all_thinking), total=len(data), desc="Scoring"):
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]

    correct, best_answer = majority_vote(candidates, is_mcq, gold, judger)

    results.append({
        "id":       item.get("id"),
        "is_mcq":   is_mcq,
        "gold":     gold,
        "response": best_answer,
        "correct":  correct,
        "thinking": thinking,
    })

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

SAVE_EVAL = True

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

think_path = Path(THINKING_PATH)
think_path.parent.mkdir(parents=True, exist_ok=True)
with open(think_path, "w") as f:
    for r in results:
        f.write(json.dumps({"id": r["id"], "thinking": r["thinking"]}) + "\n")
print(f"Saved thinking to {think_path}")

text_path = Path(OUTPUT_TEXT_PATH)
text_path.parent.mkdir(parents=True, exist_ok=True)
with open(text_path, "w") as f:
    for i, r in enumerate(results):
        item = data[i]
        f.write("=" * 60 + "\n")
        f.write(f"Q{i} ID: {r['id']}\n")
        f.write(f"QUESTION:\n{item['question']}\n")
        for j, (thinking, response) in enumerate(zip(r["thinking"], all_responses[i])):
            prefix = f"\nCANDIDATE {j}" if len(r["thinking"]) > 1 else ""
            f.write(f"{prefix}\nTHINKING:\n{thinking}\n")
            f.write(f"\nMODEL OUTPUT:\n{response}\n")
print(f"Saved text output to {text_path}")
