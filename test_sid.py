import json
import os
import re
from judger import Judger
from sympy import sympify
from sympy.parsing.latex import parse_latex

_LATEX_MARKERS = re.compile(r'[\\^{]')


def _is_plain_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def latex_part_to_sympy(s: str) -> str:
    s = s.strip()
    if not s or _is_plain_number(s) or not _LATEX_MARKERS.search(s):
        return s
    try:
        expr = parse_latex(s)
        return str(sympify(str(expr)))
    except Exception:
        return s


def convert_frq_answer(extracted: str, judger_inst) -> str:
    parts = judger_inst.split_by_comma(extracted)
    converted = [latex_part_to_sympy(p) for p in parts]
    return ", ".join(converted) if len(converted) > 1 else converted[0]


# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "qwen-math-sft2/merged"
# GPU_ID      = "3"
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH      = "results/starter_results.jsonl"
THINKING_PATH    = "results/thinking.jsonl"
OUTPUT_TEXT_PATH = "results/thinking.txt"
MAX_TOKENS  = 16384

# os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

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

# """ Accuracy = 66% """
# SYSTEM_PROMPT_MATH = (
#     "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult using a meticulous step by step process. "
#     "You also give weightage to thought to how the answer should be formatted based on the instructions\n"
#     "Follow the following instructions in given priority order"
#     "Formatting Instructions:\n"
#     "1. Put ALL your final answers at the end inside A SINGLE \\boxed{}. If the problem has multiple sub-answers, separate them by commas inside a SINGLE BOXED, e.g. if and only if the answer is 3 and/or 7 then you will output 3, 7.\n. You are NOT PERMITTED TO USE BOXED MORE THAN ONCE SO BE CAREFUL AND USE IT AT THE END.\n"
#     "2. Try to follow formatting instructions and aesthetic of the question. If the below rules contradict with how the question has been presented or how the question requests a response, prefer the rules in question\n"
#     "3. Never add the characters { } inside the \\boxed{} answer. The only allowed { } are for writting out the \\boxed{} answer.\n"
#     "4. Never use LaTex format for the final answer. Write it as a number of an expression or whatever the question"
#     "4. Do NOT ROUND your final answer. Leave complicated answers unevaluated, e.g. leave 325*(1+325) as is or leave (1/2)^[(1999-1963)/31] as is. Do not put in the values of pi and e. Leave them as the words pi and e\n" 
#     "5. Prefer answering in DECIMAL FORM and round only if necessary. For example these are some expected rounded answers [\"62.7777777777778\", \"335.927777777778\", \"604.67\"].\n"
#     "6. For questions that demand fractional answers like 5/8 output 5/8 as it is and not frac{5}{8}\n"
#     "7. For answers that include algaebraic expressions, multiplication should be written explicitly using *. 2x should be written as 2*x. Example: 2x^2+3x+4 should be written as 2*x^2+3*x+4\n"
#     "9. We write infinity as infinity and not $\infty\n"
# )

# SYSTEM_PROMPT_MATH = (
#     "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult using a meticulous step by step process. "
#     "You also give weightage to thought to how the answer should be formatted based on the instructions\n"
#     "Follow the following instructions in given priority order"
#     "Formatting Instructions:\n"
#     "1. Put ALL your final answers at the end inside A SINGLE \\boxed{}. If the problem has multiple sub-answers, separate them by commas inside a SINGLE BOXED, e.g. if and only if the answer is 3 and/or 7 then you will output \\boxed{3, 7}.\n. You are NOT PERMITTED TO USE BOXED MORE THAN ONCE SO BE CAREFUL AND USE IT AT THE END.\n"
#     "2. Try to follow formatting instructions and aesthetic of the question. If the below rules contradict with how the question has been presented or how the question requests a response, prefer the rules in question\n"
#     "3. Never add the characters { } inside the \\boxed{} answer. The only allowed { } are for writting out the \\boxed{} answer.\n"
#     "4. Never use LaTex format for the final answer. Write it as a number of an expression or whatever the question"
#     "4. Do NOT ROUND your final answer. Leave complicated answers unevaluated, e.g. leave 325*(1+325) as is or leave (1/2)^[(1999-1963)/31] as is. Do not put in the values of pi and e. Leave them as the words pi and e\n" 
#     "5. Prefer answering in DECIMAL FORM and round only if necessary. For example these are some expected rounded answers [\"62.7777777777778\", \"335.927777777778\", \"604.67\"].\n"
#     "6. For questions that demand fractional answers like 5/8 output 5/8 as it is and not frac{5}{8}\n"
#     "7. For answers that include algaebraic expressions, multiplication should be written explicitly using *. 2x should be written as 2*x. Example: 2x^2+3x+4 should be written as 2*x^2+3*x+4\n"
#     "9. We write infinity as infinity and not $\infty\n"
# )

# Accuracy = 69%
# SYSTEM_PROMPT_MATH = (
#     "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult using a meticulous step by step process. Try to recheck your answer\n"
#     "THE MOST IMPORTANT STEP: PRESERVE IRRATIONAL NUMBERS LIKE PI AND SQUARE ROOTS, DONT SIMPLIFY IT AT ANY POINT, INCLUDE IN FINAL ANSWER\n"
#     "ALSO PRESERVE FRACTIONS FOR EXAMPLE 7/11\n"
#     "It is totally okay if your answers are long equations like (1/2)^[(1999-1963)/31]\n as long as they are computable"
#     "Follow the following instructions to the tee."
#     "Formatting Instructions:\n"
#     "1. Put ALL your final answers at the end inside A SINGLE \\boxed{}. If the problem has multiple sub-answers, separate them by commas inside a SINGLE BOXED, e.g. if and only if the answer is 3 and/or 7 then you will output \\boxed{3, 7}.\n. You are NOT PERMITTED TO USE BOXED MORE THAN ONCE SO BE CAREFUL AND USE IT AT THE END.\n"
#     "2. Try to follow formatting instructions and aesthetic of the question. If the below rules contradict with how the question has been presented or how the question requests a response, prefer the rules in question\n"
#     "3. Prefer using LaTex format for the final answer. For fractions use \\frac{}, for square root use \\sqrt{}, etc. For constants like pi and e, use the words pi and e. For ln(3.4), leave it as is.  DO NOT USE THEIR ACTUAL VALUES Use the word infinity for representing infinity\n"
#     "4. If you need to write in decimal, make sure your decimal answers should have 9-10 decimal places, even if those extra decimal places are zeroes. THIS IS VERY VERY IMPORTANT\n"
#     "5. Make sure you have answered all of the sub questions in the final boxed and not missed any"
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

SYSTEM_PROMPT_MCQ = (
    "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult. Solve meticulously and step by step"
    "Read the problem and the answer choices below, then select the single best answer from the options only."
    "If your answer does not match any of the options, you should choose the option that is closest to your answer."
    "For answers that include constants like pi or e, if none of the options match you can input the values of such constants (e.g. 3.14159 for pi) and choose the closest option."
    "Double check your answer by testing your solution in some form"
    "Output ONLY THE CORRECT LETTER/LETTERS, NOT VALUES of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
    "If there are multiple parts in the question, put all of your answers inside a single \\boxed{} separated by comma, example \\boxed{C,F}"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token


def build_prompt(question, options=None):
    if options:
        letters = "ABCDEFGHIJ"
        options_text = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
        user_content  = f"{question}\n\nOptions:\n{options_text}\n"
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
    gpu_memory_utilization=0.5,
    max_model_len=MAX_TOKENS + 2048,
    trust_remote_code=True,
    max_num_seqs=96,
    max_num_batched_tokens=16384,
    reasoning_config=ReasoningConfig(
        reasoning_start_str="<think>",
        reasoning_end_str="</think>",
    )
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    thinking_token_budget=MAX_TOKENS-2048,
)

print("Model loaded.")
NUM_INPUTS=100
START_INDEX=0
prompts = [build_prompt(d["question"], d.get("options")) for d in data[START_INDEX:START_INDEX+NUM_INPUTS]]

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

responses = []

#Post processing script
for out in outputs:
    r = out.outputs[0].text.strip()
    if "infty" in r.lower():
        print("Infty seen")
        r = re.sub("infty", "infinity", r, flags=re.IGNORECASE)
        
    
    responses.append('\n'.join(r.split('\n')[:-4]).replace("oxed{\"\"","")+'\n'.join(r.split('\n')[-4:]))
    
    # ext = judger.extract_ans(r)

    # parsed = []

    # for i in ext.split(','):
    #     i = i.strip()

    #     try:
    #         parsed.append(str(parse_latex(i)))
    #     except Exception as e:
    #         print(f"Failed: {i}")
    #         print(e)
    #         parsed.append(i)

    # parsed_str = ','.join(parsed)

    # print(parsed_str)

    # responses.append(r.replace(ext, parsed_str))
    

all_thinking = [
    [getattr(candidate, "reasoning_content", None) or "" for candidate in out.outputs]
    for out in outputs
]

results = []
for i, (item, response) in enumerate(tqdm(zip(data[START_INDEX:START_INDEX+NUM_INPUTS], responses), total=NUM_INPUTS, desc="Scoring")):
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]

    extracted = judger.extract_ans(response)
    if is_mcq:
        correct = score_mcq(response, str(gold))
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
        "id":        item.get("id"),
        "is_mcq":    is_mcq,
        "gold":      gold,
        "extracted": extracted,
        "response":  response,
        "correct":   correct,
        "thinking":  all_thinking[i],
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
        record = {"id": r["id"], "gold": r["gold"], "extracted": r["extracted"], "correct": r["correct"]}
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
        for j, thinking in enumerate(r["thinking"]):
            prefix = f"\nCANDIDATE {j}" if len(r["thinking"]) > 1 else ""
            f.write(f"{prefix}\nTHINKING:\n{thinking}\n")
        f.write(f"\nMODEL OUTPUT:\n{r['response']}\n")
print(f"Saved text output to {text_path}")
