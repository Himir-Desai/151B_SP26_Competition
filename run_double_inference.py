import json
import re
import sys
from pathlib import Path

from judger import Judger
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig
from vllm.lora.request import LoRARequest
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_MODEL_ID         = "Qwen/Qwen3-4B-Thinking-2507"
SOLVER_LORA_ID        = "SkyAsl/Qwen3-olympiad-math-thinking-2507"
DATA_PATH             = "data/public.jsonl"
OUTPUT_PATH           = "results/double_inf_results.jsonl"
THINKING_PATH         = "results/double_inf_thinking.jsonl"
OUTPUT_TEXT_PATH      = "results/double_inf_output.txt"
SOLVER_MAX_TOKENS     = 32768
EXTRACTOR_MAX_TOKENS  = 8192
NUM_INPUTS            = 100
START_INDEX           = 0

# ── System prompts ─────────────────────────────────────────────────────────────

SOLVER_SYSTEM_PROMPT = (
    "You are an expert olympiad mathematician. Solve the given problem carefully and rigorously.\n\n"
    "CRITICAL FORMATTING RULES:\n"
    "1. NEVER give answers as decimals or rounded numbers. Always use exact form.\n"
    "2. Leave trigonometric and inverse trig functions unevaluated: e.g. tan(3), arcsin(1/2), cos(pi/7).\n"
    "3. Leave logarithms unevaluated: e.g. log(4.2), ln(5).\n"
    "4. Leave exponents unevaluated: e.g. (1/2)^(41/23), 2^(1/3).\n"
    "5. Leave pi, e, sqrt(...) and all irrational constants in symbolic form.\n"
    "6. Convert any decimal numbers in the problem to exact fractions before using them.\n"
    "7. If the problem has multiple parts or multiple answer blanks, answer every single one.\n"
    "8. Put your FINAL answer inside \\boxed{}. If there are multiple answers, separate them by commas inside a single \\boxed{}, e.g. \\boxed{3, tan(3), (1/2)^(41/23)}.\n"
    "9. Output exactly ONE \\boxed{} at the very end containing all final answers.\n"
    "10. Do not output any other \\boxed{} except the final answer."
)

EXTRACTOR_SYSTEM_PROMPT = (
    "You are a precise answer extractor. You will be given a math problem and a solution written by another model.\n\n"
    "Your task:\n"
    "1. Read the problem carefully and count how many answer slots there are. "
    "Answer slots include: [ANS], blanks like ___, fill-in-the-blank spaces, and question marks that request a value.\n"
    "2. Read the solution and identify the final answer for EACH answer slot.\n"
    "3. Output all answers inside a single \\boxed{}, separated by commas.\n"
    "4. Keep all answers in exact form — no decimals, no rounding.\n"
    "5. Your response must end with exactly one \\boxed{} containing all answers.\n\n"
    "EXAMPLE: If the problem has 3 answer slots and the answers are 5, tan(3), and (1/2)^(41/23), "
    "output: \\boxed{5, tan(3), (1/2)^(41/23)}"
)


def build_solver_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def build_extractor_prompt(tokenizer, question: str, solver_response: str) -> str:
    user_content = (
        f"PROBLEM:\n{question}\n\n"
        f"SOLUTION FROM SOLVER:\n{solver_response}\n\n"
        "Extract the final answer for every answer slot in the problem (every [ANS], blank ___, or place that asks for a value). "
        "Output all answers in a single \\boxed{} separated by commas."
    )
    messages = [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


# ── Load data ──────────────────────────────────────────────────────────────────
data = [json.loads(line) for line in open(DATA_PATH)]
data = [d for d in data if not d.get("options")]
print(f"Loaded {len(data)} free-form questions")
batch = data[START_INDEX:START_INDEX + NUM_INPUTS]

# ── Load tokenizer and model (once) ───────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

max_model_len = max(SOLVER_MAX_TOKENS, EXTRACTOR_MAX_TOKENS) + 2048

print(f"Loading base model: {BASE_MODEL_ID} (with LoRA support)")
llm = LLM(
    model=BASE_MODEL_ID,
    enforce_eager=True,
    dtype="bfloat16",
    enable_prefix_caching=True,
    enable_lora=True,
    max_lora_rank=64,
    gpu_memory_utilization=0.93,
    max_model_len=max_model_len,
    trust_remote_code=True,
    max_num_seqs=32,
    max_num_batched_tokens=max_model_len,
    reasoning_config=ReasoningConfig(
        reasoning_start_str="<think>",
        reasoning_end_str="</think>",
    ),
)
print("Model loaded.")

solver_lora = LoRARequest("solver", 1, SOLVER_LORA_ID)

# ── Phase 1: Solve with LoRA adapter ──────────────────────────────────────────
solver_params = SamplingParams(
    max_tokens=SOLVER_MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0,
    thinking_token_budget=SOLVER_MAX_TOKENS - 2048,
)

solver_prompts = [build_solver_prompt(tokenizer, d["question"]) for d in batch]
print(f"Phase 1: solving {len(solver_prompts)} questions with LoRA adapter ({SOLVER_LORA_ID})...")
solver_outputs = llm.generate(solver_prompts, solver_params, lora_request=solver_lora)

solver_responses = []
solver_thinking = []
for out in solver_outputs:
    text = out.outputs[0].text.strip()
    thinking = getattr(out.outputs[0], "reasoning_content", None) or ""
    solver_responses.append(text)
    solver_thinking.append(thinking)

print("Phase 1 complete.")

# ── Phase 2: Extract answers with base model ───────────────────────────────────
extractor_params = SamplingParams(
    max_tokens=EXTRACTOR_MAX_TOKENS,
    temperature=0.3,
    top_p=0.9,
    thinking_token_budget=EXTRACTOR_MAX_TOKENS - 1024,
)

extractor_prompts = [
    build_extractor_prompt(tokenizer, d["question"], resp)
    for d, resp in zip(batch, solver_responses)
]
print(f"Phase 2: extracting answers for {len(extractor_prompts)} questions with base model...")
extractor_outputs = llm.generate(extractor_prompts, extractor_params)

extractor_responses = []
extractor_thinking = []
for out in extractor_outputs:
    text = out.outputs[0].text.strip()
    thinking = getattr(out.outputs[0], "reasoning_content", None) or ""
    extractor_responses.append(text)
    extractor_thinking.append(thinking)

print("Phase 2 complete.")

# ── Scoring ────────────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
judger = Judger(strict_extract=False)

results = []
for i, (item, solver_resp, extractor_resp) in enumerate(
    tqdm(zip(batch, solver_responses, extractor_responses), total=len(batch), desc="Scoring")
):
    gold = item["answer"]
    gold_list = gold if isinstance(gold, list) else [gold]
    extracted = judger.extract_ans(extractor_resp)
    try:
        correct = judger.auto_judge(
            pred=extractor_resp,
            gold=gold_list,
            options=[[]] * len(gold_list),
        )
    except Exception:
        correct = False

    results.append({
        "id":                  item.get("id"),
        "gold":                gold,
        "extracted":           extracted,
        "solver_response":     solver_resp,
        "extractor_response":  extractor_resp,
        "correct":             correct,
        "solver_thinking":     solver_thinking[i],
        "extractor_thinking":  extractor_thinking[i],
        "question":            item["question"],
    })

correct_count = sum(r["correct"] for r in results)
print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"  Free-form: {correct_count:4d} / {len(results):4d}  ({correct_count / len(results) * 100:.2f}%)")
print("=" * 50)

# ── Save outputs ───────────────────────────────────────────────────────────────
out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    for r in results:
        f.write(json.dumps({
            "id": r["id"],
            "gold": r["gold"],
            "extracted": r["extracted"],
            "correct": r["correct"],
        }) + "\n")
print(f"Saved {len(results)} records to {out_path}")

think_path = Path(THINKING_PATH)
think_path.parent.mkdir(parents=True, exist_ok=True)
with open(think_path, "w") as f:
    for r in results:
        f.write(json.dumps({
            "id": r["id"],
            "solver_thinking": r["solver_thinking"],
            "extractor_thinking": r["extractor_thinking"],
        }) + "\n")
print(f"Saved thinking to {think_path}")

text_path = Path(OUTPUT_TEXT_PATH)
text_path.parent.mkdir(parents=True, exist_ok=True)
with open(text_path, "w") as f:
    for i, r in enumerate(results):
        f.write("=" * 60 + "\n")
        f.write(f"Q{i} ID: {r['id']}\n")
        f.write(f"QUESTION:\n{r['question']}\n\n")
        f.write(f"--- SOLVER THINKING ---\n{r['solver_thinking']}\n\n")
        f.write(f"--- SOLVER RESPONSE ---\n{r['solver_response']}\n\n")
        f.write(f"--- EXTRACTOR THINKING ---\n{r['extractor_thinking']}\n\n")
        f.write(f"--- EXTRACTOR RESPONSE ---\n{r['extractor_response']}\n\n")
        f.write(f"GOLD: {r['gold']}\nEXTRACTED: {r['extracted']}\nCORRECT: {r['correct']}\n\n")
print(f"Saved text output to {text_path}")
