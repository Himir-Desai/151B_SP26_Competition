import json
import os
from judger import Judger

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "qwen-math-sft2/merged"
# GPU_ID      = "1"
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH = "results_mini/starter_results.jsonl"
OUTPUT_TEXT_PATH = "results_mini/test_thinking.txt"

# os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig
from tqdm import tqdm

data = [json.loads(line) for line in open(DATA_PATH)]

SYSTEM_PROMPT_MATH = (
    "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult."
    "As you are an AI Model you should not fall into the common problems faced by AI models such as being overconfident in wrong answers or being misled by irrelevant information. "
    "You should answer the questions like a professional mathematician who is careful and meticulous in solving problems. "
    "Your task is to solve the problems step by step, as question difficulty varies you should first decide if a question is easy or hard and think accordingly. For easy problems, you can directly solve them and provide the final answer. For hard problems, you should break down the problem into smaller steps and solve each step carefully before arriving at the final answer."
    "Do not round your final answer. Leave complicated answers unevaluated, e.g. leave 325*(1+325) as is instead of evaluating to 105950 or leave (1/2)^[(1999-1963)/31] as is . Prefer answering in decimal form and round only if necessary. For example these are some expected rounded answers [\"62.7777777777778\", \"335.927777777778\", \"604.67\"]. For questions that demand fractional answers like 5/8 output \"5/8\" as it is and not \\frac{5}{8} or 0.625."
    "The most important thing is to put your final answer inside \\boxed{} without failure. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, e.g. \\boxed{3, 7}."
    "Never add the characters { } in the \\boxed{} answer"
)

SYSTEM_PROMPT_MCQ = (
    "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult."
    "As you are an AI Model you should not fall into the common problems faced by AI models such as being overconfident in wrong answers or being misled by irrelevant information."
    "You should answer the questions like a professional mathematician who is careful and meticulous in solving problems."
    "Read the problem and the answer choices below, then select the single best answer from the options only."
    "If your answer does not match any of the options, you should choose the option that is closest to your answer."
    "The most important thing is to put your final answer inside \\boxed{} without failure."
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

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
        enable_thinking=True,        # keeps <think> behaviour active
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

THINK_BUDGET = 3000   # caps thinking tokens; leaves ~5000 for the answer
MAX_TOKENS   = 8192
MAX_MODEL_LEN = 16384  # must be >> MAX_TOKENS + max prompt length

llm = LLM(
    model=MODEL_ID,
    enforce_eager=True,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.9,
    max_model_len=MAX_MODEL_LEN,
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
    thinking_token_budget=THINK_BUDGET,
)

# Forcing params for the retry pass: greedy, short, no thinking.
# We'll append </think> ourselves so the model goes straight to the answer.
forcing_params = SamplingParams(
    max_tokens=128,
    temperature=0.0,
    thinking_token_budget=0,   # skip thinking on retry
)

def build_forcing_prompt(original_prompt: str, thinking: str) -> str:
    """
    Reconstruct a prompt that already contains the (truncated) think block
    so the model only needs to emit the boxed answer.
    """
    truncated = thinking[:2000] if len(thinking) > 2000 else thinking
    return original_prompt + f"<think>\n{truncated}\n</think>\n\nThe answer is \\boxed{{"


print("Model loaded.")

n = 10
items = data[:n]
prompts = [build_prompt(d["question"], d.get("options")) for d in items]

outputs = llm.generate(prompts, sampling_params)

# ── Retry pass for any output missing \boxed{} ────────────────────────────────
needs_retry = []
for i, out in enumerate(outputs):
    if "\\boxed{" not in out.outputs[0].text:
        thinking = getattr(out.outputs[0], "reasoning_content", None) or ""
        needs_retry.append((i, build_forcing_prompt(prompts[i], thinking)))

if needs_retry:
    retry_indices, retry_prompts = zip(*needs_retry)
    retry_outputs = llm.generate(list(retry_prompts), forcing_params)
    for idx, retry_out in zip(retry_indices, retry_outputs):
        # Patch in the forced answer; preserve original thinking for logging
        orig_thinking = getattr(outputs[idx].outputs[0], "reasoning_content", None) or ""
        forced_text   = "\\boxed{" + retry_out.outputs[0].text
        # Monkey-patch so the write loop below sees the corrected text
        outputs[idx].outputs[0].__dict__["_forced_text"] = forced_text

os.makedirs(os.path.dirname(OUTPUT_TEXT_PATH), exist_ok=True)

with open(OUTPUT_TEXT_PATH, "w") as f:
    for i, (item, out) in enumerate(zip(items, outputs)):
        candidate = out.outputs[0]
        thinking  = getattr(candidate, "reasoning_content", None) or ""
        text      = candidate.__dict__.get("_forced_text") or candidate.text
        was_forced = "_forced_text" in candidate.__dict__
        f.write("=" * 60 + "\n")
        f.write(f"Q{i} ID: {item.get('id')}{' [FORCED]' if was_forced else ''}\n")
        f.write(f"QUESTION:\n{item['question']}\n")
        f.write(f"\nTHINKING:\n{thinking}\n")
        f.write(f"\nMODEL OUTPUT:\n{text}\n")