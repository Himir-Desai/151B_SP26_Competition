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

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))
from judger import Judger

eval_cfg = cfg["evaluation"]
model_cfg = cfg["model"]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
SYSTEM_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
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


# ── Load data ─────────────────────────────────────────────────────────────────
data = [json.loads(l) for l in open(eval_cfg["data_path"])]
n = eval_cfg.get("num_questions")
if n:
    data = data[:n]
print(f"Evaluating on {len(data)} questions")

# ── Load model with LoRA adapter via vLLM ────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    model_cfg["adapter_dir"], trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

thinking_budget = eval_cfg["thinking_budget"]

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

from vllm.lora.request import LoRARequest
lora_request = LoRARequest("sft_adapter", 1, model_cfg["adapter_dir"])

sampling_params = SamplingParams(
    max_tokens=eval_cfg["max_new_tokens"],
    temperature=eval_cfg["temperature"],
    top_p=eval_cfg["top_p"],
    top_k=eval_cfg["top_k"],
)

# ── Build prompts ─────────────────────────────────────────────────────────────
prompts = []
for item in data:
    system, user = build_prompt(item["question"], item.get("options"))
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"thinking_budget": thinking_budget},
    )
    prompts.append(prompt_text)

print(f"Generating responses...")
outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
responses = [out.outputs[0].text.strip() for out in outputs]

# ── Score ─────────────────────────────────────────────────────────────────────
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

# ── Summary ───────────────────────────────────────────────────────────────────
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

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = Path(eval_cfg["output_path"])
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"Saved → {out_path}")
