"""
Step 1 — Generate rejection-sampled training data.

For competition questions: run the base model N times per question,
keep only responses that contain \boxed{} AND are judged correct.

For external datasets: reformat existing (question, solution) pairs.

Output: sft_training/data/train.jsonl + val.jsonl
"""

import json
import os
import random
import re
import sys
from pathlib import Path

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

# ── Helpers ───────────────────────────────────────────────────────────────────
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


def build_user_prompt(question, options):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
        return f"{question}\n\nOptions:\n{opts}"
    return question


def has_boxed(text):
    return bool(re.search(r"\\boxed\{", text))


def extract_letter(text):
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_response(response, item, judger):
    is_mcq = bool(item.get("options"))
    gold = item["answer"]
    if is_mcq:
        return extract_letter(response) == str(gold).strip().upper()
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(
            pred=response,
            gold=gold_list,
            options=[[]] * len(gold_list),
        )
    except Exception:
        return False


def make_record(item, response):
    options = item.get("options")
    system = SYSTEM_MCQ if options else SYSTEM_MATH
    user = build_user_prompt(item["question"], options)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": response},
        ]
    }


# ── Load model ────────────────────────────────────────────────────────────────
rs = cfg["rejection_sampling"]
thinking_budget = rs["thinking_budget"]
max_new_tokens = thinking_budget + 512  # always room for boxed answer

model_id = cfg["model"]["model_id"]
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=model_id,
    dtype="bfloat16",
    enable_prefix_caching=True,
    gpu_memory_utilization=rs["gpu_memory_utilization"],
    max_model_len=rs["max_model_len"],
    trust_remote_code=True,
    max_num_seqs=rs["max_num_seqs"],
    max_num_batched_tokens=rs["max_model_len"],
)

sampling_params = SamplingParams(
    max_tokens=max_new_tokens,
    temperature=rs["temperature"],
    top_p=rs["top_p"],
    top_k=rs["top_k"],
    n=rs["num_samples_per_question"],
)

judger = Judger(strict_extract=False)

# ── Competition data — rejection sampling ─────────────────────────────────────
print("=== Rejection sampling on competition data ===")
competition_data = [json.loads(l) for l in open(cfg["data"]["competition_path"])]
print(f"Loaded {len(competition_data)} competition questions")

# Build one prompt per question; skip any whose prompt alone exceeds max_model_len
max_prompt_tokens = rs["max_model_len"] - max_new_tokens
prompts = []
prompt_items = []  # parallel list tracking which items have valid prompts
skipped_long = 0
for item in competition_data:
    options = item.get("options")
    system = SYSTEM_MCQ if options else SYSTEM_MATH
    user = build_user_prompt(item["question"], options)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"thinking_budget": thinking_budget},
    )
    token_len = len(tokenizer.encode(prompt_text))
    if token_len > max_prompt_tokens:
        skipped_long += 1
        continue
    prompts.append(prompt_text)
    prompt_items.append(item)

if skipped_long:
    print(f"Skipped {skipped_long} questions whose prompts exceed {max_prompt_tokens} tokens")

print(f"Generating {rs['num_samples_per_question']} samples per question "
      f"(thinking_budget={thinking_budget}, max_new_tokens={max_new_tokens})...")
outputs = llm.generate(prompts, sampling_params)

competition_records = []
truncated = 0
correct_qs = 0
for item, out in tqdm(zip(prompt_items, outputs), total=len(prompt_items), desc="Filtering"):
    kept = False
    for sample in out.outputs:
        response = sample.text.strip()
        if not has_boxed(response):
            truncated += 1
            continue
        if score_response(response, item, judger):
            competition_records.append(make_record(item, response))
            kept = True
            break  # one correct sample per question is enough
    if kept:
        correct_qs += 1

print(f"Competition: {correct_qs}/{len(competition_data)} questions had ≥1 correct sample")
print(f"Truncated (no \\boxed{{}}): {truncated} responses discarded")
print(f"Competition training records: {len(competition_records)}")

rft_path = Path("sft_training/data/competition_rft.jsonl")
rft_path.parent.mkdir(parents=True, exist_ok=True)
with open(rft_path, "w") as f:
    for r in competition_records:
        f.write(json.dumps(r) + "\n")
print(f"Saved competition data → {rft_path}")

# ── External datasets ─────────────────────────────────────────────────────────
external_records = []
ext_datasets = cfg["data"].get("external_datasets", [])
max_ext = cfg["data"].get("max_external_samples") or 999999

if ext_datasets:
    from datasets import load_dataset

    for ds_name in ext_datasets:
        print(f"\n=== Loading external dataset: {ds_name} ===")
        try:
            if ds_name == "AI-MO/NuminaMath-CoT":
                ds = load_dataset(ds_name, split="train")
                for row in ds:
                    if len(external_records) >= max_ext:
                        break
                    q = row.get("problem", "")
                    sol = row.get("solution", "")
                    if q and sol and "boxed" in sol:
                        external_records.append({
                            "messages": [
                                {"role": "system", "content": SYSTEM_MATH},
                                {"role": "user", "content": q},
                                {"role": "assistant", "content": sol},
                            ]
                        })

            elif ds_name == "hendrycks/competition_math":
                ds = load_dataset(ds_name, split="train", trust_remote_code=True)
                for row in ds:
                    if len(external_records) >= max_ext:
                        break
                    q = row.get("problem", "")
                    sol = row.get("solution", "")
                    if q and sol and "boxed" in sol:
                        external_records.append({
                            "messages": [
                                {"role": "system", "content": SYSTEM_MATH},
                                {"role": "user", "content": q},
                                {"role": "assistant", "content": sol},
                            ]
                        })

            print(f"  → {len(external_records)} external records so far")
        except Exception as e:
            print(f"  Warning: failed to load {ds_name}: {e}")

    ext_path = Path("sft_training/data/external.jsonl")
    with open(ext_path, "w") as f:
        for r in external_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved external data → {ext_path} ({len(external_records)} records)")

# ── Merge + split ──────────────────────────────────────────────────────────────
all_records = competition_records + external_records
random.shuffle(all_records)

val_n = max(1, int(len(all_records) * cfg["data"]["val_split"]))
val_records = all_records[:val_n]
train_records = all_records[val_n:]

train_path = Path(cfg["data"]["train_output"])
val_path = Path(cfg["data"]["val_output"])
train_path.parent.mkdir(parents=True, exist_ok=True)

with open(train_path, "w") as f:
    for r in train_records:
        f.write(json.dumps(r) + "\n")
with open(val_path, "w") as f:
    for r in val_records:
        f.write(json.dumps(r) + "\n")

print(f"\n{'='*50}")
print(f"Train: {len(train_records)} records → {train_path}")
print(f"Val:   {len(val_records)} records  → {val_path}")
print(f"{'='*50}")
