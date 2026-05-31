"""
run_inference.py
================
CSE 151B SP26 Math Reasoning Competition — single entry-point inference script.

Usage (Python):
    from run_inference import run_inference
    run_inference(data_path="data/private.jsonl", output_path="results/submission.csv")

Usage (CLI):
    python run_inference.py --data_path data/private.jsonl --output_path results/submission.csv
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

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

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID         = "SkyAsl/Qwen3-olympiad-math-thinking-2507"
GPU_ID           = "3"                  # CUDA_VISIBLE_DEVICES
MAX_TOKENS       = 4096
THINKING_PATH    = "results/thinking.jsonl"
OUTPUT_TEXT_PATH = "results/thinking.txt"

# LoRA adapter paths (relative to repo root)
LORA_GRPO_PATH    = "./lora_adapters/lora_grpo/lora_grpo/lora_grpo_v2"
LORA_SFT_PATH     = "./lora_adapters/lora_adapter_openr1_s1k/lora_adapter_openr1_s1k/lora_adapter_openr1_s1k"

# System prompts
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Before you start to calculate, write down your reasoning and the steps you will take to solve the problem. "
    "Do not change your reasoning after you start calculating, unless there is a serious error. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def extract_letter(text: str) -> str:
    """Extract the predicted answer letter from a \\boxed{} expression."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def extract_boxed_answer(text: str) -> str:
    """Extract the final answer from a \\boxed{} expression (free-form)."""
    matches = []
    for m in re.finditer(r"\\boxed\{", text):
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            matches.append(text[start : i - 1].strip())
    return matches[-1] if matches else text.strip()


def normalize_answer(response: str, is_mcq: bool) -> str:
    """Post-process a raw model response into a clean answer string."""
    if is_mcq:
        return extract_letter(response)
    return extract_boxed_answer(response)


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


def acc(subset: list) -> float:
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_inference(
    data_path: str = "data/private.jsonl",
    output_path: str = "results/submission.csv",
) -> None:
    """
    Full end-to-end inference pipeline.

    1. Loads the INT8-quantized Qwen3-4B-Thinking base model via vLLM with LoRA enabled.
    2. First pass  — GRPO adapter generates initial reasoning + answer.
    3. Second pass — SFT adapter refines using the first-pass output as additional context.
    4. Extracts final answers and writes submission CSV.

    Parameters
    ----------
    data_path   : Path to the private test JSONL (no ground-truth answers).
    output_path : Path to write the submission CSV (columns: id, answer).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    os.environ["PYTORCH_ALLOC_CONF"]   = "expandable_segments:True"

    # ── Lazy imports (heavy; only load when actually running) ──────────────────
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from tqdm import tqdm

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset from {data_path} ...")
    data = [json.loads(line) for line in open(data_path)]
    data = data[:100]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"  {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"Loading tokenizer for {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model with vLLM (INT8, LoRA enabled) ...")
    llm = LLM(
        model=MODEL_ID,
        enforce_eager=True,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.9,
        max_model_len=6240,
        trust_remote_code=True,
        max_num_seqs=96,
        max_num_batched_tokens=16382,
        enable_lora=True,
        max_lora_rank=64,
    )
    print("Model loaded.")

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    # ── Build base prompts ────────────────────────────────────────────────────
    #######################################################################################
    ## When uploading weights to huggingface, need to change these to download correctly ##
    #######################################################################################
    QLORA_ADAPTER = LoRARequest("qlora_sft", 1, LORA_SFT_PATH)
    GRPO_ADAPTER = LoRARequest( "grpo_rl", 2, LORA_GRPO_PATH)
    #######################################################################################
    prompts = []
    requests = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
        if bool(item.get("options")):
            requests.append(QLORA_ADAPTER)
        else:
            requests.append(GRPO_ADAPTER)

    # ── Forward Pass ─────────────────────────────────────────────────────────
    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(
        prompts, 
        sampling_params=sampling_params,
        lora_request=requests,
    )

    # ── Post-process & extract answers ───────────────────────────────────────
    print("Post-processing answers ...")
    sys.path.insert(0, ".")
    judger = Judger(strict_extract=False)

    responses = []
    for out in outputs:
        r = out.outputs[0].text.strip()
        if "infty" in r.lower():
            r = re.sub("infty", "infinity", r, flags=re.IGNORECASE)
        r = re.sub(r'(\d+)\s+([a-zA-Z])', r'\1*\2', r)
        responses.append(
            '\n'.join(r.split('\n')[:-4]).replace('oxed{""', '') + '\n'.join(r.split('\n')[-4:])
        )

    all_thinking = [
        [getattr(candidate, "reasoning_content", None) or "" for candidate in out.outputs]
        for out in outputs
    ]

    has_labels = all("answer" in item for item in data)

    results = []
    for i, (item, response) in enumerate(tqdm(zip(data, responses), total=len(data), desc="Scoring")):
        is_mcq    = bool(item.get("options"))
        extracted = judger.extract_ans(response) if not is_mcq else extract_letter(response)

        correct = None
        if has_labels:
            gold = item["answer"]
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
            "id":        item["id"],
            "is_mcq":    is_mcq,
            "gold":      item.get("answer"),
            "answer":    extracted,
            "extracted": extracted,
            "response":  response,
            "correct":   correct,
            "thinking":  all_thinking[i],
        })

    # ── Print accuracy (when ground truth is available) ───────────────────────
    if has_labels:
        mcq_res  = [r for r in results if r["is_mcq"]]
        free_res = [r for r in results if not r["is_mcq"]]
        print("=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
        print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
        print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
        print("=" * 50)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("id,answer\n")
        for r in results:
            f.write(f"{r['id']},{r['answer']}\n")

    # ── Write detailed JSONL ──────────────────────────────────────────────────
    think_path = Path(THINKING_PATH)
    think_path.parent.mkdir(parents=True, exist_ok=True)
    with open(think_path, "w") as f:
        for r in results:
            record = {"id": r["id"], "gold": r["gold"], "extracted": r["extracted"], "correct": r["correct"]}
            f.write(json.dumps(record) + "\n")
    print(f"Saved detailed results to {think_path}")

    # ── Write human-readable text ─────────────────────────────────────────────
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

    print(f"\nDone. {len(results)} answers written to {out_path}")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSE 151B Competition — run_inference")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/public.jsonl",
        help="Path to the private test JSONL file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/submission.csv",
        help="Path to write the output submission CSV",
    )
    args = parser.parse_args()
    run_inference(data_path=args.data_path, output_path=args.output_path)
