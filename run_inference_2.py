import csv
import json
import os
import re
import signal
import sys
from pathlib import Path

from prompt_builder import build_prompt
from answer_parser import normalize_pred_for_row

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "darthsid12/sft3"
GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "results/submission.csv"
MAX_TOKENS  = 32768

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


def extract_letter(text: str) -> str:
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text
    m = re.search(r"\\boxed\{([A-Za-z])\}", search_text)
    if m:
        return m.group(1).upper()
    all_matches = list(re.finditer(r"\\boxed\{([A-Za-z])\}", text))
    if all_matches:
        return all_matches[-1].group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from tqdm import tqdm

    sys.path.insert(0, ".")
    from judger import Judger

    # ── Load Dataset ──────────────────────────────────────────────────────────
    data = [json.loads(line) for line in open(DATA_PATH)]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # ── Load Model ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        enable_prefix_caching=True,
        max_model_len=32768,
        trust_remote_code=True,
        max_num_seqs=192,
        max_num_batched_tokens=32768,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )
    print("Model loaded.")

    # ── Generate Responses ────────────────────────────────────────────────────
    prompts = [build_prompt(item, tokenizer) for item in data]
    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    responses = [out.outputs[0].text.strip() for out in outputs]

    # ── Normalize and Extract Answers ─────────────────────────────────────────
    JUDGE_TIMEOUT = 15

    def _raise_timeout(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _raise_timeout)

    judger = Judger(strict_extract=False)
    rows = []

    for item, response in tqdm(zip(data, responses), total=len(responses), desc="Extracting"):
        is_mcq = bool(item.get("options"))
        normalized = normalize_pred_for_row(response, item)

        if is_mcq:
            extracted = extract_letter(normalized)
        else:
            signal.alarm(JUDGE_TIMEOUT)
            try:
                extracted = judger.extract_ans(normalized) or ""
            except TimeoutError:
                tqdm.write(f"[SKIP] extract_ans timed out on id={item.get('id')}")
                extracted = ""
            except Exception:
                extracted = ""
            finally:
                signal.alarm(0)

        rows.append({"id": item["id"], "response": normalized})

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_path}")
