"""
Step 1 — Generate rejection-sampled training data.

For competition questions: run the base model N times per question,
keep only responses that contain \boxed{} AND are judged correct.

For external datasets: reformat existing (question, solution) pairs.

Output: sft_training/data/train.jsonl + val.jsonl
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

# ── CLI args (parsed early so CUDA_VISIBLE_DEVICES is set before any GPU init) ─
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--shard-id",   type=int, default=-1)
_parser.add_argument("--num-shards", type=int, default=1)
_cli, _ = _parser.parse_known_args()

# ── Load config ───────────────────────────────────────────────────────────────
cfg_path = Path(__file__).parent / "config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

_rs_cfg = cfg.get("rejection_sampling", {})
_gpu_ids = _rs_cfg.get("gpu_ids", cfg["hardware"]["gpu_id"])

if _cli.shard_id >= 0:
    # Worker mode: use the GPU assigned to this shard
    _gpu_list = [g.strip() for g in _gpu_ids.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_list[_cli.shard_id % len(_gpu_list)]
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_ids

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Helpers ───────────────────────────────────────────────────────────────────
SYSTEM_MATH = (
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
    "7. Make sure to use the variables as asked in question, and maintain proper bracketing. While sin theta might seem conventional, if the questions explicitly asks to use t, answer with sin(t)\n\n"
    "8. When a function (atan, arctan, ln, log, sin, cos, etc.) is applied to a specific value, preserve the function form exactly — do NOT evaluate to a decimal. "
    "E.g., keep atan(4.76) not 1.360, keep ln(0.5) not -0.693, keep arctan(3) not 1.249.\n\n"
    "9. When the answer IS a decimal number (i.e. the problem asks you to compute a numerical value), "
    "preserve full precision — give at least 4 decimal places and do not round. "
    "E.g., give 143.2242 not 143.2, give 2.2892 not 2.3, give 18.8105 not 18.81."
)
SYSTEM_MCQ = (
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


def build_user_prompt(question, options):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
        return f"{question}\n\nOptions:\n{opts}"
    return question


def has_boxed(text):
    return bool(re.search(r"\\boxed\{", text))


def clean_thinking_boxes(response: str) -> str:
    """Strip \\boxed{} wrappers from inside <think> so the model learns boxed belongs only at the end."""
    think_start = response.find("<think>")
    think_end = response.find("</think>")
    if think_start == -1 or think_end == -1:
        return response
    prefix = response[:think_start + len("<think>")]
    thinking = response[think_start + len("<think>"):think_end]
    suffix = response[think_end:]  # includes </think> + answer
    # Remove \boxed{} wrappers but keep inner content (handles one level of nesting)
    cleaned = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", thinking)
    return prefix + cleaned + suffix


def _extract_last_boxed(text: str):
    """Return (pre_text, boxed_content) for the last \\boxed{} in text, or None."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[:idx], text[start : i - 1]


def format_external_solution(solution: str):
    """
    Reformat an external dataset solution so the full response contains exactly
    one \\boxed{} at the very end, outside <think>.

    Returns the formatted string, or None if the solution has no \\boxed{}.
    """
    result = _extract_last_boxed(solution)
    if result is None:
        return None
    pre, final_answer = result
    # Strip \\boxed{} wrappers from the thinking section but keep inner content
    thinking = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", pre).strip()
    return f"<think>\n{thinking}\n</think>\n\\boxed{{{final_answer}}}"


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
            {"role": "assistant", "content": clean_thinking_boxes(response)},
        ]
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from judger import Judger

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
        tensor_parallel_size=rs.get("tensor_parallel_size", 1),
        gpu_memory_utilization=rs["gpu_memory_utilization"],
        max_model_len=rs["max_model_len"],
        trust_remote_code=True,
        max_num_seqs=rs["max_num_seqs"],
        max_num_batched_tokens=rs.get("max_num_batched_tokens", rs["max_model_len"]),
    )

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=rs["temperature"],
        top_p=rs["top_p"],
        top_k=rs["top_k"],
        n=rs["num_samples_per_question"],
    )

    num_dp = rs.get("num_data_parallel", 1)
    shard_id   = _cli.shard_id
    num_shards = _cli.num_shards if _cli.shard_id >= 0 else num_dp

    # ── Data parallel: spawn workers, then exit (workers handle their own shard) ──
    if shard_id < 0 and num_dp > 1:
        print(f"=== Launching {num_dp} data-parallel workers ===")
        procs = []
        for i in range(num_dp):
            p = subprocess.Popen(
                [sys.executable, __file__,
                 "--shard-id", str(i), "--num-shards", str(num_dp)],
            )
            procs.append(p)
        for p in procs:
            p.wait()

        # Merge shard results
        competition_records = []
        for i in range(num_dp):
            shard_path = Path(f"sft_training/data/competition_rft_shard_{i}.jsonl")
            if shard_path.exists():
                competition_records.extend(json.loads(l) for l in open(shard_path))
                shard_path.unlink()
        print(f"\nMerged {len(competition_records)} competition records from {num_dp} shards")

        # Skip LLM init — workers did all the generation
        # Jump straight to external datasets + split (below)
    else:
        # ── Competition data — rejection sampling ─────────────────────────────
        all_competition = [json.loads(l) for l in open(cfg["data"]["competition_path"])]

        # Skip eval questions (first N are reserved for evaluation)
        skip_n = cfg["data"].get("competition_skip", 0)
        if skip_n > 0:
            all_competition = all_competition[skip_n:]
            print(f"Skipping first {skip_n} questions (eval set). Using {len(all_competition)} remaining.")

        # Slice this shard's portion
        if num_shards > 1:
            chunk = len(all_competition) // num_shards
            start = shard_id * chunk
            end   = start + chunk if shard_id < num_shards - 1 else len(all_competition)
            competition_data = all_competition[start:end]
            print(f"=== Shard {shard_id}/{num_shards}: questions {start}–{end-1} "
                  f"({len(competition_data)} total) ===")
        else:
            competition_data = all_competition
            print(f"=== Rejection sampling on {len(competition_data)} competition questions ===")

        judger = Judger(strict_extract=False)

        max_prompt_tokens = rs["max_model_len"] - max_new_tokens
        prompts, prompt_items, skipped_long = [], [], 0
        for item in competition_data:
            options = item.get("options")
            system  = SYSTEM_MCQ if options else SYSTEM_MATH
            user    = build_user_prompt(item["question"], options)
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"thinking_budget": thinking_budget},
            )
            if len(tokenizer.encode(prompt_text)) > max_prompt_tokens:
                skipped_long += 1
                continue
            prompts.append(prompt_text)
            prompt_items.append(item)

        if skipped_long:
            print(f"Skipped {skipped_long} questions exceeding {max_prompt_tokens} tokens")

        print(f"Generating {rs['num_samples_per_question']} samples per question "
              f"(thinking_budget={thinking_budget}, max_new_tokens={max_new_tokens})...")
        outputs = llm.generate(prompts, sampling_params)

        competition_records = []
        truncated = correct_qs = 0
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
                    break
            if kept:
                correct_qs += 1

        print(f"Competition: {correct_qs}/{len(competition_data)} questions had ≥1 correct sample")
        print(f"Truncated (no \\boxed{{}}): {truncated} responses discarded")
        print(f"Competition records: {len(competition_records)}")

        # Shard workers save their slice and exit — merging done by main process
        if num_shards > 1:
            shard_path = Path(f"sft_training/data/competition_rft_shard_{shard_id}.jsonl")
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            with open(shard_path, "w") as f:
                for r in competition_records:
                    f.write(json.dumps(r) + "\n")
            print(f"Shard {shard_id} saved → {shard_path}")
            sys.exit(0)

        rft_path = Path("sft_training/data/competition_rft.jsonl")
        rft_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rft_path, "w") as f:
            for r in competition_records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved competition data → {rft_path}")

    # ── External datasets ─────────────────────────────────────────────────────
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
                        formatted = format_external_solution(sol) if q and sol else None
                        if formatted:
                            external_records.append({
                                "messages": [
                                    {"role": "system", "content": SYSTEM_MATH},
                                    {"role": "user", "content": q},
                                    {"role": "assistant", "content": formatted},
                                ]
                            })

                elif ds_name == "hendrycks/competition_math":
                    ds = load_dataset(ds_name, split="train", trust_remote_code=True)
                    for row in ds:
                        if len(external_records) >= max_ext:
                            break
                        q = row.get("problem", "")
                        sol = row.get("solution", "")
                        formatted = format_external_solution(sol) if q and sol else None
                        if formatted:
                            external_records.append({
                                "messages": [
                                    {"role": "system", "content": SYSTEM_MATH},
                                    {"role": "user", "content": q},
                                    {"role": "assistant", "content": formatted},
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

    # ── Merge + split ─────────────────────────────────────────────────────────
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
