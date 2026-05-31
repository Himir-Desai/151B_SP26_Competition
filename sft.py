# train_qwen_math_sft.py

import re
import torch
from decimal import Decimal, InvalidOperation
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
DATASET_NAME = "open-r1/OpenR1-Math-220k"
OUT_DIR = "qwen3-4b-thinking-math-sft"

DEBUG = False
DEBUG_N = 2000

SYSTEM_PROMPT_MATH = """You are a careful mathematical reasoning model.

Solve the problem with clear reasoning.

Final answer rules:
- Show the reasoning needed to solve the problem.
- End with exactly one final answer in \\boxed{}.
- Never use \\boxed{} except on the final line.
- If the answer is exact, keep it in raw form, such as root(3), log(5), pi/7, 5/8, arctan(2), (1/2)^(31/7).
- If the answer is decimal, still prefer exact form but if you really need to, give at least 8 digits after the decimal point.
"""


class LiveUpdateCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("Training started", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"[step {state.global_step}] {logs}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        print(f"Checkpoint saved at step {state.global_step}", flush=True)


def extract_boxed(text):
    """
    Extract the final \\boxed{...} content.
    This handles simple nested braces like \\boxed{\\frac{1}{2}} better than a flat regex.
    """
    text = str(text)
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    out = []

    while i < len(text):
        ch = text[i]

        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
        else:
            out.append(ch)

        i += 1

    return None


def remove_all_boxed(text):
    """
    Remove every \\boxed{...} block while preserving the rest of the reasoning.
    Handles nested braces inside the boxed content.
    """
    text = str(text)
    marker = r"\boxed{"
    result = []
    i = 0

    while i < len(text):
        if text.startswith(marker, i):
            i += len(marker)
            depth = 1

            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1

            continue

        result.append(text[i])
        i += 1

    return "".join(result)


def clean_answer(ans):
    """
    Convert common LaTeX final-answer forms into the raw style you want:
    \\sqrt{3} -> root(3)
    \\ln{5} or \\ln(5) -> log(5)
    \\frac{5}{8} -> 5/8
    \\pi -> pi

    Also forces decimal answers to at least 8 digits after the decimal.
    """
    if ans is None:
        return None

    ans = str(ans).strip()

    # Remove common final-answer wording if the answer column accidentally has it.
    ans = re.sub(r"^\s*(final\s+answer\s*:|answer\s*:)\s*", "", ans, flags=re.IGNORECASE)

    # Convert common LaTeX wrappers.
    ans = ans.replace(r"\left", "").replace(r"\right", "")
    ans = ans.replace(r"\,", "").replace(r"\!", "")
    ans = ans.replace(r"\pi", "pi")

    # \frac{a}{b} -> a/b
    ans = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", ans)

    # \sqrt{a} -> root(a)
    ans = re.sub(r"\\sqrt\{([^{}]+)\}", r"root(\1)", ans)

    # \ln{a}, \ln(a), ln(a) -> log(a)
    ans = re.sub(r"\\ln\{([^{}]+)\}", r"log(\1)", ans)
    ans = re.sub(r"\\ln\s*\(([^()]+)\)", r"log(\1)", ans)
    ans = re.sub(r"\bln\s*\(([^()]+)\)", r"log(\1)", ans)

    # \log{a} -> log(a)
    ans = re.sub(r"\\log\{([^{}]+)\}", r"log(\1)", ans)
    ans = ans.replace(r"\log", "log")

    # Strip remaining math delimiters.
    ans = ans.replace("$", "").strip()

    # If it is a plain decimal, force at least 8 digits after decimal.
    try:
        d = Decimal(ans)
        if "." in ans:
            places = abs(d.as_tuple().exponent)
            if places < 8:
                ans = f"{d:.8f}"
    except InvalidOperation:
        pass

    # Avoid extra whitespace.
    ans = re.sub(r"\s+", "", ans)

    return ans


def pick_generation(example):
    """
    OpenR1-Math-220k has a `generations` column containing reasoning traces.
    We train on that column first, because it preserves thinking.
    """
    gens = example.get("generations")

    if isinstance(gens, list) and len(gens) > 0:
        return str(gens[0]).strip()

    if isinstance(gens, str) and gens.strip():
        return gens.strip()

    # Fallbacks for dataset revisions or alternate math datasets.
    solution = (
        example.get("solution")
        or example.get("generation")
        or example.get("messages")
        or example.get("answer")
        or ""
    )

    if isinstance(solution, list):
        assistant_msgs = [
            m["content"] for m in solution
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        return assistant_msgs[-1].strip() if assistant_msgs else str(solution).strip()

    return str(solution).strip()


def pick_answer(example, solution):
    """
    Prefer the dataset's answer column for the final canonical box.
    Fallback to other answer columns, then to the last boxed answer in the generation.
    """
    answer = (
        example.get("answer")
        or example.get("final_answer")
        or example.get("boxed_answer")
        or extract_boxed(solution)
    )
    return clean_answer(answer)


def normalize_solution(example):
    problem = (
        example.get("problem")
        or example.get("question")
        or example.get("prompt")
        or ""
    )

    solution = pick_generation(example)
    answer = pick_answer(example, solution)

    # Preserve the reasoning trace, including <think>...</think>, but remove old boxes.
    solution = remove_all_boxed(solution).strip()

    # Append exactly one final canonical boxed answer from the answer column.
    if answer:
        solution = solution + f"\n\nFinal answer:\n\\boxed{{{answer}}}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_MATH},
            {"role": "user", "content": str(problem)},
            {"role": "assistant", "content": str(solution)},
        ]
    }


def main():
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print("Loading dataset...", flush=True)
    ds = load_dataset(DATASET_NAME)

    split = "train" if "train" in ds else list(ds.keys())[0]
    train_ds = ds[split]

    print(f"Using split: {split}", flush=True)
    print(f"Raw dataset size: {len(train_ds)}", flush=True)
    print(f"Raw dataset columns: {train_ds.column_names}", flush=True)

    if DEBUG:
        train_ds = train_ds.select(range(min(DEBUG_N, len(train_ds))))
        print(f"DEBUG mode on. Dataset size: {len(train_ds)}", flush=True)

    print("Formatting dataset...", flush=True)
    train_ds = train_ds.map(
        normalize_solution,
        remove_columns=train_ds.column_names,
        num_proc=4,
    )

    print("=" * 80, flush=True)
    print("Sanity check formatted example", flush=True)
    print("=" * 80, flush=True)
    print("Problem:", train_ds[0]["messages"][1]["content"][:1000], flush=True)
    print("-" * 80, flush=True)
    print("Assistant target:", train_ds[0]["messages"][2]["content"][:4000], flush=True)
    print("-" * 80, flush=True)
    print("Assistant target ending:", train_ds[0]["messages"][2]["content"][-500:], flush=True)
    print("=" * 80, flush=True)

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

    print("Setting LoRA config...", flush=True)
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    print("Setting trainer args...", flush=True)
    args = SFTConfig(
        output_dir=OUT_DIR,
        num_train_epochs=1,
        ddp_find_unused_parameters=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1 if DEBUG else 10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        max_length=2048,
        packing=True,
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    print("Initializing trainer...", flush=True)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[LiveUpdateCallback()],
    )

    print("Starting training...", flush=True)
    trainer.train()

    print("Saving model...", flush=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"Done. Saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
