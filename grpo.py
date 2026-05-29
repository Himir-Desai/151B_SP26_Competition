from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from judger import Judger
import re
judger = Judger()

SYSTEM_PROMPT_MATH = (
    "You are a Nobel Prize winning mathematician. Solve the problem step-by-step. "
    "Think deeply and extensively. Write a detailed thinking trace inside <think>...</think> "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are a Nobel Prize winning mathematician. "
    "Think deeply and extensively. Write a detailed thinking trace inside <think>...</think> "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

def normalize_answer_with_judger(answer, ans_type=None):
    """Use the Judger's own normalization pipeline instead of rolling your own."""
    if isinstance(answer, list):
        # Normalize each element, then rejoin
        normalized = [judger.norm_ans_str(str(a).strip(), ans_type) for a in answer]
        return normalized
    return judger.norm_ans_str(str(answer).strip(), ans_type)


def wrap_answer_in_boxed(answer, is_mcq=False):
    import re

    ans_type = "MCS" if is_mcq else None

    # ── Normalize via Judger first ─────────────────────────────────────────
    if isinstance(answer, list):
        normalized = normalize_answer_with_judger(answer, ans_type)
        inner = ", ".join(normalized)
        return f"\\boxed{{{inner}}}"

    answer = str(answer).strip()

    if "\\boxed{" in answer:
        return answer  # already wrapped, don't double-box

    if is_mcq:
        # Judger's norm_ans_str handles letter normalization for MCS
        normalized = judger.norm_ans_str(answer, "MCS")
        letter = re.match(r"^([A-Ja-j])[.):\s]?", normalized)
        if letter:
            return f"\\boxed{{{letter.group(1).upper()}}}"

    # ── For numeric/expression answers: normalize then try to evaluate ─────
    normalized = judger.norm_ans_str(answer, ans_type)

    # Safe arithmetic eval for unevaluated expressions like 325*(1+325)
    if re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", normalized):
        try:
            result = eval(normalized)
            normalized = str(int(result)) if float(result).is_integer() else str(result)
        except Exception:
            pass

    return f"\\boxed{{{normalized}}}"


def verify_example(example, formatted_text):
    """
    Optional: sanity-check that the boxed answer in the formatted text
    can be re-extracted and still matches the gold answer.
    Skips examples where extraction fails (flags them for review).
    """
    gold = example["answer"]
    options = example.get("options", [])
    is_mcq  = bool(options)
    ans_type = "MCS" if is_mcq else "NV"  # adjust per your dataset's type field

    # Re-extract from the formatted text, exactly as eval would
    extracted = judger.extract_ans(formatted_text)
    if not extracted:
        return False, "extraction_failed"

    gold_list = [gold] if not isinstance(gold, list) else gold
    gold_list = [judger.norm_ans_str(str(g), ans_type) for g in gold_list]

    try:
        result = judger.is_equal(extracted, gold_list[0], options=options)
        return result, "ok" if result else "mismatch"
    except Exception as e:
        return False, f"judge_error: {e}"


def format_example(example):
    question = example["question"]
    answer   = example["answer"]
    options  = example.get("options")
    is_mcq   = bool(options)

    if is_mcq:
        letters = "ABCDEFGHIJ"
        options_text = "\n".join(
            f"{letters[i]}. {opt}" for i, opt in enumerate(options)
        )
        user_content  = f"{question}\n\nOptions:\n{options_text}"
        system_prompt = SYSTEM_PROMPT_MCQ
    else:
        user_content  = question
        system_prompt = SYSTEM_PROMPT_MATH

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )

    boxed_answer   = wrap_answer_in_boxed(answer, is_mcq=is_mcq)
    thinking_trace = example.get("thinking", "")

    assistant_turn = (
        f"<|im_start|>assistant\n"
        f"<think>\n{thinking_trace}\n</think>\n"
        f"{boxed_answer}"
        f"<|im_end|>"
    )

    full_text = prompt + assistant_turn

    # ── Optional verification pass ─────────────────────────────────────────
    ok, reason = verify_example(example, full_text)
    return {"text": full_text, "verified": ok, "verify_reason": reason}
    
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load your data (adjust for your file type)
dataset = load_dataset("json", data_files="data/public.jsonl")  # or "csv"

# Apply formatting
dataset = dataset.map(format_example)

#Inspect how many failed
from collections import Counter
reasons = Counter(dataset["train"]["verify_reason"])
print(reasons)
# Counter({'ok': 892, 'mismatch': 61, 'extraction_failed': 47})

# Drop unverifiable examples — better to train on less clean data than dirty data
clean_dataset = dataset.filter(lambda x: x["verified"])
print(f"Kept {len(clean_dataset['train'])} / {len(dataset['train'])} training examples")



print(clean_dataset["train"][0]["text"])
print("---")
print(clean_dataset["train"][1]["text"])

# Split into train/eval (90/10)
dataset = clean_dataset["train"].train_test_split(test_size=0.1)


from trl import GRPOTrainer, GRPOConfig
def reward_fn(completions, solution, **kwargs):
    answer = solution
    rewards = []
    
    for completion in completions:
        # Extract thinking
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""
        
        extracted = judger.extract_ans(completion)
        is_correct = bool(extracted) and judger.is_equal(extracted, answer[0])
        
        base = 1.0 if is_correct else -0.25
        
        think_score = 0.0
        if thinking:
            think_len = len(thinking.split())
            # Better step detection
            num_steps = len(re.findall(r'(?:\n|^)\s*(?:\d+\.|\-|\*|\•|Step|First|Second|Then|Therefore|Thus|Hence|Finally)', 
                                     thinking, re.IGNORECASE))
            
            length_bonus = min(think_len / 160.0, 1.0) * 0.5
            step_bonus = min(num_steps / 7.0, 1.0) * 0.4
            
            think_score = length_bonus + step_bonus
            
            if re.search(r'Let |Assume |Suppose |Consider |Proof|Therefore|Hence', thinking):
                think_score += 0.2
        has_proper_format = "<think>" in completion and "</think>" in completion and "\\boxed{" in completion
        format_bonus = 0.15 if has_proper_format else -0.1

        if is_correct:
            final_reward = base + think_score + format_bonus
        else:
            final_reward = base + (think_score * 0.35) + format_bonus
        
        final_reward = max(min(final_reward, 2.0), -0.6)
        rewards.append(final_reward)
    
    return rewards

print("Loading SFT model. Starting GRPO...")


def prepare_for_grpo(example):
    # Extract the prompt part (everything before assistant response)
    full_text = example["text"]
    
    # Split on the assistant turn
    if "<|im_start|>assistant" in full_text:
        prompt_part = full_text.split("<|im_start|>assistant")[0]
    else:
        prompt_part = full_text  # fallback
    
    return {
        "prompt": prompt_part.strip(),
        "answer": example["answer"],           # for your reward function
        # Optional: keep other fields if needed
    }

# Apply this to your split dataset
grpo_dataset = dataset.map(prepare_for_grpo, remove_columns=["text", "verified", "verify_reason"])
grpo_dataset = grpo_dataset.rename_column("answer", "solution")  # common name

print(grpo_dataset["train"][0].keys())  # Should show ['prompt', 'solution']

model = AutoModelForCausalLM.from_pretrained(
    "./qwen-math-sft/final",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

training_args = GRPOConfig(
    output_dir="./qwen-math-grpo",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,   # Add this if VRAM is tight
    learning_rate=8e-7,              # Slightly more conservative
    bf16=True,
    logging_steps=8,
    save_steps=100,
    max_completion_length=1536,      # Important!
    num_generations=4,               # Try 4-8 generations per prompt
    temperature=0.9,
    top_p=0.95,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=grpo_dataset["train"],
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model("./qwen-math-grpo/final_model")

# Also save the tokenizer
tokenizer.save_pretrained("./qwen-math-grpo/final_model")

print("Training completed and model saved!")