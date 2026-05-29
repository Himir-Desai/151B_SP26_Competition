from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from judger import Judger
import re

judger = Judger()

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
SYSTEM_PROMPT_MCQ = (
    "You are a nobel prize winning mathematician tasked to solve math problems easy or difficult."
    "As you are an AI Model you should not fall into the common problems faced by AI models such as being overconfident in wrong answers or being misled by irrelevant information."
    "You should answer the questions like a professional mathematician who is careful and meticulous in solving problems."
    "Read the problem and the answer choices below, then select the single best answer from the options only."
    "If your answer does not match any of the options, you should choose the option that is closest to your answer."
    "For answers that include constants like pi or e, if none of the options match you can input the values of such constants (e.g. 3.14159 for pi) and choose the closest option."
    "The most important thing is to put your final answer inside \\boxed{} without failure."
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

def normalize_answer_with_judger(answer, ans_type=None):
    if isinstance(answer, list):
        normalized = [judger.norm_ans_str(str(a).strip(), ans_type) for a in answer]
        return normalized
    return judger.norm_ans_str(str(answer).strip(), ans_type)


def wrap_answer_in_boxed(answer, is_mcq=False):
    ans_type = "MCS" if is_mcq else None

    if isinstance(answer, list):
        normalized = normalize_answer_with_judger(answer, ans_type)
        inner = ", ".join(normalized)
        return f"\\boxed{{{inner}}}"

    answer = str(answer).strip()

    if "\\boxed{" in answer:
        return answer

    if is_mcq:
        normalized = judger.norm_ans_str(answer, "MCS")
        letter = re.match(r"^([A-Ja-j])[.):\s]?", normalized)
        if letter:
            return f"\\boxed{{{letter.group(1).upper()}}}"

    normalized = judger.norm_ans_str(answer, ans_type)

    if re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", normalized):
        try:
            result = eval(normalized)
            normalized = str(int(result)) if float(result).is_integer() else str(result)
        except Exception:
            pass

    return f"\\boxed{{{normalized}}}"


def verify_example(example, formatted_text):
    gold = example["answer"]
    options = example.get("options", [])
    is_mcq  = bool(options)
    ans_type = "MCS" if is_mcq else "NV"

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


def generate_thinking_trace(question, answer, options=None, is_mcq=False):
    ans_str = (
        str(answer).strip() if not isinstance(answer, list)
        else ", ".join(str(a) for a in answer)
    )

    if is_mcq:
        letters = "ABCDEFGHIJ"
        gold_letter = str(answer).strip().upper()
        lines = [
            "Let me carefully read and analyze this problem.",
            "",
            f"The question asks: {question[:200].strip()}",
            "",
            "I will evaluate each option systematically:",
        ]
        if options:
            for i, opt in enumerate(options[:10]):
                letter = letters[i]
                if letter == gold_letter:
                    lines.append(f"- Option {letter}: {opt[:100].strip()} — This satisfies the problem conditions.")
                else:
                    lines.append(f"- Option {letter}: {opt[:100].strip()} — This does not satisfy the requirements.")
        lines += [
            "",
            f"After careful elimination and analysis, option {gold_letter} is the correct answer.",
            "The other options can be ruled out as they either contradict the given conditions",
            "or don't match the expected result.",
        ]
        return "\n".join(lines)
    else:
        lines = [
            "Let me solve this problem step by step.",
            "",
            "First, I identify what we're being asked to find.",
            f"Problem: {question[:300].strip()}",
            "",
            "Setting up the approach:",
            "- Identify the key quantities and relationships in the problem.",
            "- Choose the appropriate mathematical method or formula.",
            "- Work through the calculations carefully, checking each step.",
            "",
            "Working through the solution:",
            "Step 1: Parse the problem and identify all given information.",
            "Step 2: Determine which mathematical principles apply here.",
            "Step 3: Set up and perform the required calculations.",
            "Step 4: Verify the result is reasonable and matches the expected form.",
            "",
            f"After working through this carefully, the answer is: {ans_str}",
        ]
        return "\n".join(lines)


def format_example_sft(example):
    question = example["question"]
    answer   = example["answer"]
    options  = example.get("options")
    is_mcq   = bool(options)
    thinking_trace = example.get("thinking", "").strip()

    if is_mcq:
        letters = "ABCDEFGHIJ"
        options_text = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
        user_content = f"{question}\n\nOptions:\n{options_text}"
        system_prompt = SYSTEM_PROMPT_MCQ
    else:
        user_content = question
        system_prompt = SYSTEM_PROMPT_MATH

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if not thinking_trace or len(thinking_trace.split()) < 100:
        thinking_trace = generate_thinking_trace(question, answer, options, is_mcq)

    boxed_answer = wrap_answer_in_boxed(answer, is_mcq=is_mcq)

    assistant_turn = (
        f"<think>\n{thinking_trace}\n</think>\n"
        f"{boxed_answer}"
    )

    full_text = prompt + assistant_turn
    return {"text": full_text}


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

    # Only verify against the final answer, not intermediate \boxed{} in the thinking trace
    verify_text = full_text.split("</think>")[-1]
    ok, reason = verify_example(example, verify_text)
    return {"text": full_text, "verified": ok, "verify_reason": reason}


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_dataset("json", data_files="data/refined.jsonl")

dataset = dataset.map(format_example)

from collections import Counter
reasons = Counter(dataset["train"]["verify_reason"])
print(reasons)

print(f"Verified {sum(dataset['train']['verified'])} / {len(dataset['train'])} training examples (training on all)")

dataset = dataset["train"].train_test_split(test_size=0.1)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.gradient_checkpointing_enable()
model.config.use_cache = False

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

dataset_sft = dataset.map(format_example_sft)

sft_args = SFTConfig(
    output_dir="./qwen-math-sft3",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    packing=False,
)

sft_trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_sft["train"],
    processing_class=tokenizer,
    args=sft_args,
)

print("Starting SFT training...")
sft_trainer.train()

sft_trainer.save_model("./qwen-math-sft3/final")
tokenizer.save_pretrained("./qwen-math-sft3/final")

print("Training completed and model saved!")
