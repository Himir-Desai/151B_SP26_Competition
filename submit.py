import csv
import json
import re
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_ID         = "qwen-math-sft2/merged"
DATA_PATH        = "data/private.jsonl"
SUBMISSION_PATH  = "results_sub/submission.csv"
OUTPUT_TEXT_PATH = "results_sub/thinking.txt"

MCQ_MAX_TOKENS      = 16384
FRQ_MAX_TOKENS      = 16384
MCQ_THINKING_BUDGET = 14336
FRQ_THINKING_BUDGET = 8192

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


def frq_postprocessing(r):
    if "infty" in r.lower():
        print("Infty seen")
        r = re.sub("infty", "infinity", r, flags=re.IGNORECASE)
    lines = r.split('\n')
    return '\n'.join(lines[:-4]).replace('oxed{""', '') + '\n'.join(lines[-4:])


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    data = [json.loads(line) for line in open(DATA_PATH)]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        max_model_len=max(MCQ_MAX_TOKENS, FRQ_MAX_TOKENS) + 2048,
        trust_remote_code=True,
        max_num_seqs=96,
        max_num_batched_tokens=16384,
        reasoning_config=ReasoningConfig(
            reasoning_start_str="<think>",
            reasoning_end_str="</think>",
        ),
    )
    print("Model loaded.")

    mcq_sampling_params = SamplingParams(
        max_tokens=MCQ_MAX_TOKENS,
        temperature=0.1,
        top_p=0.95,
        top_k=20,
        min_p=0.05,
        presence_penalty=0.0,
        repetition_penalty=1,
        thinking_token_budget=MCQ_THINKING_BUDGET,
    )
    frq_sampling_params = SamplingParams(
        max_tokens=FRQ_MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        thinking_token_budget=FRQ_THINKING_BUDGET,
    )

    run_pipeline(
        model=llm,
        dataset=data,
        mcq_sampling_params=mcq_sampling_params,
        frq_sampling_params=frq_sampling_params,
        tokenizer=tokenizer,
        output_paths={
            "submission": SUBMISSION_PATH,
            "text":       OUTPUT_TEXT_PATH,
        },
    )


# ── Pipeline ───────────────────────────────────────────────────────────────

def run_pipeline(
    model,
    dataset,
    mcq_sampling_params,
    frq_sampling_params,
    tokenizer,
    mcq_system_prompt=None,
    frq_system_prompt=None,
    output_paths=None,
):
    mcq_prompt = mcq_system_prompt or SYSTEM_PROMPT_MCQ
    frq_prompt = frq_system_prompt or SYSTEM_PROMPT_MATH

    mcq_items = [(i, d) for i, d in enumerate(dataset) if     d.get("options")]
    frq_items = [(i, d) for i, d in enumerate(dataset) if not d.get("options")]
    print(f"Pipeline: {len(mcq_items)} MCQ + {len(frq_items)} FRQ (of {len(dataset)} total)")

    responses = [None] * len(dataset)

    if mcq_items:
        mcq_responses = run_mcq_questions(
            model=model,
            mcq_system_prompt=mcq_prompt,
            input_prompts=[d for _, d in mcq_items],
            tokenizer=tokenizer,
            sampling_params=mcq_sampling_params,
        )
        for (i, _), resp in zip(mcq_items, mcq_responses):
            responses[i] = resp

    if frq_items:
        frq_responses = run_frq_questions(
            model=model,
            frq_system_prompt=frq_prompt,
            input_prompts=[d for _, d in frq_items],
            tokenizer=tokenizer,
            sampling_params=frq_sampling_params,
        )
        for (i, _), resp in zip(frq_items, frq_responses):
            responses[i] = resp

    _save_submission(dataset, responses, output_paths or {})


# ── Question runners ───────────────────────────────────────────────────────

def run_mcq_questions(
    model,
    mcq_system_prompt,
    input_prompts,
    tokenizer,
    sampling_params,
    postprocess_fn=lambda x: x,
):
    prompts = [
        build_prompt(d["question"], mcq_system_prompt, tokenizer, d.get("options"))
        for d in input_prompts
    ]
    print(f"Generating {len(prompts)} MCQ responses...")
    outputs = model.generate(prompts, sampling_params)
    return [postprocess_fn(out.outputs[0].text.strip()) for out in outputs]


def run_frq_questions(
    model,
    frq_system_prompt,
    input_prompts,
    tokenizer,
    sampling_params,
    postprocess_fn=frq_postprocessing,
):
    prompts = [
        build_prompt(d["question"], frq_system_prompt, tokenizer)
        for d in input_prompts
    ]
    print(f"Generating {len(prompts)} FRQ responses...")
    outputs = model.generate(prompts, sampling_params)
    return [postprocess_fn(out.outputs[0].text.strip()) for out in outputs]


# ── Helpers ────────────────────────────────────────────────────────────────

def build_prompt(question, system_prompt, tokenizer, options=None):
    if options:
        letters      = "ABCDEFGHIJ"
        options_text = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
        user_content = f"{question}\n\nOptions:\n{options_text}"
    else:
        user_content = question
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def _split_thinking(response):
    if "</think>" in response:
        thinking, resp = response.split("</think>", 1)
        return thinking.strip(), resp.strip()
    return "", response.strip()


def _save_submission(items, responses, output_paths):
    sub_path = Path(output_paths.get("submission", "results_sub/submission.csv"))
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sub_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "response"])
        for item, response in zip(items, responses):
            thinking, clean_response = _split_thinking(response)
            full_trace = f"<think>{thinking}</think>{clean_response}" if thinking else clean_response
            writer.writerow([item["id"], full_trace])
    print(f"Saved {len(items)} records to {sub_path}")

    text_path = Path(output_paths.get("text", "results_sub/thinking.txt"))
    text_path.parent.mkdir(parents=True, exist_ok=True)
    with open(text_path, "w") as f:
        for i, (item, response) in enumerate(zip(items, responses)):
            thinking, clean_response = _split_thinking(response)
            qtype = "MCQ" if item.get("options") else "FRQ"
            tw = len(thinking.split())
            rw = len(clean_response.split())
            f.write("=" * 70 + "\n")
            f.write(f"Q{i:03d} | ID:{item['id']} | {qtype}\n")
            f.write(f"       think={tw}w  response={rw}w\n")
            f.write("=" * 70 + "\n")
            f.write(f"QUESTION:\n{item['question']}\n")
            f.write(f"\nTHINKING ({tw} words):\n{thinking}\n")
            f.write(f"\nRESPONSE ({rw} words):\n{clean_response}\n\n")
    print(f"Saved text output to {text_path}")


if __name__ == "__main__":
    main()
