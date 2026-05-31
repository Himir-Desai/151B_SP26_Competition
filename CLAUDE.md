# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CSE 151B competition: improve accuracy of Qwen3-4B-Thinking-2507 on a mixed math dataset (MCQ + free-form) scored by `judger.py`. The competition metric is exact-match accuracy via the Judger.

## Commands

```bash
# Run inference + score on public dataset (100 questions)
.venv/bin/python starter.py

# Full SFT pipeline (run in order)
.venv/bin/python sft_training/generate_data.py   # rejection-sampling data collection (~2-3h with 4 GPUs)
.venv/bin/python sft_training/train.py            # LoRA fine-tuning (~30-60min)
.venv/bin/python sft_training/evaluate.py         # eval with LoRA adapter (no merge needed)

# Merge LoRA adapter into base model for deployment
.venv/bin/python merge_adapter.py
.venv/bin/python merge_adapter.py --adapter sft_training/checkpoints/adapter --output sft-2-merged

# Kill zombie GPU processes (needed after OOM crashes)
kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')

# Check free GPUs
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

## Architecture

### Inference path (`starter.py`)
Loads a merged model via vLLM, applies `build_prompt()` (selects SYSTEM_PROMPT_MATH or SYSTEM_PROMPT_MCQ based on whether `options` key exists), runs batched generation, calls `postprocess_response()` to fix known boxed-answer edge cases, then scores with `Judger.auto_judge()`. All execution is inside `if __name__ == "__main__":` to prevent vLLM spawn-mode multiprocessing errors.

### SFT pipeline (`sft_training/`)
Three sequential scripts all driven by `sft_training/config.yaml`:

1. **`generate_data.py`** — Rejection-sampling: generates N responses per question with vLLM, keeps only correct ones (Judger-verified + `\boxed{}` present), formats as `<think>…</think>\n\boxed{answer}`. Supports data-parallel generation: when `num_data_parallel > 1`, spawns N subprocesses each on a separate GPU (avoids tensor-parallel overhead for 4B models). Also pulls external math datasets (NuminaMath-CoT, hendrycks/competition_math) and reformats them via `format_external_solution()` to enforce the single-boxed-at-end format.

2. **`train.py`** — LoRA fine-tuning via TRL `SFTTrainer`. Reads `train.jsonl`/`val.jsonl`, applies chat template, trains with PEFT LoRA (r=16). Uses `hardware.gpu_id` from config.

3. **`evaluate.py`** — Loads base model + LoRA adapter via vLLM `LoRARequest` (no merge needed), runs inference matching starter.py params exactly (no thinking_budget, max_tokens=32768).

### Scoring (`judger.py`)
`Judger.auto_judge(pred, gold, options)` is the main entry point. It:
- Calls `extract_ans()` → `extract_boxed_answer()` which strips `</think>` then finds the last contiguous group of `\boxed{}` expressions
- Splits by comma for multi-part answers
- Normalizes via `norm_math_str()` (LaTeX normalization, sympy parsing)
- Tries all judge types: NV (numerical), EX (expression), EQ (equation), OL/UOL (lists), TF, MCS/MCM, OE

### GPU allocation convention
- `hardware.gpu_id` (default: `"1"`) → used by `train.py` and `evaluate.py`
- `rejection_sampling.gpu_ids` (default: `"3,5,6,7"`) → used by `generate_data.py` for data-parallel workers

## Key Design Decisions

**Prompt alignment**: `SYSTEM_PROMPT_MATH` and `SYSTEM_PROMPT_MCQ` in `starter.py`, `generate_data.py`, and `evaluate.py` must be identical. Mismatch between training and inference prompts directly causes accuracy drops.

**No thinking_budget in inference**: `starter.py` does NOT pass `chat_template_kwargs={"thinking_budget": N}`. The training pipeline uses `thinking_budget: 4096` for speed during data generation only. Do not add thinking_budget to starter.py without testing — it has historically reduced accuracy on this model.

**Data parallel > tensor parallel for 4B models**: Tensor parallelism adds all-reduce overhead that negates gains for small models. Use `num_data_parallel: 4` with separate single-GPU workers instead.

**Merged model for deployment**: `evaluate.py` uses base model + LoRA via vLLM `LoRARequest` (no merge needed). `starter.py` uses a merged model (run `merge_adapter.py` first, then set `MODEL_ID` to the output dir).

**vLLM spawn guard**: Any script that creates `LLM(...)` must wrap all execution code and GPU imports inside `if __name__ == "__main__":`. vLLM uses spawn (not fork) for worker processes.

## Current Accuracy Baseline

| Model | Prompt | MCQ | Free-form | Overall |
|-------|--------|-----|-----------|---------|
| Base Qwen3-4B | Simple | 78.9% | 51.6% | 62.0% |
| Base Qwen3-4B | Improved (7-rule) | ~84% | ~62% | 71.0% |
| SFT-2 (this run) | Improved (7-rule) | 81.6% | 61.3% | 69.0% |
