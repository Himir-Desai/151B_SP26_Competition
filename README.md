# CSE 151B Spring 2026 Competition

Mixed math dataset (MCQ + free-form) — scored by exact-match accuracy via `judger.py`.

**Hardware inference times** (private test set, 943 questions, 300 MCQ and 543 FRQ):
- A6000 GPU: ~2 hours
- B200 GPU: ~7 minutes

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Inference

**Option 1 — Base model** (sometimes better):
```bash
python run_inference.py
```
Uses `Qwen/Qwen3-4B-Thinking-2507` from HuggingFace directly.

**Option 2 — LoRA fine-tuned model** (sometimes better):
```bash
python run_inference_2.py
```
Uses `darthsid12/sft3`, a LoRA-fine-tuned checkpoint of the base model.

Both scripts write results to `results/submission.csv`.

---

## `run_inference.py` Pipeline

1. **Load dataset** — reads `data/private.jsonl`; splits questions into MCQ (has `options` key) and free-form.

2. **Build prompts** (`prompt_builder.py`) — for each question:
   - MCQ questions get `SYSTEM_MC_TYPED_V1`: instructs the model to solve independently first, then match against options, and output only `\boxed{LETTER}` after the thinking block.
   - Free-form questions get `SYSTEM_FF_TYPED_V1`: instructs exact symbolic answers (no decimals), proper `\boxed{}` formatting, and comma-separated multi-part answers.
   - Per-question traits are detected (high-precision numeric, exact symbolic, multi-select, tuple/interval, etc.) and injected as targeted requirements into the user turn.

3. **Load model via vLLM** — loads `Qwen/Qwen3-4B-Thinking-2507` in `bfloat16` with prefix caching enabled (`max_model_len=32768`).

4. **Generate responses** — batched generation with `temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_tokens=32768`. The model reasons inside a `<think>…</think>` block before emitting the final `\boxed{}` answer.

5. **Normalize answers** (`answer_parser.py`) — strips LaTeX whitespace markers, normalizes `\dfrac`→`\frac`, converts `sqrt(x)`→`\sqrt{x}`, canonicalizes `\infty`, repairs multi-part formatting, and applies question-aware fixes (currency rounding, percent-growth formulas, letter-list concatenation for multi-select).

6. **Extract final answer**:
   - MCQ: regex searches for the last `\boxed{LETTER}` after `</think>`, falls back to the last uppercase letter.
   - Free-form: calls `judger.extract_ans()` with a 15-second timeout per question.

7. **Save CSV** — writes `id` and `response` columns to `results/submission.csv`.
