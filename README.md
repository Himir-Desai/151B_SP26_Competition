# CSE 151B/251B Spring 2026 — Mathematical Reasoning Competition
**Team: Clanker Bashers**

## Hardware
- **Training:** NVIDIA A6000 (48GB VRAM)
- **Inference:** NVIDIA A6000 (48GB VRAM) / NVIDIA RTX 3070 Ti (8GB VRAM)
- **Approximate inference time:** ~3-4 minutes for 96 prompts on A6000

---

## Model Weights

Our fine-tuned model is hosted on HuggingFace Hub at:
```
[YOUR-USERNAME/YOUR-MODEL-NAME]
```

To download and set up the model weights:
```bash
huggingface-cli login
huggingface-cli download [YOUR-USERNAME/YOUR-MODEL-NAME] --local-dir ./qwen-math-sft/merged
```

Place the downloaded model in the following directory:
```
151B_SP26_Competition/
└── qwen-math-sft/
    └── merged/
        ├── model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── ...
```

---

## How to Run Inference

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run inference
```python
from run_inference import run_inference
run_inference()
```

Or directly via command line:
```bash
python run_inference.py
```

This will:
- Load the fine-tuned model from `qwen-math-sft/merged`
- Run inference on the private dataset
- Output the final submission CSV to `results/submission.csv`

---

## Approach Summary

### Training
- **Base model:** Qwen/Qwen3-4B-Thinking-2507
- **Fine-tuning method:** Supervised Fine-Tuning (SFT) via LoRA (Low-Rank Adaptation) using the PEFT library
- **Training data:** open-r1/OpenR1-Math-220k — trained on thinking traces + final answers
- **Best SFT strategy:** Multiple iterations of the model training on its own correct responses
- **GRPO:** Attempted with reward signals for correct answer (+1), answer in `\boxed{}` (+0.3), and optimal thinking length (+0.05–0.1) — too slow to complete

### Inference
- **Engine:** vLLM
- **Separate pipelines** for MCQ and free-form questions with different token budgets
  - MCQ: `thinking_token_budget=1024`, `max_tokens=6144`
  - Free-form: `thinking_token_budget=4096`, `max_tokens=8192`
- **Double inference:** Model solves problem, then a second pass extracts the final answer — minor improvement observed
- **Prompt engineering:** Separate system prompts for MCQ vs free-form, role framing as "Nobel Prize-winning mathematician"

### Key Hyperparameters
```
max_model_len         = 10240
max_num_seqs          = 96
max_num_batched_tokens = 16384
gpu_memory_utilization = 0.90
temperature           = 0.6
top_p                 = 0.95
top_k                 = 20
```

---

## Repository Structure
```
151B_SP26_Competition/
├── data/
│   └── public.jsonl          # Public evaluation dataset
├── results/
│   └── submission.csv        # Final submission file
├── qwen-math-sft/
│   └── merged/               # Fine-tuned model weights
├── run_inference.py          # Main entry point
├── judger.py                 # Answer evaluation engine
├── requirements.txt
└── README.md
```
