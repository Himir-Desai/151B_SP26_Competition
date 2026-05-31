"""
Step 2 — LoRA SFT training with TRL SFTTrainer.

Reads config.yaml and sft_training/data/train.jsonl + val.jsonl.
Saves LoRA adapter to config.model.output_dir.

Run after generate_data.py.
"""

import json
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Load config ───────────────────────────────────────────────────────────────
cfg_path = Path(__file__).parent / "config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["hardware"]["gpu_id"]

model_cfg = cfg["model"]
lora_cfg = cfg["lora"]
train_cfg = cfg["training"]
data_cfg = cfg["data"]

# ── Load data ─────────────────────────────────────────────────────────────────
def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

train_records = load_jsonl(data_cfg["train_output"])
val_records   = load_jsonl(data_cfg["val_output"])
print(f"Train: {len(train_records)} | Val: {len(val_records)}")

train_ds = Dataset.from_list(train_records)
val_ds   = Dataset.from_list(val_records)

# ── Load tokenizer ────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    model_cfg["model_id"], trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_example(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

train_ds = train_ds.map(lambda x: {"text": format_example(x)}, remove_columns=["messages"])
val_ds   = val_ds.map(lambda x: {"text": format_example(x)}, remove_columns=["messages"])

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_cfg["model_id"],
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

if train_cfg["gradient_checkpointing"]:
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

# ── Apply LoRA ────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=lora_cfg["r"],
    lora_alpha=lora_cfg["lora_alpha"],
    target_modules=lora_cfg["target_modules"],
    lora_dropout=lora_cfg["lora_dropout"],
    bias=lora_cfg["bias"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Training args ─────────────────────────────────────────────────────────────
output_dir = model_cfg["output_dir"]
Path(output_dir).mkdir(parents=True, exist_ok=True)

warmup_steps = max(1, int(
    train_cfg["warmup_ratio"]
    * train_cfg["num_train_epochs"]
    * len(train_ds)
    // (train_cfg["per_device_train_batch_size"] * train_cfg["gradient_accumulation_steps"])
))

sft_args = SFTConfig(
    output_dir=output_dir,
    # data
    dataset_text_field="text",
    max_length=train_cfg["max_seq_length"],
    packing=False,
    # optimisation
    num_train_epochs=train_cfg["num_train_epochs"],
    per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
    per_device_eval_batch_size=train_cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
    learning_rate=train_cfg["learning_rate"],
    lr_scheduler_type=train_cfg["lr_scheduler_type"],
    warmup_steps=warmup_steps,
    weight_decay=train_cfg["weight_decay"],
    bf16=train_cfg["bf16"],
    fp16=train_cfg["fp16"],
    # logging / saving
    logging_steps=train_cfg["logging_steps"],
    save_strategy=train_cfg["save_strategy"],
    save_steps=train_cfg["save_steps"],
    eval_strategy=train_cfg["eval_strategy"],
    eval_steps=train_cfg["eval_steps"],
    load_best_model_at_end=train_cfg["load_best_model_at_end"],
    metric_for_best_model=train_cfg["metric_for_best_model"],
    dataloader_num_workers=train_cfg["dataloader_num_workers"],
    torch_compile=train_cfg.get("torch_compile", False),
    report_to="none",
    remove_unused_columns=False,
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

print("Starting training...")
trainer.train()

# ── Save adapter ──────────────────────────────────────────────────────────────
adapter_dir = model_cfg["adapter_dir"]
Path(adapter_dir).mkdir(parents=True, exist_ok=True)
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"Adapter saved → {adapter_dir}")
