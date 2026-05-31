"""
Merges the LoRA adapter into the base model and saves a standalone model.

Usage:
    .venv/bin/python merge_adapter.py
    .venv/bin/python merge_adapter.py --adapter sft_training/checkpoints/adapter --output merged_model
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter (overrides config)")
parser.add_argument("--output",  type=str, default=None, help="Where to save merged model (default: merged_model/)")
args = parser.parse_args()

cfg_path = Path("sft_training/config.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["hardware"]["gpu_id"]

model_id   = cfg["model"]["model_id"]
adapter_dir = args.adapter or cfg["model"]["adapter_dir"]
output_dir  = args.output  or "merged_model"

print(f"Base model : {model_id}")
print(f"Adapter    : {adapter_dir}")
print(f"Output     : {output_dir}")

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, adapter_dir)

print("Merging...")
model = model.merge_and_unload()

print(f"Saving merged model → {output_dir}")
Path(output_dir).mkdir(parents=True, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Done. Use the merged model by pointing model_id to:", output_dir)
