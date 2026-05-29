import shutil
import os
import torch
from peft import AutoPeftModelForCausalLM

model_name = "grpo"
file_end = "checkpoint-400"
model = AutoPeftModelForCausalLM.from_pretrained(
    f"qwen-math-{model_name}/{file_end}",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
merged = model.merge_and_unload()
merged.save_pretrained(f"qwen-math-{model_name}/merged")

for f in os.listdir(f"qwen-math-{model_name}/{file_end}"):
    if f.startswith("tokenizer") or f == "chat_template.jinja":
        shutil.copy(f"qwen-math-{model_name}/{file_end}/{f}", f"qwen-math-{model_name}/merged/{f}")

print(f"Done — merged model saved to qwen-math-{model_name}/merged")
