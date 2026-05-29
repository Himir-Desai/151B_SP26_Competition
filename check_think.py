import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========================= CONFIG =========================
MODEL_PATH = "./qwen-math-grpo/merged"   # ← Change this to your actual SFT folder

SYSTEM_PROMPT = (
    "You are a Nobel Prize winning mathematician. Solve the problem step-by-step. "
    "Think deeply and extensively. Write a detailed thinking trace inside <think>...</think> "
    "before giving your final answer. Put your final answer inside \\boxed{}."
)

TEST_QUESTIONS = [
    "What is the integral of x^2 from 0 to 1?",
    "Solve for x: 2x + 5 = 17",
    "What is 15% of 240?",
    "Find the area of a circle with radius 7.",
    "If a triangle has sides 3, 4, and 5, what is its area?",
    "What is the derivative of sin(x^2)?",
    "A train travels 60 km/h. How long to travel 240 km?",
    "Solve the equation: x² - 5x + 6 = 0",
    "What is 25% of 80 plus 40% of 150?",
]

# =========================================================

print(f"Loading model from: {MODEL_PATH}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Prepare prompts
prompts = []
for question in TEST_QUESTIONS:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    prompts.append(prompt)

# Tokenize batch
inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)

print("Generating all answers in batch...\n")

# Batch generation (faster on A6000)
outputs = model.generate(
    **inputs,
    max_new_tokens=8192,
    temperature=0.75,
    top_p=0.92,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

# Analyze results
total_think_words = 0

print("=" * 80)

for i, output in enumerate(outputs):
    generated_text = tokenizer.decode(output, skip_special_tokens=False)
    question = TEST_QUESTIONS[i]
    
    print(f"Q{i+1}: {question}")

    # Use total output length as thinking length
    words = len(generated_text.split())
    total_think_words += words
    print(f"Output length: {words} words")
    print(f"Preview: {generated_text}")
    
    # Extract answer
    boxed = re.search(r"\\boxed\{(.*?)\}", generated_text, re.DOTALL)
    if boxed:
        print(f"Answer: \\boxed{{{boxed.group(1)}}}")
    
    print("-" * 80)

# Summary
avg_words = total_think_words / len(TEST_QUESTIONS)
print(f"\n=== FINAL SUMMARY ===")
print(f"Average output length: {avg_words:.1f} words across {len(TEST_QUESTIONS)} questions")

if avg_words >= 100:
    print("✅ EXCELLENT - Deep thinking preserved after SFT")
elif avg_words >= 70:
    print("✅ GOOD - Thinking is solid")
elif avg_words >= 45:
    print("⚠️  MEDIUM - Thinking is acceptable but could be deeper")
else:
    print("❌ WEAK - SFT likely collapsed the thinking behavior")

print("\nCheck completed!")