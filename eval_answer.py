import json
from judger_copy import Judger
false = False
# ── Edit your entries here ────────────────────────────────────────────────────
ENTRIES = [
{"id": 202, "gold": ["(8,infinity)"], "extracted": "(8,infinity]", "correct": False}
]
# ─────────────────────────────────────────────────────────────────────────────

judger = Judger()

for entry in ENTRIES:
    gold = entry["gold"]
    extracted = entry["extracted"]
    pred = f"\\boxed{{{extracted}}}"
    options = [[] for _ in gold]
    normed_g = judger.norm_ans_str(gold[0])
    normed_p = judger.norm_ans_str(extracted.split(',')[0])
    print("Normal g")
    print(normed_g)
    print("Normal p")
    print(normed_p)
    result = judger.auto_judge(pred, gold, options)
    print(json.dumps({**entry, "judged": result}))
