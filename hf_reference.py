"""Reference probabilities from Hugging Face transformers (what a typical training stack produces).
Compare with the vLLM /classify output of test_endpoint.py to confirm serving parity. Needs a CUDA GPU.

  python hf_reference.py [CHECKPOINT_DIR]   (default: model/, built by env/build_checkpoint.py)
"""
import json
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from test_endpoint import DOCS, LABELS, PROMPT

path = sys.argv[1] if len(sys.argv) > 1 else "model"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
print(type(model).__name__, model.config.id2label)

out = []
with torch.no_grad():
    for gold, doc in DOCS:
        ids = tok(PROMPT.format(doc=doc), return_tensors="pt").to("cuda")
        logits = model(**ids).logits.float()[0]
        probs = torch.softmax(logits, -1).tolist()
        pred = LABELS[int(torch.argmax(logits))]
        print(f"gold={gold:10} pred={pred:10} " + ", ".join(f"{l}:{p:.3f}" for l, p in zip(LABELS, probs)))
        out.append({"gold": gold, "pred": pred, "probs": dict(zip(LABELS, probs))})
json.dump(out, open("hf_reference.json", "w"), indent=2)
