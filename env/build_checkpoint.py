"""Build a stand-in classifier checkpoint: Qwen3.5-0.8B (text) + a sequence-classification head.

A fine-tuned checkpoint has a *trained* `score.weight` [num_labels, hidden]. Here the head is
initialised from the LM-head rows of each label's first token (Qwen3.5 ties lm_head to embed_tokens),
which gives a meaningful zero-shot classifier with the exact same on-disk format:
  architectures = ["Qwen3_5ForSequenceClassification"], full backbone weights + score.weight.

Usage: python env/build_checkpoint.py [BASE_DIR] [OUT_DIR]   (defaults: ./base ./model)
Also run inside the cloud image build (env/Dockerfile.baked).
"""
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "base"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "model"
LABELS = ["NDA", "Employment", "Lease", "Loan", "Purchase", "Invoice", "Other"]

tok = Tokenizer.from_file(str(BASE / "tokenizer.json"))
first_ids = []
for label in LABELS:
    ids = tok.encode(label, add_special_tokens=False).ids
    print(f"{label!r:14} -> {ids} {[tok.id_to_token(i) for i in ids]}")
    first_ids.append(ids[0])
assert len(set(first_ids)) == len(first_ids), "label first-tokens must be unique"

state = load_file(str(BASE / "model.safetensors"))
emb = state["model.embed_tokens.weight"]
state["score.weight"] = emb[first_ids].clone().contiguous()
print("score.weight", tuple(state["score.weight"].shape), state["score.weight"].dtype)

OUT.mkdir(parents=True, exist_ok=True)
save_file(state, str(OUT / "model.safetensors"), metadata={"format": "pt"})

cfg = json.loads((BASE / "config.json").read_text())
cfg["architectures"] = ["Qwen3_5ForSequenceClassification"]
cfg["id2label"] = {str(i): l for i, l in enumerate(LABELS)}
cfg["label2id"] = {l: i for i, l in enumerate(LABELS)}
cfg["num_labels"] = len(LABELS)
cfg["pad_token_id"] = 248044
(OUT / "config.json").write_text(json.dumps(cfg, indent=2))

for f in ["tokenizer.json", "tokenizer_config.json"]:
    shutil.copy(BASE / f, OUT / f)
print("wrote", OUT, sorted(p.name for p in OUT.iterdir()))
