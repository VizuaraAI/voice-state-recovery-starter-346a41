#!/usr/bin/env bash
# MultiWOZ downloads itself on first use. SpokenWOZ needs a manual step.
set -euo pipefail

echo "== MultiWOZ 2.2 =="
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("multi_woz_v22", split="validation", trust_remote_code=True)
print(f"MultiWOZ ready: {len(ds)} validation dialogues cached.")
PY

echo
echo "== SpokenWOZ =="
echo "SpokenWOZ is not redistributable, so download it yourself:"
echo "  1. Open https://spokenwoz.github.io/"
echo "  2. Download the text annotations (text_5700_train_dev)."
echo "  3. Unpack into data/spokenwoz/ so that this path exists:"
echo "       data/spokenwoz/text_5700_train_dev/data.json"
echo "  4. Set data.source: spokenwoz in your config."
echo
echo "Paper reference: Si et al., SpokenWOZ, NeurIPS 2023."
echo "  https://arxiv.org/abs/2305.13040"
