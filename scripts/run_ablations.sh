#!/usr/bin/env bash
# Fill Figure 2(b): ReState with each component removed in turn.
set -euo pipefail
CONFIG="${1:-configs/default.yaml}"
OUT="${2:-runs/abl.json}"
python -m src.eval --config "$CONFIG" --ablations --out "$OUT"
