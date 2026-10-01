#!/usr/bin/env bash
# Fill Table 1: every policy, one run, one report.
set -euo pipefail
CONFIG="${1:-configs/default.yaml}"
OUT="${2:-runs/main.json}"
python -m src.eval --config "$CONFIG" --all-policies --out "$OUT"
