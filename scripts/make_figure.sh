#!/usr/bin/env bash
# Regenerate the paper's Figure 2 from your own measured numbers.
set -euo pipefail
MAIN="${1:-runs/main.json}"
ABL="${2:-runs/abl.json}"
OUT="${3:-../paper/figures/fig2_results.png}"
python -m src.plot_results --main "$MAIN" --ablations "$ABL" --out "$OUT"
echo "Figure written to $OUT. Update the caption: the values are yours now."
