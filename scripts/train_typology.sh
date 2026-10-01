#!/usr/bin/env bash
# Train the interruption typology classifier, then switch the config to use it.
set -euo pipefail
CONFIG="${1:-configs/default.yaml}"
python -m src.train --config "$CONFIG"
echo
echo "Now set restate.typology_model: learned in $CONFIG"
