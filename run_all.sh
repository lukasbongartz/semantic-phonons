#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Periodic ==="
python experiments/periodic.py --llm "$@"

echo ""
echo "=== Neumann ==="
python experiments/neumann.py --llm "$@"

echo ""
echo "=== Log scale ==="
python experiments/log_scale.py --llm "$@"

echo ""
echo "=== 2-D Neumann (Gemma 2 27B) ==="
python experiments/neumann_2d.py --llm --model google/gemma-2-27b --batch-size 4

echo ""
echo "All experiments done."
