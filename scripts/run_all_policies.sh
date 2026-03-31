#!/bin/bash
# Run all 4 reset policies for comparison.
# Usage: bash scripts/run_all_policies.sh [max_episodes] [seed]

MAX_EPISODES=${1:-100}
SEED=${2:-42}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== ABD Policy Comparison ==="
echo "Max episodes: $MAX_EPISODES, Seed: $SEED"
echo ""

for POLICY in no_reset periodic naive abd; do
    LOG_DIR="outputs/logs/${POLICY}_${TIMESTAMP}"
    echo "--- Running policy: $POLICY -> $LOG_DIR ---"
    python scripts/main.py \
        policy=$POLICY \
        max_episodes=$MAX_EPISODES \
        seed=$SEED \
        log_dir=$LOG_DIR
    echo ""
done

echo "=== All policies complete ==="
echo "Run 'python scripts/analyze_results.py outputs/logs/*_${TIMESTAMP}' to compare results."
