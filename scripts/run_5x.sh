#!/usr/bin/env bash
# run_5x.sh — Run the rotation counter 5 consecutive times and log all outputs.
# Usage: ./scripts/run_5x.sh [VIDEO_PATH]

set -euo pipefail

VIDEO="${1:-rotationsTest.qt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "================================================"
echo "  Rotation Counter — 5x Consistency Test"
echo "  Video: $VIDEO"
echo "  $(date)"
echo "================================================"

PASS=0
FAIL=0
RESULTS=()
TIMES=()
STATS_FILE="$PROJECT_DIR/last_run_stats.txt"

for i in $(seq 1 5); do
    echo ""
    echo "--- Run $i/5 ---"
    RUN_START=$(date +%s%N)
    OUTPUT=$(python main.py "$VIDEO" 2>&1)
    RUN_END=$(date +%s%N)
    ELAPSED_MS=$(( (RUN_END - RUN_START) / 1000000 ))
    ELAPSED_S=$(awk "BEGIN {printf \"%.2f\", $ELAPSED_MS / 1000}")

    # Extract the final integer (last non-empty line)
    COUNT=$(echo "$OUTPUT" | tail -n 1 | tr -d '[:space:]')

    echo "$OUTPUT"
    echo ""
    echo "→ Count returned: $COUNT"
    echo "→ Elapsed: ${ELAPSED_S}s"

    RESULTS+=("$COUNT")
    TIMES+=("$ELAPSED_S")

    if [[ "$COUNT" == "5" ]]; then
        ((PASS++))
    else
        ((FAIL++))
    fi

    # Brief pause to avoid rate limit issues on rapid consecutive runs
    if [[ $i -lt 5 ]]; then
        sleep 2
    fi
done

echo ""
echo "================================================"
echo "  RESULTS SUMMARY"
echo "================================================"
for i in "${!RESULTS[@]}"; do
    echo "  Run $((i+1)): ${RESULTS[$i]}  (${TIMES[$i]}s)"
done
echo ""
echo "  Passed: $PASS/5"
echo "  Failed: $FAIL/5"

if [[ $FAIL -eq 0 ]]; then
    STATUS="✓ ALL RUNS CORRECT"
    echo "  STATUS: $STATUS"
    EXIT_CODE=0
else
    STATUS="✗ CONSISTENCY FAILURE"
    echo "  STATUS: $STATUS"
    EXIT_CODE=1
fi

# Write stats file
{
    echo "================================================"
    echo "  Rotation Counter — 5x Consistency Test"
    echo "  Video: $VIDEO"
    echo "  $(date)"
    echo "================================================"
    echo ""
    for i in "${!RESULTS[@]}"; do
        echo "  Run $((i+1)): ${RESULTS[$i]}  (${TIMES[$i]}s)"
    done
    echo ""
    echo "  Passed: $PASS/5"
    echo "  Failed: $FAIL/5"
    echo "  STATUS: $STATUS"
} | tee "$STATS_FILE" > /dev/null

echo ""
echo "  Stats written to: $STATS_FILE"

exit $EXIT_CODE
