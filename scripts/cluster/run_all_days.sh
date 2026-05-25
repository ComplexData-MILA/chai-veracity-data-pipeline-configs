#!/usr/bin/env bash
# Run DBSCAN + HNSW clustering on all available days, one day at a time.
# Usage: ./scripts/cluster/run_all_days.sh [--skip-upload] [--eps 0.7] [--min-samples 5] ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# Source secrets
set -a
source .env
set +a

# Memory limit: 32 GB virtual address space (in KB)
#MEMLIMIT_KB=33554432
MEMLIMIT_KB=50331648


# --- collect all available days -----------------------------------------------
echo "=== Listing available days ==="
DAYS=$(
    uv run python -c "
import asyncio
from scripts.cluster.clustering_simplified import _list_batches, _group_batches_by_day
async def main():
    batches = await _list_batches()
    days = _group_batches_by_day(batches)
    for d in sorted(days):
        print(d)
asyncio.run(main())
" 2>/dev/null
)

if [ -z "$DAYS" ]; then
    echo "ERROR: no days found."
    exit 1
fi

readarray -t DAY_ARRAY <<< "$DAYS"
echo "Found ${#DAY_ARRAY[@]} days: ${DAY_ARRAY[*]}"
echo ""

# --- process each day ---------------------------------------------------------
PASSED_ARGS="$*"
FAILED_DAYS=()

for day in "${DAY_ARRAY[@]}"; do
    echo "============================================================"
    echo "  Day: $day  ( $(date '+%Y-%m-%d %H:%M:%S') )"
    echo "============================================================"

    set +e
    ulimit -v "$MEMLIMIT_KB" && \
    uv run python scripts/cluster/clustering_simplified.py \
        --day "$day" \
        $PASSED_ARGS \
        2>&1
    RC=$?
    set -e

    if [ $RC -ne 0 ]; then
        echo "ERROR: day $day failed with exit code $RC"
        FAILED_DAYS+=("$day")
    else
        echo "OK: day $day completed successfully."
    fi
    echo ""
done

# --- summary ------------------------------------------------------------------
echo "============================================================"
echo "  Summary"
echo "============================================================"

if [ ${#FAILED_DAYS[@]} -eq 0 ]; then
    echo "All ${#DAY_ARRAY[@]} days completed successfully."
else
    echo "${#FAILED_DAYS[@]} / ${#DAY_ARRAY[@]} days failed:"
    for d in "${FAILED_DAYS[@]}"; do
        echo "  - $d"
    done
    exit 1
fi
