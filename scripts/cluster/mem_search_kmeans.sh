#!/bin/bash
# Binary search for minimum memory (GB) needed to run clustering_kmeans.py
# without OOM.  Uses --skip-upload to isolate the CPU/memory cost of
# collection + k-means++ + simhash dedup + sampling.
#
# Usage:  bash scripts/cluster/mem_search_kmeans.sh [MIN_GB] [MAX_GB]
#           MIN_GB   lower bound in GB  (default: 1)
#           MAX_GB   upper bound in GB  (default: 128)

set -euo pipefail

export PROJECT_HOME="${PROJECT_HOME:-$HOME/20260331-chai-veracity}"
cd "$PROJECT_HOME"

MIN_GB="${1:-1}"
MAX_GB="${2:-128}"
TOLERANCE_GB="${3:-1}"

LPAR="("; RPAR=")"
LOGDIR="$PROJECT_HOME/logs/mem_search"
mkdir -p "$LOGDIR"

# ------------------------------------------------------------------
run_trial() {
    local mem_gb="$1"
    local label="mem${mem_gb}gb"
    local out="$LOGDIR/${label}.out"
    local err="$LOGDIR/${label}.err"

    echo "=== Submitting trial: --mem=${mem_gb}G ==="
    local job_id
    job_id=$(sbatch --parsable \
        -J "mem-${mem_gb}g" \
        -p long-cpu \
        -c 8 \
        --mem="${mem_gb}G" \
        -t 2:00:00 \
        -o "$out" \
        -e "$err" \
        --wrap="
set -e
cd $PROJECT_HOME
source $PROJECT_HOME/.venv/bin/activate
source $PROJECT_HOME/.env
python scripts/cluster/clustering_kmeans.py \\
    --limit 1 \\
    --sample-size 50 \\
    --skip-upload \\
    --dataset-name posts_clustered_kmeans_memtest
"
    )
    echo "  Job: $job_id"

    # Wait for job to finish
    local state
    while true; do
        state=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null || echo "UNKNOWN")
        case "$state" in
            COMPLETED|FAILED|OUT_OF_MEMORY|CANCELLED|TIMEOUT|UNKNOWN) break ;;
            *) sleep 5 ;;
        esac
    done

    echo "  State: $state"
    case "$state" in
        COMPLETED)
            local exit_code
            exit_code=$(sacct -j "$job_id" -X --noheader --format=ExitCode 2>/dev/null | tr -d ' ' | cut -d: -f1)
            if [ "${exit_code:-1}" -eq 0 ]; then
                echo "  => SUCCESS (exit 0)"
                return 0
            else
                echo "  => FAILED (exit $exit_code)"
                return 1
            fi
            ;;
        OUT_OF_MEMORY)
            echo "  => OOM"
            return 1
            ;;
        *)
            echo "  => FAILED / TIMEOUT / UNKNOWN"
            return 1
            ;;
    esac
}

# ------------------------------------------------------------------
echo "Binary search: min=${MIN_GB}G  max=${MAX_GB}G  tolerance=${TOLERANCE_GB}G"
echo

low=$MIN_GB
high=$MAX_GB
last_success=$MAX_GB  # highest GB known to succeed (start with max)

# First, verify the upper bound actually works
echo "### Verifying upper bound (${high}G) ###"
if run_trial "$high"; then
    echo "Upper bound ${high}G succeeded."
else
    echo "ERROR: Upper bound ${high}G failed! Increase MAX_GB."
    exit 1
fi

# Binary search
while [ $((high - low)) -gt "$TOLERANCE_GB" ]; do
    mid=$(( (low + high) / 2 ))
    echo
    echo "### Range [${low}G, ${high}G]  trying mid=${mid}G ###"

    if run_trial "$mid"; then
        echo "  ${mid}G succeeded -> narrowing upper bound"
        last_success=$mid
        high=$mid
    else
        echo "  ${mid}G failed -> raising lower bound"
        low=$((mid + 1))
    fi
done

echo
echo "=============================================="
echo "Minimum memory: ${last_success}G"
echo "Range:  [${low}G, ${high}G]  (tolerance=${TOLERANCE_GB}G)"
echo "=============================================="
echo "Logs: $LOGDIR/"
