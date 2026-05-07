#!/bin/bash
# Master launcher: submits one SLURM job per dataset, plus a dependent merge job.
#
# Usage:
#   bash slurm/launch_compare_split.sh [n_samples] [n_judge_runs] [output_dir]
#
# Defaults: 100 samples, 3 judge runs, outputs/compare_dataset_quality_split

set -e

PROJECT_HOME="$(dirname "$(dirname "$(readlink -f "$0")")")"
N_SAMPLES="${1:-100}"
N_JUDGE_RUNS="${2:-3}"
OUTPUT_DIR="${3:-outputs/compare_dataset_quality_split}"
MERGE_OUTPUT_DIR="${OUTPUT_DIR}_merged"

# Dataset definitions: "name|path|subset|split"
# Use "-" for null subset/split
DATASETS=(
    "chai-veracity (clustered)|ComplexDataLab/chai-veracity-dry-run-20260506-clustered|-|-"
    "fever|ComplexDataLab/Misinfo_Datasets|fever|test"
    "liar|ComplexDataLab/Misinfo_Datasets|liar|test"
    "liar_new|ComplexDataLab/Misinfo_Datasets|liar_new|test"
)

echo "============================================"
echo "Launching split dataset comparison"
echo "  n_samples:    $N_SAMPLES"
echo "  n_judge_runs: $N_JUDGE_RUNS"
echo "  output_dir:   $OUTPUT_DIR"
echo "  merge_dir:    $MERGE_OUTPUT_DIR"
echo "============================================"

JOB_IDS=()

for ds in "${DATASETS[@]}"; do
    IFS='|' read -r DS_NAME DS_PATH DS_SUBSET DS_SPLIT <<< "$ds"

    ENV_ARGS=(
        "DATASET_NAME=$DS_NAME"
        "DATASET_PATH=$DS_PATH"
        "N_SAMPLES=$N_SAMPLES"
        "N_JUDGE_RUNS=$N_JUDGE_RUNS"
        "OUTPUT_DIR=$OUTPUT_DIR"
    )
    [ "$DS_SUBSET" != "-" ] && ENV_ARGS+=("DATASET_SUBSET=$DS_SUBSET")
    [ "$DS_SPLIT" != "-" ] && ENV_ARGS+=("DATASET_SPLIT=$DS_SPLIT")

    # Build the export list for sbatch
    EXPORT_STR=$(IFS=,; echo "${ENV_ARGS[*]}")

    JOB_ID=$(sbatch --export="$EXPORT_STR" \
        --job-name="cmp-${DS_NAME// /-}" \
        "$PROJECT_HOME/slurm/slurm_compare_single_dataset.sh" \
        | awk '{print $NF}')
    JOB_IDS+=($JOB_ID)
    echo "  Submitted $DS_NAME -> Job $JOB_ID"
done

# Submit merge job dependent on all dataset jobs
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")
MERGE_JOB_ID=$(sbatch \
    --job-name="cmp-merge" \
    --dependency="afterok:$DEP_STR" \
    --cpus-per-task=2 \
    --mem=8GB \
    --time=0:15:00 \
    --output="logs/compare_merge_%j.out" \
    --error="logs/compare_merge_%j.err" \
    --wrap="\
cd $PROJECT_HOME && \
uv run python scripts/analysis/merge_dataset_comparisons.py \
    --input-dir $OUTPUT_DIR \
    --output-dir $MERGE_OUTPUT_DIR" \
    | awk '{print $NF}')

echo "  Submitted merge -> Job $MERGE_JOB_ID (depends on: ${JOB_IDS[*]})"
echo ""
echo "All jobs submitted. Merge job ($MERGE_JOB_ID) will run after all dataset jobs complete."
echo "Check status: squeue -u \$USER"
echo "Results will be at: $MERGE_OUTPUT_DIR/"
