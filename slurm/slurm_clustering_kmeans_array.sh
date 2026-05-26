#!/bin/bash
#SBATCH --job-name=kmeans
#SBATCH --output=logs/clustering_kmeans_%A_%a.out
#SBATCH --error=logs/clustering_kmeans_%A_%a.err

#SBATCH --array=0-22%2
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH -t 16:00:00

set -e

export PROJECT_HOME=$HOME/20260331-chai-veracity

# All dates from 20260427 through 20260519 (23 days)
DATES=(
    20260427 20260428 20260429 20260430
    20260501 20260502 20260503 20260504 20260505 20260506
    20260507 20260508 20260509 20260510 20260511 20260512
    20260513 20260514 20260515 20260516 20260517 20260518 20260519
)

DAY="${DATES[$SLURM_ARRAY_TASK_ID]}"

echo "Array Task ID: $SLURM_ARRAY_TASK_ID | Date: $DAY | CPUs: $SLURM_CPUS_PER_TASK | Mem: 32GB"

cd $PROJECT_HOME
source $PROJECT_HOME/.venv/bin/activate
source $PROJECT_HOME/.env

# Overridable defaults
DATASET_NAME="${DATASET_NAME:-posts_clustered_kmeans_001}"
N_CLUSTERS="${N_CLUSTERS:-100}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
COLLECT_LIMIT="${COLLECT_LIMIT:-0}"
BATCH="${BATCH:-}"
SKIP_UPLOAD="${SKIP_UPLOAD:-}"
OUTPUT_BATCH_NAME="${OUTPUT_BATCH_NAME:-}"

CMD="python scripts/cluster/clustering_kmeans.py"
CMD="$CMD --limit 1"
CMD="$CMD --dataset-name $DATASET_NAME"
CMD="$CMD --sample-size $SAMPLE_SIZE"
CMD="$CMD --collect-limit $COLLECT_LIMIT"
CMD="$CMD --n-clusters $N_CLUSTERS"
CMD="$CMD --day $DAY"
if [ -n "$BATCH" ]; then
    CMD="$CMD --batch $BATCH"
fi
if [ "$SKIP_UPLOAD" = "true" ]; then
    CMD="$CMD --skip-upload"
fi
if [ -n "$OUTPUT_BATCH_NAME" ]; then
    CMD="$CMD --output-batch-name $OUTPUT_BATCH_NAME"
fi

echo "Starting k-means clustering for $DAY..."
echo "Command: $CMD"
eval $CMD

EXIT_CODE=$?
echo "Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE
