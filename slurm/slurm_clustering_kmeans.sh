#!/bin/bash
#SBATCH --job-name=cdl-kmeans
#SBATCH --output=logs/clustering_kmeans_%j.out
#SBATCH --error=logs/clustering_kmeans_%j.err

#SBATCH -c 2
#SBATCH --mem=8GB
#SBATCH -t 4:00:00

set -e

export PROJECT_HOME=$HOME/20260331-chai-veracity

echo "Job ID: $SLURM_JOB_ID | CPUs: $SLURM_CPUS_PER_TASK | Mem: 32GB | Time: 2h"

cd $PROJECT_HOME
source $PROJECT_HOME/.venv/bin/activate
source $PROJECT_HOME/.env

# Overridable defaults
LIMIT="${LIMIT:-1}"
DATASET_NAME="${DATASET_NAME:-posts_clustered_kmeans_001}"
N_CLUSTERS="${N_CLUSTERS:-100}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
COLLECT_LIMIT="${COLLECT_LIMIT:-0}"
DAY="${DAY:-}"
BATCH="${BATCH:-}"
SKIP_UPLOAD="${SKIP_UPLOAD:-}"
OUTPUT_BATCH_NAME="${OUTPUT_BATCH_NAME:-}"

CMD="python scripts/cluster/clustering_kmeans.py"
CMD="$CMD --limit $LIMIT"
CMD="$CMD --dataset-name $DATASET_NAME"
CMD="$CMD --sample-size $SAMPLE_SIZE"
CMD="$CMD --collect-limit $COLLECT_LIMIT"
if [ -n "$N_CLUSTERS" ]; then
    CMD="$CMD --n-clusters $N_CLUSTERS"
fi
if [ -n "$DAY" ]; then
    CMD="$CMD --day $DAY"
fi
if [ -n "$BATCH" ]; then
    CMD="$CMD --batch $BATCH"
fi
if [ "$SKIP_UPLOAD" = "true" ]; then
    CMD="$CMD --skip-upload"
fi
if [ -n "$OUTPUT_BATCH_NAME" ]; then
    CMD="$CMD --output-batch-name $OUTPUT_BATCH_NAME"
fi

echo "Starting k-means clustering..."
echo "Command: $CMD"
eval $CMD

EXIT_CODE=$?
echo "Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE
