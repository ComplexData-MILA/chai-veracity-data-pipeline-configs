#!/bin/bash
#SBATCH --job-name=cdl-kmeans
#SBATCH --output=logs/clustering_kmeans_%j.out
#SBATCH --error=logs/clustering_kmeans_%j.err

#SBATCH -c 8
#SBATCH --mem=128GB
#SBATCH -t 4:00:00

set -e

export PROJECT_HOME=$HOME/20260331-chai-veracity

echo "Job ID: $SLURM_JOB_ID | CPUs: $SLURM_CPUS_PER_TASK | Mem: 128GB | Time: 4h"

cd $PROJECT_HOME
source $PROJECT_HOME/.venv/bin/activate
source $PROJECT_HOME/.env

# Overridable defaults
LIMIT="${LIMIT:-1}"
DATASET_NAME="${DATASET_NAME:-posts_clustered_kmeans_001}"
N_CLUSTERS="${N_CLUSTERS:-}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
COLLECT_LIMIT="${COLLECT_LIMIT:-0}"
BATCH="${BATCH:-}"

CMD="python scripts/cluster/clustering_kmeans.py"
CMD="$CMD --limit $LIMIT"
CMD="$CMD --dataset-name $DATASET_NAME"
CMD="$CMD --sample-size $SAMPLE_SIZE"
CMD="$CMD --collect-limit $COLLECT_LIMIT"
if [ -n "$N_CLUSTERS" ]; then
    CMD="$CMD --n-clusters $N_CLUSTERS"
fi
if [ -n "$BATCH" ]; then
    CMD="$CMD --batch $BATCH"
fi

echo "Starting k-means clustering..."
echo "Command: $CMD"
eval $CMD

EXIT_CODE=$?
echo "Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE
