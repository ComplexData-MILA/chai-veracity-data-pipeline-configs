#!/bin/bash
#SBATCH --job-name=cdl-cluster-simple
#SBATCH --output=logs/clustering_simplified_%j.out
#SBATCH --error=logs/clustering_simplified_%j.err

#SBATCH -c 8
#SBATCH --mem=128GB
#SBATCH -t 6:00:00

set -e

export PROJECT_HOME=$HOME/20260331-chai-veracity

echo "Job ID: $SLURM_JOB_ID | CPUs: $SLURM_CPUS_PER_TASK | Mem: 128GB"

cd $PROJECT_HOME
source $PROJECT_HOME/.venv/bin/activate
source $PROJECT_HOME/.env

# Limit OpenBLAS/OpenMP threads to avoid contention in FAISS
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"

# Pass through optional overrides from environment
LIMIT="${LIMIT:-1}"
DATASET_NAME="${DATASET_NAME:-posts_clustered_simplified_001}"
K="${K:-}"
DROP_FRAC="${DROP_FRAC:-0.95}"

CMD="python scripts/cluster/clustering_simplified.py"
CMD="$CMD --limit $LIMIT"
CMD="$CMD --dataset-name $DATASET_NAME"
CMD="$CMD --drop-frac $DROP_FRAC"
if [ -n "$K" ]; then
    CMD="$CMD --k $K"
fi

echo "Starting simplified clustering..."
echo "Command: $CMD"
eval $CMD

EXIT_CODE=$?
echo "Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE
