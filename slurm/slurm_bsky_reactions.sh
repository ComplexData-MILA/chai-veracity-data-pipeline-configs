#!/bin/bash
#SBATCH --job-name=bsky-reactions
#SBATCH --output=logs/bsky_reactions_%j.out
#SBATCH --error=logs/bsky_reactions_%j.err

#SBATCH -c 1
#SBATCH --mem=4GB
#SBATCH -t 24:00:00

set -e

export PROJECT_HOME=$HOME/20260331-chai-veracity
cd $PROJECT_HOME

mkdir -p logs

echo "Starting Bluesky reactions ingestion at $(date)"
echo "Job ID: $SLURM_JOB_ID"

source $PROJECT_HOME/.env
uv run --env-file .env python scripts/ingestion/bsky_reactions.py --dataset-name reactions

EXIT_CODE=$?
echo "Job finished with exit code $EXIT_CODE at $(date)"
exit $EXIT_CODE
