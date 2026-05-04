#!/bin/bash
#SBATCH --job-name=cls-train
#SBATCH --output=logs/classifier_%j.out
#SBATCH --error=logs/classifier_%j.err

#SBATCH -c 8
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=48GB
#SBATCH -t 6:00:00

set -e

export PROJECT_HOME=$HOME/20260331-chai-veracity

mkdir -pv /tmp/$USER/torchinductor
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torchinductor

cd $PROJECT_HOME
source $PROJECT_HOME/.env

# Hyperparameters from environment with defaults
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-$SCRATCH/classifier_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/classifier_models/lr${LR}_bs${BATCH_SIZE}_seed${SEED}}"

echo "Job ID: $SLURM_JOB_ID"
echo "LR=$LR BS=$BATCH_SIZE EPOCHS=$EPOCHS WD=$WEIGHT_DECAY"
echo "Output: $OUTPUT_DIR"

# Copy shared venv to local disk to avoid BeeGFS metadata cache races.
VENV_LOCAL="/tmp/$USER/uv-venv/train_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Copying venv to $VENV_LOCAL ..."
mkdir -pv "$VENV_LOCAL"
cp -r "$SCRATCH/uv-venv/train/"* "$VENV_LOCAL"
source "$VENV_LOCAL/bin/activate"

python scripts/train_classifier/train.py \
    --data_dir "$DATA_DIR" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_length "$MAX_LENGTH" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
