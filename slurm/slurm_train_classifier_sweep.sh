#!/bin/bash
#SBATCH --job-name=cls-sweep
#SBATCH --output=logs/classifier_%A_%a.out
#SBATCH --error=logs/classifier_%A_%a.err

#SBATCH -c 8
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=48GB
#SBATCH -t 3:00:00

# --array must be set at submit time, e.g.:
#   N=$(wc -l < params.txt)
#   sbatch --array=1-$N slurm/slurm_train_classifier_sweep.sh

set -e

export PROJECT_HOME=$HOME/20260331-chai-veracity

mkdir -pv /tmp/$USER/torchinductor
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torchinductor

# Use task ID to pick the right param line
PARAMS_FILE="${PARAMS_FILE:-$PROJECT_HOME/params.txt}"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE")

if [ -z "$LINE" ]; then
    echo "ERROR: no params line for task $SLURM_ARRAY_TASK_ID in $PARAMS_FILE"
    exit 1
fi

# Source the params into environment variables
echo "Task $SLURM_ARRAY_TASK_ID params: $LINE"
for pair in $LINE; do
    export "$pair"
done

cd $PROJECT_HOME
source $PROJECT_HOME/.env

BASE_DIR=$SCRATCH/20260331-chai-veracity/
DATA_DIR="${DATA_DIR:-$BASE_DIR/classifier_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_DIR/classifier_models/lr${LR}_bs${BATCH_SIZE}_wd${WEIGHT_DECAY}_seed${SEED}}"

echo "Job ${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID} | LR=$LR BS=$BATCH_SIZE WD=$WEIGHT_DECAY SEED=$SEED"
echo "Output: $OUTPUT_DIR"

# Copy shared venv to local disk to avoid BeeGFS metadata cache races.
VENV_LOCAL="/tmp/$USER/uv-venv/train_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -pv "$VENV_LOCAL"
uv venv $VENV_LOCAL
source "$VENV_LOCAL/bin/activate"
uv pip install torch transformers scikit-learn tqdm

python3 scripts/train_classifier/train.py \
--data_dir "$DATA_DIR" \
--lr "$LR" \
--batch_size "$BATCH_SIZE" \
--epochs "$EPOCHS" \
--weight_decay "$WEIGHT_DECAY" \
--warmup_ratio "$WARMUP_RATIO" \
--max_length "$MAX_LENGTH" \
--seed "$SEED" \
--output_dir "$OUTPUT_DIR"
