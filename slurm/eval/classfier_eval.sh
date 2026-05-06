#!/bin/bash
#SBATCH --job-name=cdl-classifier-eval
#SBATCH --output=logs/classifier_eval_%j.out
#SBATCH --error=logs/classifier_eval_%j.err

#SBATCH -c 6
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=32GB
#SBATCH -t 3:00:00

set -e

export PROJECT_HOME=$HOME/20260331-chai-veracity
export MODEL_PATH="${MODEL_PATH:-$SCRATCH/classifier_models/default/best_model}"
export MODEL_NAME="${MODEL_NAME:-custom-classifier}"
export VLLM_PORT="${VLLM_PORT:-$((8000 + ${SLURM_JOB_ID: -3}))}"

# Default test files: both annotator evaluation sets produced by prepare_external_eval_data.py
export TEST_FILES="${TEST_FILES:-$SCRATCH/external_eval_data/annotator_a.jsonl $SCRATCH/external_eval_data/annotator_b.jsonl}"

echo "Job ID: $SLURM_JOB_ID | Model: $MODEL_PATH | Port: $VLLM_PORT"
echo "Test files: $TEST_FILES"

mkdir -pv /tmp/$USER/torchinductor
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torchinductor

# Copy shared venv to local disk to avoid BeeGFS metadata cache races.
VENV_LOCAL="/tmp/$USER/uv-venv/vllm-20260503_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Copying venv to $VENV_LOCAL ..."
mkdir -pv "$VENV_LOCAL"
cp -r $SCRATCH/uv-venv/vllm-20260503/* "$VENV_LOCAL"
source "$VENV_LOCAL/bin/activate"

trap 'kill $(jobs -p) 2>/dev/null' EXIT

vllm serve $MODEL_PATH \
    --port $VLLM_PORT \
    --served-model-name $MODEL_NAME &

SERVER_PID=$!
echo "SERVER_PID: $SERVER_PID"
echo "Waiting for vLLM server to start..."
max_retries=180
retry_count=0
while ! curl -s "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null; do
    if [ $retry_count -ge $max_retries ]; then
        echo "Server failed to start within timeout."
        exit 1
    fi
    sleep 5
    ((++retry_count))
    echo "Retrying... ($retry_count/$max_retries)"
done
echo "Server is healthy."

deactivate

cd $PROJECT_HOME
source $PROJECT_HOME/.env

uv run python scripts/analysis/evaluation_classifier.py \
    --test_files $TEST_FILES \
    --base_url "http://127.0.0.1:${VLLM_PORT}" \
    --model_name $MODEL_NAME \
    --batch_size 256

EXIT_CODE=$?

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE
