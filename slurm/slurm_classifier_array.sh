#!/bin/bash
#SBATCH --job-name=cdl-cls
#SBATCH --output=logs/classifier_%A_%a.out
#SBATCH --error=logs/classifier_%A_%a.err

#SBATCH -c 6
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=32GB
#SBATCH -t 3:00:00

set -e

# --- Configuration via environment variables (set at submit time) ---
export PROJECT_HOME=$HOME/20260331-chai-veracity
export MODEL_PATH="${MODEL_PATH:-$SCRATCH/classifier_models/default/best_model}"
export MODEL_NAME="${MODEL_NAME:-'custom-classifier'}"
export VLLM_PORT="${VLLM_PORT:-$((8000 + ${SLURM_ARRAY_TASK_ID:-0} * 100))}"

ANNOTATOR_NAME="${ANNOTATOR_NAME:-feasibility_classifier_001_full}"
FRACTION="${FRACTION:-1.0}"
MAX_BATCHES="${MAX_BATCHES:-20}"
BATCH="${BATCH:-}"

echo "=== Task $SLURM_ARRAY_TASK_ID ==="
echo "Job ID: $SLURM_ARRAY_JOB_ID / Task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL_PATH | Port: $VLLM_PORT"
echo "Annotator: $ANNOTATOR_NAME | Fraction: $FRACTION | Max Batches: $MAX_BATCHES"
[ -n "$BATCH" ] && echo "Specific batch: $BATCH"

mkdir -pv /tmp/$USER/torchinductor
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torchinductor

# Transfer venv tarball to local disk to avoid BeeGFS metadata cache races.
VENV_LOCAL="/tmp/$USER/uv-venv/vllm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Extracting venv tarball to $VENV_LOCAL ..."
mkdir -pv "$(dirname "$VENV_LOCAL")"
tar -xzf "$SCRATCH/uv-venv/vllm.tar.gz" -C "$(dirname "$VENV_LOCAL")"
EXTRACTED_DIR="$(dirname "$VENV_LOCAL")/$(tar -tzf "$SCRATCH/uv-venv/vllm.tar.gz" | head -1 | cut -d/ -f1)"
echo Extracting to $EXTRACTED_DIR
echo Using local venv copy at $VENV_LOCAL
[ "$EXTRACTED_DIR" != "$VENV_LOCAL" ] && mv "$EXTRACTED_DIR" "$VENV_LOCAL"
OLD_VENV=$(grep "^VIRTUAL_ENV=" "$VENV_LOCAL/bin/activate" | head -1 | sed "s/VIRTUAL_ENV=//;s/['\"]//g")
sed -i "s|$OLD_VENV|$VENV_LOCAL|g" "$VENV_LOCAL/bin/activate"
[ -f "$VENV_LOCAL/bin/activate.csh" ] && sed -i "s|$OLD_VENV|$VENV_LOCAL|g" "$VENV_LOCAL/bin/activate.csh"
[ -f "$VENV_LOCAL/bin/activate.fish" ] && sed -i "s|$OLD_VENV|$VENV_LOCAL|g" "$VENV_LOCAL/bin/activate.fish"
for f in "$VENV_LOCAL/bin/"*; do
    [ -f "$f" ] && [ ! -L "$f" ] && head -c2 "$f" | grep -q '#!' && sed -i "s|$OLD_VENV|$VENV_LOCAL|g" "$f"
done
unset UV_VENVS_BASE VIRTUAL_ENV
source "$VENV_LOCAL/bin/activate"
export

trap 'kill $(jobs -p) 2>/dev/null' EXIT

vllm serve $MODEL_PATH \
--port $VLLM_PORT \
--served-model-name $MODEL_NAME &

SERVER_PID=$!
echo SERVER_PID: $SERVER_PID
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

BATCH_ARGS=()
[ -n "$BATCH" ] && BATCH_ARGS=(--batch "$BATCH")

uv run -m scripts.filter.classifier \
    --model_name $MODEL_NAME \
    --base_url "http://127.0.0.1:${VLLM_PORT}" \
    --max_concurrency 128 \
    --batch_size 256 \
    --annotator_name "$ANNOTATOR_NAME" \
    --fraction "$FRACTION" \
    --max_batches "$MAX_BATCHES" \
    "${BATCH_ARGS[@]}"

EXIT_CODE=$?

kill $SERVER_PID 2>/dev/null
sleep 5
kill -9 $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true

echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $EXIT_CODE"
exit $EXIT_CODE
