#!/bin/bash
#SBATCH --job-name=cdl-embedding
#SBATCH --output=logs/embedding_%j.out
#SBATCH --error=logs/embedding_%j.err

#SBATCH -c 8
#SBATCH -p main
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=48GB
#SBATCH -t 3:00:00

set -e

export PROJECT_HOME=$HOME/20260331-chai-veracity
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
export EMBEDDING_DIMENSIONS="${EMBEDDING_DIMENSIONS:128}"
export VLLM_PORT="${VLLM_PORT:-$((8000 + ${SLURM_JOB_ID: -3}))}"

echo "Job ID: $SLURM_JOB_ID | Model: $MODEL_NAME | Port: $VLLM_PORT"

mkdir -pv /tmp/$USER/torchinductor
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torchinductor

# Copy shared venv to local disk to avoid BeeGFS metadata cache races.
VENV_LOCAL="/tmp/$USER/uv-venv/vllm_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Copying venv to $VENV_LOCAL ..."
mkdir -pv "$VENV_LOCAL"
cp -r "$SCRATCH/uv-venv/vllm/"* "$VENV_LOCAL"
source "$VENV_LOCAL/bin/activate"

# Build server launch command
SERVER_CMD="$VENV_LOCAL/bin/vllm serve"
SERVER_CMD="$SERVER_CMD $MODEL_NAME"
SERVER_CMD="$SERVER_CMD --port $VLLM_PORT"

# Configure Matryoshka dimensions if specified
echo "Using Matryoshka dimensions: $EMBEDDING_DIMENSIONS"
SERVER_CMD="$SERVER_CMD --hf-overrides '{\"is_matryoshka\": true}'"

echo "Launching vLLM embedding server..."
echo "Command: $SERVER_CMD"

trap 'kill $(jobs -p) 2>/dev/null' EXIT

eval $SERVER_CMD &
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

export OPENAI_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1
export OPENAI_API_KEY="EMPTY"

# Build annotation command
ANNOTATION_CMD="uv run scripts/preprocess/embed.py --model_name $MODEL_NAME --max_concurrency 36"
if [ -n "$EMBEDDING_DIMENSIONS" ]; then
    ANNOTATION_CMD="$ANNOTATION_CMD --dimensions $EMBEDDING_DIMENSIONS"
fi

echo "Starting embedding annotation..."
eval $ANNOTATION_CMD

EXIT_CODE=$?

echo "Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE
