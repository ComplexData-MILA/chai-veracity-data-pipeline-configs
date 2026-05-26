#!/bin/bash
#SBATCH --job-name=cdl-embedding
#SBATCH --output=logs/embedding_%j.out
#SBATCH --error=logs/embedding_%j.err

#SBATCH -c 8
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

# Transfer venv tarball to local disk to avoid BeeGFS metadata cache races.
VENV_LOCAL="/tmp/$USER/uv-venv/vllm_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Extracting venv tarball to $VENV_LOCAL ..."
mkdir -pv "$(dirname "$VENV_LOCAL")"
tar -xzf "$SCRATCH/uv-venv/vllm.tar.gz" -C "$(dirname "$VENV_LOCAL")"
# Rename the extracted directory if the tarball root differs from VENV_LOCAL
EXTRACTED_DIR="$(dirname "$VENV_LOCAL")/$(tar -tzf "$SCRATCH/uv-venv/vllm.tar.gz" | head -1 | cut -d/ -f1)"
echo Extracting to $EXTRACTED_DIR
echo Using local venv copy at $VENV_LOCAL
[ "$EXTRACTED_DIR" != "$VENV_LOCAL" ] && mv "$EXTRACTED_DIR" "$VENV_LOCAL"
# Fix hardcoded paths in the extracted venv
OLD_VENV=$(grep "^VIRTUAL_ENV=" "$VENV_LOCAL/bin/activate" | head -1 | sed "s/VIRTUAL_ENV=//;s/['\"]//g")
sed -i "s|$OLD_VENV|$VENV_LOCAL|g" "$VENV_LOCAL/bin/activate"
[ -f "$VENV_LOCAL/bin/activate.csh" ] && sed -i "s|$OLD_VENV|$VENV_LOCAL|g" "$VENV_LOCAL/bin/activate.csh"
[ -f "$VENV_LOCAL/bin/activate.fish" ] && sed -i "s|$OLD_VENV|$VENV_LOCAL|g" "$VENV_LOCAL/bin/activate.fish"
# Fix shebangs in bin scripts that reference the old venv
for f in "$VENV_LOCAL/bin/"*; do
    [ -f "$f" ] && [ ! -L "$f" ] && head -c2 "$f" | grep -q '#!' && sed -i "s|$OLD_VENV|$VENV_LOCAL|g" "$f"
done
source "$VENV_LOCAL/bin/activate"
unset UV_VENVS_BASE VIRTUAL_ENV

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
ANNOTATION_CMD="uv run -m scripts.preprocess.embed --model_name $MODEL_NAME --max_concurrency 36"
if [ -n "$EMBEDDING_DIMENSIONS" ]; then
    ANNOTATION_CMD="$ANNOTATION_CMD --dimensions $EMBEDDING_DIMENSIONS"
fi
# Only process the x-posts batch
ANNOTATION_CMD="$ANNOTATION_CMD --batch x-posts-20260507-06 --fraction 1.0"

echo "Starting embedding annotation..."
eval $ANNOTATION_CMD

EXIT_CODE=$?

echo "Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE
