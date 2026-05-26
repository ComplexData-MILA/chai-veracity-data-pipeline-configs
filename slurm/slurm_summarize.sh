#!/bin/bash
#SBATCH --job-name=cdl-summarize
#SBATCH --output=logs/summarize_%j.out
#SBATCH --error=logs/summarize_%j.err

#SBATCH -c 8
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=48GB
#SBATCH -t 3:00:00

set -e

export PROJECT_HOME=$HOME/20260331-chai-veracity
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
export MASTER_PORT=0

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
unset UV_VENVS_BASE VIRTUAL_ENV
source "$VENV_LOCAL/bin/activate"
export

export VLLM_PORT="${VLLM_PORT:-$((8000 + ${SLURM_JOB_ID: -3}))}"

echo "Job ID: $SLURM_JOB_ID | Port: $VLLM_PORT"

trap 'kill $(jobs -p) 2>/dev/null' EXIT

vllm serve $MODEL_NAME \
    --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 &

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

# Pass through optional overrides from environment
MAX_TOKENS="${MAX_TOKENS:-1024}"
DATASET_NAME="${DATASET_NAME:-posts_summarized_001}"
INPUT_DATASET="${INPUT_DATASET:-posts_clustered_kmeans_001}"
TEXT_COLUMN="${TEXT_COLUMN:-central_sample_texts}"
COPY_COLUMNS="${COPY_COLUMNS:-central_sample_ids}"
BATCH="${BATCH:-}"
MAX_CLUSTERS="${MAX_CLUSTERS:-}"

BATCH_ARG=""
[ -n "$BATCH" ] && BATCH_ARG="--batch $BATCH"
MAX_CLUSTERS_ARG=""
[ -n "$MAX_CLUSTERS" ] && MAX_CLUSTERS_ARG="--max-clusters $MAX_CLUSTERS"

uv run scripts/cluster/summarize.py \
    --model_name $MODEL_NAME \
    --max_concurrency 36 \
    --max_tokens $MAX_TOKENS \
    --dataset-name $DATASET_NAME \
    --input-dataset $INPUT_DATASET \
    --text-column $TEXT_COLUMN \
    --copy-columns $COPY_COLUMNS \
    $BATCH_ARG \
    $MAX_CLUSTERS_ARG

EXIT_CODE=$?

# Stop the vLLM server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE
