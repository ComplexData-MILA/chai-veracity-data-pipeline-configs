#!/bin/bash
#SBATCH --job-name=cdl-annotation
#SBATCH --output=logs/feasibility_%j.out
#SBATCH --error=logs/feasibility_%j.err

#SBATCH -c 8
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=48GB
#SBATCH -t 3:00:00

set -e

export PROJECT_HOME=$HOME/20260331-chai-veracity
export MODEL_NAME=Qwen/Qwen3.5-9B

mkdir -pv /tmp/$USER/torchinductor
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torchinductor

source $SCRATCH/uv-venv/sglang/bin/activate

export TEMPLATE_NAME="${TEMPLATE_NAME:-binary_no_search}"
export SGLANG_PORT="${SGLANG_PORT:-8000}"

echo "Job ID: $SLURM_JOB_ID | Template: $TEMPLATE_NAME | Port: $SGLANG_PORT"

trap 'kill $(jobs -p) 2>/dev/null' EXIT

$VIRTUAL_ENV/bin/python -m sglang.launch_server \
    --model-path $MODEL_NAME \
    --port $SGLANG_PORT \
    --tp-size 1 \
    --mem-fraction-static 0.8 \
    --context-length 262144 \
    --reasoning-parser qwen3 \
    --speculative-algo NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 &

SERVER_PID=$!
echo SERVER_PID: $SERVER_PID
echo "Waiting for SGLang server to start..."
max_retries=180
retry_count=0
while ! curl -s "http://127.0.0.1:${SGLANG_PORT}/health" > /dev/null; do
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

export OPENAI_BASE_URL=http://127.0.0.1:${SGLANG_PORT}/v1
export OPENAI_API_KEY="EMPTY"

# Launch long-running scripts in parallel (add more as needed)
BG_PIDS=()

uv run scripts/filter/feasibility.py \
    --model_name $MODEL_NAME \
    --max_concurrency 36 &
BG_PIDS+=($!)

uv run scripts/preprocess/extract_keywords.py \
    --model_name $MODEL_NAME \
    --max_concurrency 36 &
BG_PIDS+=($!)

# Wait for all non-server background jobs
FAILED=0
for pid in "${BG_PIDS[@]}"; do
    wait "$pid" || FAILED=1
done

# Stop the sglang server now that all scripts are done
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

exit $FAILED
