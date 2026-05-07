#!/bin/bash
#SBATCH --job-name=cdl-cmp-single
#SBATCH --output=logs/compare_single_%j.out
#SBATCH --error=logs/compare_single_%j.err

#SBATCH -c 8
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=48GB
#SBATCH -t 3:00:00

# Generic single-dataset comparison job.
# Set these env vars when submitting:
#   DATASET_NAME         Display name (required)
#   DATASET_PATH         HF Hub path (required)
#   DATASET_SUBSET       Optional subset/config
#   DATASET_SPLIT        Optional split
#   N_SAMPLES            Examples to subsample (default: 100)
#   N_JUDGE_RUNS         Judge runs per example (default: 5)
#   OUTPUT_DIR           Parent output directory (default: outputs/compare_dataset_quality_split)

set -e

export PROJECT_HOME=$HOME/20260331-chai-veracity
export MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3.5-9B}
export N_SAMPLES=${N_SAMPLES:-100}
export N_JUDGE_RUNS=${N_JUDGE_RUNS:-5}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/compare_dataset_quality_split}

if [ -z "${DATASET_NAME:-}" ] || [ -z "${DATASET_PATH:-}" ]; then
    echo "ERROR: DATASET_NAME and DATASET_PATH must be set"
    exit 1
fi

mkdir -pv /tmp/$USER/torchinductor
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torchinductor

# Transfer venv tarball to local disk
VENV_LOCAL="/tmp/$USER/uv-venv/vllm_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Extracting venv tarball to $VENV_LOCAL ..."
mkdir -pv "$(dirname "$VENV_LOCAL")"
tar -xzf "$SCRATCH/uv-venv/vllm.tar.gz" -C "$(dirname "$VENV_LOCAL")"
EXTRACTED_DIR="$(dirname "$VENV_LOCAL")/$(tar -tzf "$SCRATCH/uv-venv/vllm.tar.gz" | head -1 | cut -d/ -f1)"
echo Extracting to $EXTRACTED_DIR
echo Using local venv copy at $VENV_LOCAL
[ "$EXTRACTED_DIR" != "$VENV_LOCAL" ] && mv "$EXTRACTED_DIR" "$VENV_LOCAL"
# Fix hardcoded paths in the extracted venv
OLD_VENV=$(grep "^VIRTUAL_ENV=" "$VENV_LOCAL/bin/activate" | head -1 | sed "s/VIRTUAL_ENV=//;s/['\"]//g")
sed -i "s|$OLD_VENV|$VENV_LOCAL|g" "$VENV_LOCAL/bin/activate"
[ -f "$VENV_LOCAL/bin/activate.csh" ] && sed -i "s|$OLD_VENV|$VENV_LOCAL|g" "$VENV_LOCAL/bin/activate.csh"
[ -f "$VENV_LOCAL/bin/activate.fish" ] && sed -i "s|$OLD_VENV|$VENV_LOCAL|g" "$VENV_LOCAL/bin/activate.fish"
for f in "$VENV_LOCAL/bin/"*; do
    [ -f "$f" ] && [ ! -L "$f" ] && head -c2 "$f" | grep -q '#!' && sed -i "s|$OLD_VENV|$VENV_LOCAL|g" "$f"
done
unset UV_VENVS_BASE VIRTUAL_ENV
source "$VENV_LOCAL/bin/activate"

# Multiply by 5 to leave port gaps: 3 vLLM instances on one node each need
# an API port + an NCCL internal port (auto-incremented by 1). A gap of 5
# prevents the EngineCore from stepping on the next job's API port.
export VLLM_PORT="${VLLM_PORT:-$((8000 + 10#${SLURM_JOB_ID: -3} * 5))}"
echo "Job ID: $SLURM_JOB_ID | Port: $VLLM_PORT | Dataset: $DATASET_NAME"

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

# Build extra args for optional subset/split
EXTRA_ARGS=()
[ -n "${DATASET_SUBSET:-}" ] && EXTRA_ARGS+=(--dataset-subset "$DATASET_SUBSET")
[ -n "${DATASET_SPLIT:-}" ] && EXTRA_ARGS+=(--dataset-split "$DATASET_SPLIT")

uv run scripts/analysis/compare_dataset_quality.py \
    --model-name $MODEL_NAME \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    "${EXTRA_ARGS[@]}" \
    --n-samples $N_SAMPLES \
    --n-judge-runs $N_JUDGE_RUNS \
    --max-concurrency 128 \
    --output-dir "$OUTPUT_DIR"

# Stop the vLLM server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
