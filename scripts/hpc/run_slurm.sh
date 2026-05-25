#!/usr/bin/env bash
# ============================================================================
# SLURM launcher for offline DBSCAN clustering.
#
# Usage on HPC login node (after "module load apptainer/1.4.5"):
#   bash run_slurm.sh /path/to/extracted/tarball
#
# This submits one array job per day, each running in an Apptainer container
# with local parquet data (read-only) and local output directory.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARBALL_DIR="${1:-$SCRIPT_DIR}"

DATA_DIR="${TARBALL_DIR}/data"
OUTPUT_BASE="${TARBALL_DIR}/output"
IMAGE="${TARBALL_DIR}/clustering_offline.sif"
LOG_DIR="${TARBALL_DIR}/logs"

# These match the original run_all_days.sh invocation
DATASET_NAME="${DATASET_NAME:-posts_clustered_dbscan_005_b}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-50}"
EPS="${EPS:-0.9}"
MIN_SAMPLES="${MIN_SAMPLES:-100}"

# SLURM resource parameters (from user request)
SLURM_MEM="${SLURM_MEM:-72GB}"
SLURM_CPUS="${SLURM_CPUS:-18}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-ctb-liyue}"

mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

# ---- discover days -----------------------------------------------------------
echo "=== Discovering days from local data ==="
DAYS=$(
    apptainer exec --no-home \
        --bind "${DATA_DIR}:/data:ro" \
        "$IMAGE" \
        python -c "
from clustering_offline import list_batches, group_batches_by_day
from pathlib import Path
batches = list_batches(Path('/data'))
days = group_batches_by_day(batches)
for d in sorted(days):
    print(d)
"
)

if [ -z "$DAYS" ]; then
    echo "ERROR: no days found in ${DATA_DIR}."
    exit 1
fi

readarray -t DAY_ARRAY <<< "$DAYS"
NUM_DAYS="${#DAY_ARRAY[@]}"
echo "Found ${NUM_DAYS} days: ${DAY_ARRAY[*]}"
echo ""

# ---- write the job script to a file (so sbatch can read it) ------------------
JOB_SCRIPT="${TARBALL_DIR}/_slurm_job_script.sh"

cat > "$JOB_SCRIPT" << 'INNER_SCRIPT'
#!/usr/bin/env bash
#SBATCH --mem=MEM_PLACEHOLDER
#SBATCH --cpus-per-task=CPUS_PLACEHOLDER
#SBATCH --account=ACCOUNT_PLACEHOLDER

set -euo pipefail

TARBALL_DIR="TARBALL_DIR_PLACEHOLDER"
DATA_DIR="${TARBALL_DIR}/data"
OUTPUT_BASE="${TARBALL_DIR}/output"
IMAGE="${TARBALL_DIR}/clustering_offline.sif"
DATASET_NAME="DATASET_NAME_PLACEHOLDER"
SAMPLE_SIZE="SAMPLE_SIZE_PLACEHOLDER"
MIN_CLUSTER_SIZE="MIN_CLUSTER_SIZE_PLACEHOLDER"
EPS="EPS_PLACEHOLDER"
MIN_SAMPLES="MIN_SAMPLES_PLACEHOLDER"

# DAY_ARRAY is indexed by SLURM_ARRAY_TASK_ID (1-indexed)
DAY_ARRAY=(DAY_ARRAY_PLACEHOLDER)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
DAY="${DAY_ARRAY[$IDX]}"

OUTPUT_DIR="${OUTPUT_BASE}/${DAY}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "  Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Day: ${DAY}"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Host: $(hostname)"
echo "============================================================"

module load apptainer/1.4.5

apptainer exec --no-home \
    --bind "${DATA_DIR}:/data:ro" \
    --bind "${OUTPUT_DIR}:/output" \
    "${IMAGE}" \
    python /opt/scripts/clustering_offline.py \
        --data-dir /data \
        --output-dir /output \
        --day "${DAY}" \
        --dataset-name "${DATASET_NAME}" \
        --sample-size "${SAMPLE_SIZE}" \
        --min-cluster-size "${MIN_CLUSTER_SIZE}" \
        --eps "${EPS}" \
        --min-samples "${MIN_SAMPLES}"

RC=$?
if [ $RC -eq 0 ]; then
    echo "OK: Day ${DAY} completed successfully."
    touch "${OUTPUT_DIR}/_SUCCESS"
else
    echo "ERROR: Day ${DAY} failed with exit code ${RC}"
    touch "${OUTPUT_DIR}/_FAILED"
    exit $RC
fi
INNER_SCRIPT

# Substitute placeholders with actual values
sed -i \
    -e "s|TARBALL_DIR_PLACEHOLDER|${TARBALL_DIR}|g" \
    -e "s|DATASET_NAME_PLACEHOLDER|${DATASET_NAME}|g" \
    -e "s|SAMPLE_SIZE_PLACEHOLDER|${SAMPLE_SIZE}|g" \
    -e "s|MIN_CLUSTER_SIZE_PLACEHOLDER|${MIN_CLUSTER_SIZE}|g" \
    -e "s|EPS_PLACEHOLDER|${EPS}|g" \
    -e "s|MIN_SAMPLES_PLACEHOLDER|${MIN_SAMPLES}|g" \
    -e "s|MEM_PLACEHOLDER|${SLURM_MEM}|g" \
    -e "s|CPUS_PLACEHOLDER|${SLURM_CPUS}|g" \
    -e "s|ACCOUNT_PLACEHOLDER|${SLURM_ACCOUNT}|g" \
    -e "s|DAY_ARRAY_PLACEHOLDER|${DAY_ARRAY[*]}|g" \
    "$JOB_SCRIPT"

# ---- submit array job --------------------------------------------------------
JOB_ID=$(sbatch \
    --parsable \
    --job-name="clustering" \
    --array="1-${NUM_DAYS}" \
    --output="${LOG_DIR}/clustering_%A_%a.out" \
    --error="${LOG_DIR}/clustering_%A_%a.err" \
    "$JOB_SCRIPT"
)

echo "Submitted array job ${JOB_ID} with ${NUM_DAYS} tasks."
echo ""
echo "Monitor with:  squeue -j ${JOB_ID}"
echo "Logs in:      ${LOG_DIR}/"
