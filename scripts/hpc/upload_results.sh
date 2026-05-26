#!/usr/bin/env bash
# ============================================================================
# Upload clustering results from local output directory back to S3.
#
# Usage on HPC login node (has internet access):
#   bash upload_results.sh /path/to/extracted/tarball [--dataset-name ...]
#
# Reads the same .env credentials used by the original pipeline.
# Only uploads days marked with _SUCCESS sentinel files.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARBALL_DIR="${1:-$SCRIPT_DIR}"

OUTPUT_BASE="${TARBALL_DIR}/output"
DATASET_NAME="${DATASET_NAME:-posts_clustered_dbscan_005_b}"

# ---- source credentials ------------------------------------------------------
if [ -f "${TARBALL_DIR}/.env" ]; then
    set -a
    source "${TARBALL_DIR}/.env"
    set +a
fi

# ---- find completed days -----------------------------------------------------
echo "=== Scanning for completed days in ${OUTPUT_BASE} ==="
COMPLETED=()
FAILED=()

for day_dir in "$OUTPUT_BASE"/*/; do
    day=$(basename "$day_dir")
    if [ -f "${day_dir}/_SUCCESS" ]; then
        COMPLETED+=("$day")
    elif [ -f "${day_dir}/_FAILED" ]; then
        FAILED+=("$day")
    fi
done

echo "Completed: ${#COMPLETED[@]} days"
echo "Failed:    ${#FAILED[@]} days"

if [ ${#COMPLETED[@]} -eq 0 ]; then
    echo "No completed days to upload."
    if [ ${#FAILED[@]} -gt 0 ]; then
        echo "Failed days: ${FAILED[*]}"
    fi
    exit 0
fi

echo ""
echo "Days to upload: ${COMPLETED[*]}"
echo ""

# ---- upload each day's JSONL files to S3 -------------------------------------
# We use the original project's venv and s3_data_tool from the tarball.
# The upload script is a Python one-liner that reuses the existing
# dataset_generator pattern.

for day in "${COMPLETED[@]}"; do
    echo "=== Uploading day ${day} ==="
    day_output="${OUTPUT_BASE}/${day}"
    batch_name="bsky-trending-${day}"

    # Use the project's Python to upload via aioboto3 directly.
    # We read each JSONL file and upload under the dataset/batch prefix.
    cd "$TARBALL_DIR"

    # Find all JSONL files for this day
    jsonl_files=("$day_output"/*.jsonl)
    if [ ${#jsonl_files[@]} -eq 0 ]; then
        echo "  WARNING: No JSONL files found in ${day_output}; skipping."
        continue
    fi

    echo "  Found ${#jsonl_files[@]} JSONL chunk(s)."

    # Upload using the Apptainer image (has aioboto3 + all deps)
    # Pass S3 credentials explicitly (--env-file doesn't handle 'export' syntax)
    apptainer exec --no-home \
        --bind "${day_output}:/upload:ro" \
        --bind "${TARBALL_DIR}/upload_single_day.py:/opt/scripts/upload_single_day.py:ro" \
        --env "S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-}" \
        --env "S3_BUCKET=${S3_BUCKET}" \
        --env "S3_PREFIX=${S3_PREFIX:-datasets}" \
        --env "S3_ACCESS_KEY=${S3_ACCESS_KEY}" \
        --env "S3_SECRET_KEY=${S3_SECRET_KEY}" \
        "${TARBALL_DIR}/clustering_offline.sif" \
        python /opt/scripts/upload_single_day.py \
            --output-dir /upload \
            --dataset-name "$DATASET_NAME" \
            --batch "$batch_name"
done

echo ""
echo "=== Upload complete ==="
echo "Uploaded: ${COMPLETED[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Not uploaded (failed): ${FAILED[*]}"
fi
