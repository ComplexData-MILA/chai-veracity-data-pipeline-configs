#!/usr/bin/env bash
# ============================================================================
# Package everything needed for offline HPC clustering into a tarball.
#
# Usage (on machine with internet access):
#   bash package.sh [--output tarball.tar.gz]
#
# The resulting tarball contains:
#   clustering_offline.sif   - Apptainer image
#   data/                    - All parquet data (read-only)
#   run_slurm.sh             - Launcher script (submit array job)
#   upload_results.sh        - Upload results back to S3
#   upload_single_day.py     - Upload helper
#   .env                     - Credentials for S3 upload (OPTIONAL)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT="${1:-${REPO_ROOT}/hpc-clustering-offline.tar.gz}"

WORKDIR="$(mktemp -d)"
trap "rm -rf $WORKDIR" EXIT

echo "=== Packaging HPC offline clustering ==="

# Copy Apptainer image
echo "Copying Apptainer image..."
cp "${SCRIPT_DIR}/clustering_offline.sif" "${WORKDIR}/"

# Copy scripts
echo "Copying runner scripts..."
cp "${SCRIPT_DIR}/run_slurm.sh" "${WORKDIR}/"
cp "${SCRIPT_DIR}/upload_results.sh" "${WORKDIR}/"
cp "${SCRIPT_DIR}/upload_single_day.py" "${WORKDIR}/"

# Copy .env if it exists (user should review before transferring to HPC)
if [ -f "${REPO_ROOT}/.env" ]; then
    echo "Copying .env (review before transferring!)..."
    cp "${REPO_ROOT}/.env" "${WORKDIR}/"
fi

# Create README
cat > "${WORKDIR}/README.md" << 'EOF'
# Offline DBSCAN Clustering for HPC (SLURM + Apptainer)

## Contents

- `clustering_offline.sif` — Apptainer image with all Python dependencies
- `data/` — Parquet data (extracted from this tarball)
- `run_slurm.sh` — Submits array job (one per day)
- `upload_results.sh` — Uploads completed days back to S3
- `upload_single_day.py` — Python upload helper
- `.env` — S3 credentials (optional; provide on HPC if missing)

## Quickstart

1. Transfer the tarball to the HPC system:
   ```
   scp hpc-clustering-offline.tar.gz user@hpc-login:/path/to/work/
   ```

2. Extract on HPC:
   ```
   tar -xzf hpc-clustering-offline.tar.gz -C /path/to/work/
   cd /path/to/work/
   ```

3. (If needed) Create .env with S3 credentials for result upload:
   ```
   export S3_ENDPOINT_URL=...
   export S3_BUCKET=...
   export S3_PREFIX=datasets
   export S3_ACCESS_KEY=...
   export S3_SECRET_KEY=...
   ```

4. Run clustering:
   ```
   module load apptainer/1.4.5
   bash run_slurm.sh .
   ```

5. After all jobs complete, upload results:
   ```
   bash upload_results.sh .
   ```

## Parameters

Override defaults via environment variables:
```
DATASET_NAME=posts_clustered_dbscan_005_b \
SAMPLE_SIZE=50 \
MIN_CLUSTER_SIZE=50 \
EPS=0.9 \
MIN_SAMPLES=100 \
bash run_slurm.sh .
```

## SLURM resources

Default: `--mem 72GB -c 18 --account ctb-liyue`
Override: `SLURM_MEM=48GB SLURM_CPUS=8 bash run_slurm.sh .`
EOF

# Add data directory symlink note for tarball
echo "Data directory should be placed alongside the extracted contents as 'data/'."

echo ""
echo "=== Creating tarball: ${OUTPUT} ==="
tar -czf "${OUTPUT}" -C "${WORKDIR}" .

SIZE=$(du -h "${OUTPUT}" | cut -f1)
echo "Done. Tarball: ${OUTPUT} (${SIZE})"
