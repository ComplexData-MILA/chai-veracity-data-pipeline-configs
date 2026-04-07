#!/bin/bash
# Cron job for merge - runs every 30 minutes
# Merges jsonl temp files into parquet

set -e

cd "$(dirname "$0")/.."

LOG_FILE="logs/merge_$(date +\%Y\%m\%d_\%H\%M\%S).log"

echo "[$(date)] Starting merge job" >> "$LOG_FILE"

uv run --env-file .env s3-data-tool-clean-up >> "$LOG_FILE" 2>&1

echo "[$(date)] Merge job completed" >> "$LOG_FILE"
