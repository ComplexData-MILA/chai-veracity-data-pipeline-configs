#!/bin/bash
# Cron job for retrieve - runs every 30 minutes
# Retrieves base dataset from S3 as jsonl temp files

set -e

cd "$(dirname "$0")/.."

LOG_FILE="logs/retrieve_$(date +\%Y\%m\%d_\%H\%M\%S).log"

echo "[$(date)] Starting retrieve job" >> "$LOG_FILE"

uv run --env-file .env scripts/ingestion/bsky_trending.py >> "$LOG_FILE" 2>&1

echo "[$(date)] Retrieve job completed" >> "$LOG_FILE"
