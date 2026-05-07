#!/bin/bash
cd /mnt/chai-veracity-data-pipeline-configs
/home/ubuntu/.local/bin/uv run --env-file .env s3-data-tool-clean-up
