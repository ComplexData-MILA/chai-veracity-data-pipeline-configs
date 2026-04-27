# Task-Specific Data Processing Scripts for CDL Chai-Veracity Data Pipeline

Folder structure:


```yaml
scripts/
    # Scripts for creating dataset
    - ingestion
    # Scripts for annotating dataset
    - annotation
```

## Getting Started

Store secrets in `.env`, based on `example.env`. 

To update to a more recent version of the s3-data-tool package, run:

```bash
uv lock --upgrade-package s3-data-tool
uv sync
```

For LLM-powered annotation, make sure the `sglang` server is installed in a virtual environment at `$SCRATCH/uv-venv/sglang/bin/activate`.

## Retrieve, Annotate, Export

Retrieve "base dataset"- these are initially stored as jsonl temp files in the S3.

```bash
uv run --env-file .env scripts/ingestion/bsky_trending.py
```

Merge these jsonl temp files of the base dataset into parquet.

```bash
uv run --env-file .env s3-data-tool-clean-up
```

Run annotation. 

```bash
# SLURM cluster and local SGLang instance
sbatch slurm/slurm_llm_judge.sh

# Alternatively, set OPENAI_BASE_URL and OPENAI_API_KEY
uv run scripts/filter/feasibility.py \
--model_name $MODEL_NAME \
--max_concurrency 36
```