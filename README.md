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

## Distill Annotations into Classifier

Set up the training virtual environment (one-time):

```bash
VENV_BASE=$SCRATCH/uv-venv/train
source $VENV_BASE/bin/activate
uv pip install torch transformers scikit-learn tqdm
```

Prepare data — fetch labeled posts from S3, deduplicate, split, and cache locally:

```bash
uv run --active scripts/train_classifier/prepare_data.py --classes=0,1,2
```

This saves `train.jsonl`, `val.jsonl`, `test.jsonl` under `$SCRATCH/classifier_data` (override with `--output_dir`).

Run a single training:

```bash
source $VENV_BASE/bin/activate
python scripts/train_classifier/train.py
```

All hyperparameters are CLI args (`--lr`, `--batch_size`, `--epochs`, `--weight_decay`, `--warmup_ratio`, `--max_length`, `--seed`). Defaults train on `Qwen/Qwen3-0.6B-Base`.

To run a hyperparameter sweep via SLURM job array:

```bash
# 1. Generate the parameter file
python scripts/train_classifier/generate_sweep_params.py \
    --lr "1e-5,5e-5,1e-4" \
    --batch_size "8,16,32" \
    --seed "18,365,613" \
    --epochs 10 \
    --output params.txt

# 2. Submit the array (one job per line in params.txt)
N=$(wc -l < params.txt) && sbatch --array=1-$N slurm/slurm_train_classifier_sweep.sh
```

Each array task sources its line from `params.txt` and saves its best checkpoint under `$SCRATCH/20260331-chai-veracity/classifier_models/`.

