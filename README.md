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
VENV_BASE=$HOME/uv-venv/train
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

Collect results into a sorted table using:

```bash
uv run scripts/train_classifier/aggregate_sweep.py logs/classifier_${SLURM_JOB_ID}_*.out
```

Copy hyperparameters from the log file (a list of environment variables) and launch the train job with these parameters.

```bash
WEIGHT_DECAY=0.01 EPOCHS=10 WARMUP_RATIO=0.1 MAX_LENGTH=512 LR=5e-05 BATCH_SIZE=8 SEED=365 sbatch slurm/slurm_train_classifier.sh
```

## Evaluate Classifier on External Data

### 1. Prepare external evaluation data

Convert the external feasibility annotations into the classifier-compatible JSONL format, split by annotator. Reports Fleiss' Kappa inter-annotator agreement on claims where both annotators labeled.

```bash
uv run python scripts/analysis/prepare_external_eval_data.py
```

This writes `annotator_a.jsonl` and `annotator_b.jsonl` to `$SCRATCH/external_eval_data/`.

### 2. Evaluate the trained classifier

Launch a vLLM server with the trained classifier checkpoint and run evaluation against one or more test JSONL files. Reports per-file accuracy and Fleiss' Kappa; when multiple files are provided, also reports combined Fleiss' Kappa across the model and all annotators.

```bash
MODEL_PATH=$SCRATCH/20260331-chai-veracity/classifier_models/lr5e-05_bs8_wd0.01_seed365/best_model \
    sbatch slurm/eval/classfier_eval.sh
```

Override the test files via `TEST_FILES` (space-separated paths):

```bash
MODEL_PATH=$SCRATCH/20260331-chai-veracity/classifier_models/lr5e-05_bs8_wd0.01_seed365/best_model \
    TEST_FILES="$SCRATCH/external_eval_data/annotator_a.jsonl $SCRATCH/external_eval_data/annotator_b.jsonl" \
    sbatch slurm/eval/classfier_eval.sh
```

Or run directly against an already-running vLLM server:

```bash
uv run python scripts/analysis/evaluation_classifier.py \
    --test_files $SCRATCH/external_eval_data/annotator_a.jsonl \
                  $SCRATCH/external_eval_data/annotator_b.jsonl \
    --base_url http://127.0.0.1:8000 \
    --model_name custom-classifier
```

### 3. Evaluate the LLM judge

Run the LLM feasibility judge (from `scripts/filter/feasibility.py`) against the same test data. Reports the same metrics as the classifier evaluation.

**Via SLURM** (spins up a vLLM server with the judge model):

```bash
MODEL_NAME=Qwen/Qwen3.5-9B \
    sbatch slurm/eval/llm_judge_eval.sh
```

Override the test files via `TEST_FILES`:

```bash
MODEL_NAME=Qwen/Qwen3.5-9B \
    TEST_FILES="$SCRATCH/external_eval_data/annotator_a.jsonl $SCRATCH/external_eval_data/annotator_b.jsonl" \
    sbatch slurm/eval/llm_judge_eval.sh
```

**Direct invocation** (against an already-running OpenAI-compatible API):

```bash
uv run python scripts/analysis/evaluation_llm_judge.py \
    --test_files $SCRATCH/external_eval_data/annotator_a.jsonl \
                  $SCRATCH/external_eval_data/annotator_b.jsonl \
    --model_name gpt-4.1 \
    --base_url $OPENAI_BASE_URL \
    --api_key $OPENAI_API_KEY \
    --max_concurrency 32
```