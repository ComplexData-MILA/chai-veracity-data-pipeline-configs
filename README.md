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

## Cluster Posts by Embedding Similarity

Cluster posts one day at a time using a kNN-graph + connected-components approach. Each day's posts are loaded from S3 (base dataset joined with 128-d embeddings), a sparse kNN graph is built over cosine similarity, and connected components become clusters. Results are uploaded back to S3 as a new dataset.

```bash
uv run --env-file .env python scripts/cluster/clustering.py \
--limit 1 \
--drop-frac 0.9 \
--min-cluster-size 2 \
--outlier-threshold 0.1 \
--dataset-name posts_clustered_001
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--limit` | `1` | Max days to process (`0` = all days). Only days with embeddings count. |
| `--k` | auto (`log2(n)`) | kNN neighbours per node. Lower = sparser graph = more clusters. |
| `--drop-frac` | `0.95` | Fraction of farthest neighbor edges to prune per node. Higher = sparser graph = more clusters. |
| `--dataset-name` | `posts_clustered_001` | Target dataset name in S3. |
| `--min-cluster-size` | `0` | Merge clusters smaller than this into nearest larger cluster (`0` = disabled). |
| `--max-cluster-size` | `0` | Split clusters larger than this by re-clustering (`0` = disabled). |
| `--split-k-scale` | `0.5` | Multiplier for `k` when sub-clustering oversized components. |
| `--outlier-threshold` | `0.0` | Cosine similarity floor. Small clusters below this vs all large clusters are dropped as outliers (`0.0` = disabled). Only active with `--min-cluster-size`. |

The default `--drop-frac 0.95` keeps only the single closest neighbor per node (with `k=14`), producing many small topic-specific clusters. Use `--drop-frac 0.9` (keep ~2 edges) for fewer, larger clusters, or `--drop-frac 0.8` (keep ~3 edges) for even coarser granularity.

To balance cluster sizes, combine `--min-cluster-size` and `--max-cluster-size`:
```bash
uv run --env-file .env python scripts/cluster/clustering.py \
--limit 1 --k 14 --drop-frac 0.95 \
--min-cluster-size 5 --max-cluster-size 100 \
--outlier-threshold 0.3
```
This merges clusters smaller than 5 posts, splits clusters larger than 100 posts (using k-means fallback when kNN can't find boundaries), and drops clusters whose embedding centroid is less than 0.3 cosine-similar to any larger cluster.

Each day's clusters are uploaded as a separate batch named `bsky-trending-{YYYYMMDD}`.

### Tune Clustering Hyperparameters with LLM-as-a-Judge

Sweep `drop_frac` × `min_cluster_size` × `outlier_threshold` and evaluate cluster topic coherence via LLM:

```bash
uv run --env-file .env --env-file .llm.env python scripts/analysis/tune_clustering.py \
--drop-frac "0.5,0.8,0.9" \
--min-cluster-size "2,5,10" \
--outlier-threshold "0.1,0.2,0.5" \
--sample-size 10 --n-judge-runs 5 \
--max-concurrency 32 \
--limit 1 \
--output-dir outputs/tune_clustering
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--drop-frac` | `0.1,0.5,0.8,0.9,0.95,0.99` | Comma-separated drop_frac values to sweep |
| `--min-cluster-size` | `2,3,5,10` | Comma-separated min_cluster_size values to sweep |
| `--sample-size` | `20` | Clusters to sample per run for LLM evaluation |
| `--n-judge-runs` | `1` | Independent sampling+judging runs per combo; `>=2` enables 95% t-interval CI columns (SEM, CI low/high) and `±SEM` annotations on the coherence heatmap |
| `--model-name` | env `MODEL_NAME` | OpenAI-compatible model for judging cluster coherence |
| `--max-concurrency` | `16` | Max concurrent LLM API calls |
| `--outlier-threshold` | `0.0` | Comma-separated outlier_threshold values to sweep |
| `--limit` | `1` | Days to process |
| `--k` | auto (`log2(n)`) | kNN neighbours |
| `--max-cluster-size` | `50` | Max cluster size before splitting |
| `--seed` | `42` | Random seed for cluster sampling |

Results are saved to `outputs/tune_clustering/results.csv` (pandas table), `results.json` (full diagnostics), and per-threshold heatmaps (`coherence_heatmap_thr000.png`, `cluster_count_heatmap_thr000.png`, etc.).

## Summarize Clusters with LLM

Extract common verifiable claims from each cluster using an LLM. The script reads clustered posts from S3 (`posts_clustered_002`), sends batches to the LLM, and writes summarized claims back to S3.

```bash
# Via SLURM (launches a local vLLM server with Qwen3.5-9B)
sbatch slurm/slurm_summarize.sh

# With optional overrides
MAX_TOKENS=2048 DATASET_NAME=posts_summarized_003 sbatch slurm/slurm_summarize.sh

# Central
sbatch --export=ALL,INPUT_DATASET=posts_clustered_dbscan_002,TEXT_COLUMN=central_sample_texts,COPY_COLUMNS=central_sample_ids,DATASET_NAME=posts_summarized_004_a_dry_run_central,BATCH=bsky-trending-20260503 slurm/slurm_summarize.sh

# Diverse
sbatch --export=ALL,INPUT_DATASET=posts_clustered_dbscan_002,TEXT_COLUMN=diverse_sample_texts,COPY_COLUMNS=diverse_sample_ids,DATASET_NAME=posts_summarized_004_a_dry_run_diverse,BATCH=bsky-trending-20260503 slurm/slurm_summarize.sh

Full runs (all batches, drop BATCH and the _dry_run suffix):

# Central
sbatch --export=ALL,INPUT_DATASET=posts_clustered_dbscan_002,TEXT_COLUMN=central_sample_texts,COPY_COLUMNS=central_sample_ids,DATASET_NAME=posts_summarized_004_central slurm/slurm_summarize.sh

# Diverse
sbatch --export=ALL,INPUT_DATASET=posts_clustered_dbscan_002,TEXT_COLUMN=diverse_sample_texts,COPY_COLUMNS=diverse_sample_ids,DATASET_NAME=posts_summarized_004_diverse slurm/slurm_summarize.sh
```

**Via external API** (OpenAI-compatible endpoint):

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY="EMPTY"
export MODEL_NAME=Qwen/Qwen3.5-9B

uv run scripts/cluster/summarize.py \
    --model_name $MODEL_NAME \
    --max_concurrency 36 \
    --max_tokens 1024 \
    --dataset-name posts_summarized_001
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | (required) | OpenAI-compatible model name |
| `--max_concurrency` | `16` | Max concurrent LLM API calls |
| `--max_retries` | `6` | Retries per cluster on API failure |
| `--max_tokens` | `1024` | Max completion tokens per LLM call; caps output to prevent loops |
| `--dataset-name` | `posts_summarized_001_dry_run` | Target dataset name in S3 |

**Test prompt template locally** (uses a real S3 sample, prints LLM output without writing back):

```bash
uv run python scripts/cluster/test_summarize.py
```

Uses secrets from `.env` and `.llm-test.env`.

## Double-Clustering Pipeline

Second-level clustering that groups extracted claims into higher-level themes ("meta-clusters"), then synthesizes claims within each meta-cluster. The pipeline reuses the same kNN-graph clustering and LLM summarization infrastructure as the first-level pipeline.

### Data Flow

```
posts_summarized_003_b        (claims from first-level summarization)
  -> embed_claims.py          (embed claim texts)
posts_claims_embedded_001
  -> cluster_claims.py        (cluster claims by embedding similarity)
posts_clustered_meta_001
  -> summarize.py             (synthesize higher-level claims from meta-clusters)
posts_double_summarized_001
```

### Embed Claims

Reads claims from `posts_summarized_003_b` and embeds them using an OpenAI-compatible embedding API.

```bash
# Via SLURM (starts vLLM embedding server with Qwen3-Embedding-0.6B)
sbatch slurm/slurm_embed_claims.sh

# With overrides
EMBEDDING_DIMENSIONS=256 MODEL_NAME=Qwen/Qwen3-Embedding-0.6B sbatch slurm/slurm_embed_claims.sh

# Direct invocation (against an already-running embedding API)
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY="EMPTY"
uv run --env-file .env python scripts/cluster/embed_claims.py \
    --model_name Qwen/Qwen3-Embedding-0.6B \
    --dimensions 128 \
    --batch-size 256 \
    --max-concurrency 36
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | env `MODEL_NAME` | Embedding model name |
| `--dimensions` | `128` | Matryoshka embedding dimensions |
| `--batch-size` | `256` | Items per API call |
| `--source-dataset` | `posts_summarized_003_b` | Source dataset with claims |
| `--target-dataset` | `posts_claims_embedded_001` | Destination for embedded claims |
| `--max-concurrency` | `36` | Max concurrent API calls |
| `--max-claims` | `0` (all) | Limit number of claims to embed |
| `--source-filter` | `post_count >= 2` | DuckDB filter for source claims |

### Cluster Claims

Clusters embedded claims using the kNN-graph + connected-components algorithm (same as first-level clustering).

```bash
# Default: auto-k, drop_frac=0.9, min_cluster_size=2
uv run --env-file .env python scripts/cluster/cluster_claims.py

# Aim for ~10 meta-clusters with high coherence
uv run --env-file .env python scripts/cluster/cluster_claims.py \
    --drop-frac 0.8 --min-cluster-size 3 --outlier-threshold 0.1

# Run with diagnostics to inspect cluster sizes and small-cluster similarities
uv run --env-file .env python scripts/cluster/cluster_claims.py \
    --diagnostics --drop-frac 0.9 --min-cluster-size 2
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--k` | auto (`log2(n)`) | kNN neighbours per node |
| `--drop-frac` | `0.9` | Fraction of farthest edges to prune |
| `--min-cluster-size` | `2` | Merge clusters smaller than this |
| `--max-cluster-size` | `0` (disabled) | Split clusters larger than this |
| `--split-k-scale` | `0.5` | Scale factor for k when sub-clustering |
| `--outlier-threshold` | `0.0` (disabled) | Cosine similarity floor for outlier detection |
| `--source-dataset` | `posts_claims_embedded_001` | Source dataset with embedded claims |
| `--target-dataset` | `posts_clustered_meta_001` | Destination for meta-clusters |
| `--diagnostics` | `false` | Print diagnostic info without uploading |

### Summarize Meta-Clusters

Summarizes meta-clusters using LLM to synthesize higher-level claims. Uses the existing `summarize.py` script with a specialized template.

```bash
# Via SLURM (starts vLLM chat server with Qwen3.5-9B)
sbatch slurm/slurm_summarize_meta_clusters.sh

# Direct invocation (against an already-running API)
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY="EMPTY"
uv run --env-file .env scripts/cluster/summarize.py \
    --model_name Qwen/Qwen3.5-9B \
    --max_concurrency 36 \
    --max_tokens 1024 \
    --input-dataset posts_clustered_meta_001 \
    --dataset-name posts_double_summarized_001 \
    --template templates/summarize_claims_cluster.txt
```

### Tune Double-Clustering Hyperparameters

Sweeps `drop_frac` x `min_cluster_size` x `outlier_threshold` and evaluates meta-cluster coherence via LLM-as-a-Judge. Uses secrets from `.llm.env`.

```bash
uv run --env-file .env --env-file .llm.env python scripts/analysis/tune_double_clustering.py \
    --drop-frac "0.5,0.8,0.9,0.95" \
    --min-cluster-size "2,3,5,10" \
    --outlier-threshold "0.0,0.1,0.2" \
    --sample-size 10 --n-judge-runs 3 \
    --max-concurrency 32 \
    --output-dir outputs/tune_double_clustering
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--drop-frac` | `0.5,0.8,0.9,0.95` | Comma-separated drop_frac values |
| `--min-cluster-size` | `2,3,5,10` | Comma-separated min_cluster_size values |
| `--outlier-threshold` | `0.0,0.1,0.2` | Comma-separated outlier_threshold values |
| `--sample-size` | `20` | Clusters to sample per combo for LLM evaluation |
| `--n-judge-runs` | `1` | Independent sampling runs; >=2 enables 95% CI |
| `--model-name` | env `MODEL_NAME` | Model for coherence judging |
| `--max-concurrency` | `16` | Max concurrent LLM calls |
| `--source-dataset` | `posts_claims_embedded_001` | Source of embedded claims |
| `--output-dir` | `outputs/tune_double_clustering` | Output directory |
| `--write-example-clusters` | `3` | Example cluster text files for best config |

Results saved to `outputs/tune_double_clustering/results.csv`, `results.json`, heatmaps, and `example_clusters/` text files.

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

Run annotation on full dataset:

```bash
MODEL_PATH=$SCRATCH/20260331-chai-veracity/classifier_models/lr5e-05_bs8_wd0.01_seed365/best_model
MODEL_NAME=20260506_lr5e-05_bs8_wd0.01_seed365
sbatch --array=1-10 slurm/slurm_classifier_infer.sh
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
## Fact-Check Evaluation

Evaluate how well LLMs fact-check claims with vs. without web search access. Compares a no-search LLM (internal knowledge only) against pseudo-labels from web-search-equipped agents and ground truth labels from benchmark datasets.

Two search backends are available:
- **OpenAI web search** (`WebSearchTool`) — original pseudo-label source
- **Brave Search API** (custom `FunctionTool`) — alternative search backend, with optional date-range staleness

### Original Evaluation (OpenAI Web Search)

```bash
source .oai.env && \
uv run python scripts/analysis/fact_check_eval.py \
--n-samples 25 --n-agent-runs 5 --n-judge-runs 5 --max-concurrency 16 \
--agent-model-name gpt-5-mini --judge-model-name gpt-5-nano
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-samples` | `25` | Examples to subsample per dataset |
| `--n-agent-runs` | `5` | Web-search agent runs per example (for majority-vote pseudo-label) |
| `--n-judge-runs` | `5` | No-search LLM runs per example (for t-distribution CI) |
| `--max-concurrency` | `16` | Max concurrent LLM calls |
| `--model-name` | `$MODEL_NAME` | OpenAI model for both agent and judge |
| `--agent-model-name` | (same as `--model-name`) | Model for web-search agent (must support `web_search_preview`) |
| `--judge-model-name` | (same as `--model-name`) | Model for no-search judge |
| `--output-dir` | `outputs/fact_check_eval` | Output directory |
| `--seed` | `42` | Random seed for subsampling |
| `--datasets` | (built-in defaults) | Path to JSON file with custom dataset configs |

Outputs: `results.csv`, `results.json`, `agreement.json`, `fact_check_accuracy.png/pdf`, and per-example reasoning traces under `raw/`.

### Brave Search Baseline (with Staleness Axis)

Adds two Brave Search variants to the existing evaluation without re-running completed experiments:

- **Brave-full** (no freshness filter): pseudo-label generator, equivalent to OpenAI web search
- **Brave-stale** (`freshness=2020-01-01` to two weeks before today): judge evaluated against Brave-full pseudo-labels

```bash
source .brave.env && source .oai.env && \
uv run python scripts/analysis/fact_check_brave_eval.py \
    --n-samples 25 --n-agent-runs 5 --n-judge-runs 5 --max-concurrency 16 \
    --agent-model-name gpt-5-mini
```

Requires `BRAVE_SEARCH_API_KEY` set in `.brave.env` (subscription token from [Brave Search API](https://api-dashboard.search.brave.com)).

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-samples` | `25` | Examples per dataset (must match the original run's seed for consistent subsampling) |
| `--n-agent-runs` | `5` | Brave agent runs per example (both full and stale) |
| `--n-judge-runs` | `5` | Runs per example in existing no-search raw data (auto-detected if omitted) |
| `--max-concurrency` | `16` | Max concurrent LLM calls |
| `--agent-model-name` | `gpt-5-mini` | Model for Brave search agent calls |
| `--output-dir` | `outputs/fact_check_eval` | Output directory (merges with existing results) |
| `--existing-results` | `outputs/fact_check_eval/results.json` | Path to existing results for merging |
| `--seed` | `42` | Random seed for subsampling |
| `--datasets` | (built-in defaults) | Path to JSON file with custom dataset configs |

**Evaluation Matrix:**

| Judge | vs. Pseudo-label | vs. Ground Truth |
|---|---|---|
| No-search LLM | OpenAI pseudo + Brave pseudo | Ground truth |
| Brave-stale LLM | Brave pseudo | Ground truth |

A secondary chart (`fact_check_accuracy_brave_vs_oai.png`) shows Brave-stale accuracy vs. OpenAI pseudo-labels for reviewer reference.

**Outputs:** Merged `results.csv`, `results.json`, `agreement.json`, main chart with all bar groups (`fact_check_accuracy.png/pdf`), secondary chart (`fact_check_accuracy_brave_vs_oai.png/pdf`), and per-example Brave reasoning traces under `raw/<dataset>_brave_full/` and `raw/<dataset>_brave_stale/`. Existing `raw/<dataset>_search/` and `raw/<dataset>_nosearch/` directories are untouched.

### End-to-End Test

Quick verification on 5 examples with 3 runs each:

```bash
# Original OpenAI web-search eval
source .oai.env && \
uv run python scripts/analysis/fact_check_eval.py \
    --n-samples 5 --n-agent-runs 3 --n-judge-runs 3 --max-concurrency 4 \
    --agent-model-name gpt-5-mini --judge-model-name gpt-5-nano \
    --output-dir outputs/fact_check_eval_test

# Brave search baseline
source .brave.env && source .oai.env && \
uv run python scripts/analysis/fact_check_brave_eval.py \
    --n-samples 5 --n-agent-runs 3 --n-judge-runs 3 --max-concurrency 4 \
    --agent-model-name gpt-5-mini \
    --existing-results outputs/fact_check_eval/results.json
```
