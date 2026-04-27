# Draft: scripts/cluster/summarize.py Design

## Requirements (from user)
- Transform output of `clustering.py` using an LLM
- Turn each cluster (one row in `posts_clustered_001`) into a list of synopses (one per topic discussed in that cluster)
- Use `filter_for_annotation` to stream clusters
- Use `dataset_generator.from_async_iterator` to create a new synopsis dataset
- Access methods similar to clustering.py

## Technical Context (research findings)

### Existing patterns
- **feasibility.py**: Uses `filter_for_annotation` + `annotator_view.annotate()` — built-in concurrency (producer-consumer)
- **clustering.py**: Uses `filter_for_annotation` as read-only generator + `dataset_generator.from_async_iterator` — sequential iteration

### S3DataTool API
- `filter_for_annotation(name, annotator_name, base_columns, annotator_columns?, ...)` → async context manager yielding `FilterForAnnotation`
- `FilterForAnnotation.__aiter__()` → `AsyncIterator[DataItem]` (read-only iteration)
- `FilterForAnnotation.annotate(annotation_fn, max_concurrency, ...)` → producer-consumer pipeline
- `dataset_generator()` → async context manager yielding `DatasetGenerator`
- `DatasetGenerator.from_async_iterator(iterator, name, batch, streaming_configs, deduplicate_on?)` → creates dataset

### DataItem fields
- `data: dict[str, Any]` — the row data
- `id: str` — unique identifier
- `batch: str` — batch name

### Cluster dataset structure (posts_clustered_001)
- Each row = one cluster
- `text`: list of all post texts in cluster
- `original_ids`: list of original post IDs (renamed from `id`)
- Other columns stacked as lists
- `embedding` column removed during creation

### LLM setup
- Uses `openai.AsyncOpenAI()` with env vars (`OPENAI_BASE_URL`, `OPENAI_API_KEY`)
- `@backoff.on_exception(backoff.expo, [openai.APIConnectionError])` on inner generate
- Retry wrapper with `max_retries` loop
- Templates in `templates/` directory

## Open Questions

### Q1: How to read `posts_clustered_001` with `filter_for_annotation`?
`filter_for_annotation(name, annotator_name, ...)` joins a base dataset with an annotator's data. The cluster dataset was created via `dataset_generator.from_async_iterator` as a standalone dataset, NOT as annotations. 

Options:
- A) Use `filter_for_annotation(name="posts_clustered_001", annotator_name="summarize", ...)` — would this work without existing annotations?
- B) Maybe the pipeline needs a step to annotate posts with cluster_id first?
- C) Use `filter_for_export` instead?

### Q2: One output row per cluster or per topic?
User said "a list of synopsis, one for each topic discussed in that cluster." This could mean:
- A) One output row per cluster, containing a list of synopsis strings
- B) Multiple output rows, one per synopsis (topic), with cluster reference

### Q3: Concurrency approach?
- Pattern from feasibility.py (`annotator_view.annotate()`) handles concurrency internally but outputs annotations to the same dataset
- Pattern from clustering.py (read-only iteration) requires manual concurrency management

## Scope Boundaries
- INCLUDE: `scripts/cluster/summarize.py` script
- INCLUDE: `templates/summarize.txt` prompt template
- EXCLUDE: Changes to clustering.py
- EXCLUDE: SLURM script changes (unless needed)
