import argparse
import asyncio
import base64
import logging
import math
import re
import resource
from collections import defaultdict
from dataclasses import dataclass, field
from typing import AsyncIterator, Any

import numpy as np
from sklearn.cluster import KMeans
from trafilatura.deduplication import Simhash

from s3_data_tool import S3DataTool, DataItem, RawDuckFilter
from s3_data_tool.s3_utils import enumerate_batches

logger = logging.getLogger(__name__)

_DAY_RE = re.compile(r"(\d{8})")

# Simhash is 64 bits; use the top 16 bits as a single LSH band.
_LSH_BAND_BITS = 16
_LSH_BAND_MASK = (1 << _LSH_BAND_BITS) - 1


def _decode_embeddings(encoded: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float32)


@dataclass
class _Collected:
    rows: list[DataItem] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)


async def _collect(
    batches: list[str] | None = None,
    collect_limit: int = 0,
) -> _Collected:
    """Collect rows with embeddings, optionally capped at *collect_limit* (0 = no limit)."""
    result = _Collected()
    total_rows = 0
    async with S3DataTool().filter_for_export(
        name="posts",
        base_columns=["text"],
        annotator_columns={"embeddings_128d": ["embedding"]},
        annotator_filters={
            "embeddings_128d": RawDuckFilter(sql="embedding IS NOT NULL"),
        },
    ) as generator:
        async for row in generator._iter_filtered_items(batches=batches):
            total_rows += 1
            encoded = row.data.get("embedding")
            if not encoded:
                continue
            result.rows.append(row)
            result.embeddings.append(_decode_embeddings(encoded))
            if collect_limit > 0 and len(result.rows) >= collect_limit:
                break
    logger.info(
        "Query returned %d rows, %d have embeddings (collect_limit=%s).",
        total_rows,
        len(result.rows),
        collect_limit or "none",
    )
    return result


async def _list_batches() -> list[str]:
    s3_tool = S3DataTool()
    kwargs = {}
    if s3_tool._endpoint_url:
        kwargs["endpoint_url"] = s3_tool._endpoint_url
    async with s3_tool._session.client("s3", **kwargs) as s3_client:
        return await enumerate_batches(
            s3_client, s3_tool._bucket, s3_tool._prefix, "posts"
        )


def _group_batches_by_day(batches: list[str]) -> dict[str, list[str]]:
    days: dict[str, list[str]] = defaultdict(list)
    for batch in batches:
        if m := _DAY_RE.search(batch):
            days[m.group(1)].append(batch)
        else:
            logger.warning("Could not parse date from batch name %r; skipping.", batch)
    return dict(days)


# ---------------------------------------------------------------------------
#  Cosine distance helpers
# ---------------------------------------------------------------------------

def _cosine_distances(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Cosine distances between *embeddings* (n × d) and *centroid* (d,)."""
    dots = np.dot(embeddings, centroid)
    emb_norms = np.linalg.norm(embeddings, axis=1)
    cent_norm = np.linalg.norm(centroid)
    denom = np.maximum(emb_norms * cent_norm, 1e-12)
    return 1.0 - dots / denom


# ---------------------------------------------------------------------------
#  Simhash + LSH deduplication
# ---------------------------------------------------------------------------

def _simhash_dedup_mask(
    texts: list[str],
    centroid: np.ndarray,
    embeddings: np.ndarray,
) -> list[int]:
    """Return indices of texts to keep after simhash LSH deduplication.

    Within each LSH bucket (top _LSH_BAND_BITS of the 64-bit simhash) only the
    element nearest to *centroid* is retained.
    """
    bucket_best: dict[int, int] = {}  # lsh_key -> index nearest to centroid
    distances = _cosine_distances(embeddings, centroid)

    for i, text in enumerate(texts):
        sh = Simhash(text).hash
        lsh_key = (sh >> (64 - _LSH_BAND_BITS)) & _LSH_BAND_MASK
        if lsh_key not in bucket_best or distances[i] < distances[bucket_best[lsh_key]]:
            bucket_best[lsh_key] = i

    return list(bucket_best.values())


# ---------------------------------------------------------------------------
#  Sampling strategies
# ---------------------------------------------------------------------------

def _sample_central(
    texts: list[str],
    embeddings: np.ndarray,
    centroid: np.ndarray,
    sample_size: int,
) -> list[int]:
    """Deduplicate via simhash LSH, then select up to *sample_size* nearest to centroid."""
    n = len(texts)
    if n == 0:
        return []
    if n <= sample_size:
        return list(range(n))

    kept = _simhash_dedup_mask(texts, centroid, embeddings)

    if len(kept) <= sample_size:
        return sorted(kept, key=lambda i: _cosine_distances(embeddings[i : i + 1], centroid)[0])

    distances = _cosine_distances(embeddings[kept], centroid)
    order = np.argsort(distances)
    return [kept[i] for i in order[:sample_size]]


def _sample_diverse(
    embeddings: np.ndarray,
    centroid: np.ndarray,
    sample_size: int,
) -> list[int]:
    """Select up to *sample_size* elements that maximize embedding diversity.

    Starts with the element nearest to *centroid*, then iteratively adds the
    element whose embedding is farthest from the mean of already-selected embeddings.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n <= sample_size:
        return list(range(n))

    distances = _cosine_distances(embeddings, centroid)
    first = int(np.argmin(distances))

    selected: list[int] = [first]
    remaining: set[int] = set(range(n)) - {first}

    for _ in range(sample_size - 1):
        mean_emb = np.mean(embeddings[selected], axis=0)
        best_idx = -1
        best_dist = -1.0
        for idx in remaining:
            d = np.linalg.norm(embeddings[idx] - mean_emb)
            if d > best_dist:
                best_dist = d
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


# ---------------------------------------------------------------------------
#  k-means clustering
# ---------------------------------------------------------------------------

def cluster_kmeans(
    rows: list[DataItem],
    embeddings: list[np.ndarray],
    n_clusters: int | None = None,
    random_state: int = 42,
) -> tuple[list[list[DataItem]], list[np.ndarray], np.ndarray]:
    """Cluster *rows* using k-means++ and return (clusters, centroids, all_embeddings).

    Returns clusters ordered largest-first, their centroids, and the full embedding matrix.
    """
    if not rows:
        return [], [], np.array([])

    n = len(rows)
    matrix = np.stack(embeddings).astype(np.float32)

    if n_clusters is None:
        n_clusters = max(2, int(math.sqrt(n / 2)))
        logger.info("Auto-selected n_clusters=%d for n=%d embeddings", n_clusters, n)

    n_clusters = min(n_clusters, n)
    if n_clusters < 2:
        logger.info("n=%d is too small for k-means; treating as single cluster", n)
        return [list(rows)], [matrix.mean(axis=0)], matrix

    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init="auto",
        random_state=random_state,
    )
    labels = km.fit_predict(matrix)
    centroids = km.cluster_centers_

    unique_labels = np.unique(labels)
    clusters: list[list] = [[] for _ in range(len(unique_labels))]
    label_to_cluster_idx = {lab: i for i, lab in enumerate(unique_labels)}
    for row, lab in zip(rows, labels):
        clusters[label_to_cluster_idx[lab]].append(row)

    # Sort largest-first; track permutation so centroids stay aligned
    sort_order = np.argsort([-len(c) for c in clusters])
    clusters[:] = [clusters[i] for i in sort_order]
    sorted_centroids = [centroids[label_to_cluster_idx[unique_labels[i]]] for i in sort_order]

    logger.info("Found %d clusters via k-means++ (n=%d)", n_clusters, n)
    return clusters, sorted_centroids, matrix


# ---------------------------------------------------------------------------
#  Output iterator
# ---------------------------------------------------------------------------

async def _get_cluster_iterator(
    clusters: list[list[DataItem]],
    centroids: list[np.ndarray],
    embeddings_matrix: np.ndarray,
    orig_idx_map: dict[str, int],
    sample_size: int,
) -> AsyncIterator[dict[str, list[Any]]]:
    """Yield one dict per cluster with full cluster data plus central / diverse samples.

    *orig_idx_map* maps row.id → index in *embeddings_matrix* (original collection order).
    """
    for ci, cluster in enumerate(clusters):
        n = len(cluster)
        cluster_indices = [orig_idx_map[row.id] for row in cluster]
        cluster_embs = embeddings_matrix[cluster_indices]
        centroid = centroids[ci]
        texts = [row.data.get("text", "") for row in cluster]

        central_idx = _sample_central(texts, cluster_embs, centroid, sample_size)
        diverse_idx = _sample_diverse(cluster_embs, centroid, sample_size)

        central_rows = [cluster[i] for i in central_idx]
        diverse_rows = [cluster[i] for i in diverse_idx]

        buffer: dict[str, list[Any]] = defaultdict(list)

        # Full cluster data
        for row in cluster:
            for column, value in row.data.items():
                if column != "embedding":
                    buffer[column].append(value)
        buffer["original_ids"] = buffer.pop("id", [])

        buffer["cluster_size"] = [n]
        buffer["central_sample_texts"] = [r.data.get("text", "") for r in central_rows]
        buffer["central_sample_ids"] = [r.id for r in central_rows]
        buffer["diverse_sample_texts"] = [r.data.get("text", "") for r in diverse_rows]
        buffer["diverse_sample_ids"] = [r.id for r in diverse_rows]

        yield buffer


# ---------------------------------------------------------------------------
#  Per-day processing
# ---------------------------------------------------------------------------

async def _process_day(
    day: str,
    day_batches: list[str],
    dataset_name: str,
    n_clusters: int | None = None,
    sample_size: int = 50,
    collect_limit: int = 0,
    output_batch_name: str | None = None,
    skip_upload: bool = False,
) -> list[list[DataItem]]:
    logger.info(
        "Day %s: processing %d batches (%s ... %s)",
        day,
        len(day_batches),
        day_batches[0],
        day_batches[-1],
    )

    collected = await _collect(batches=day_batches, collect_limit=collect_limit)

    if not collected.rows:
        logger.warning("Day %s: no embeddings found; skipping.", day)
        return []

    logger.info("Day %s: collected %d embeddings.", day, len(collected.rows))

    clusters, centroids, matrix = cluster_kmeans(
        collected.rows, collected.embeddings, n_clusters=n_clusters,
    )

    orig_idx_map = {row.id: i for i, row in enumerate(collected.rows)}

    for i, cluster in enumerate(clusters):
        logger.info(
            "Day %s cluster %d: %d items (central=%d, diverse=%d)",
            day,
            i,
            len(cluster),
            min(sample_size, len(cluster)),
            min(sample_size, len(cluster)),
        )

    batch_name = output_batch_name or f"bsky-kmeans-{day}"

    if skip_upload:
        # Drain the iterator to exercise the full sampling code path without writing to S3.
        async for _ in _get_cluster_iterator(clusters, centroids, matrix, orig_idx_map, sample_size):
            pass
        logger.info("Day %s: processed %d clusters (upload skipped).", day, len(clusters))
    else:
        async with S3DataTool().dataset_generator() as dataset_generator:
            await dataset_generator.from_async_iterator(
                _get_cluster_iterator(clusters, centroids, matrix, orig_idx_map, sample_size),
                name=dataset_name,
                batch=batch_name,
                streaming_configs=S3DataTool.StreamingConfigs(chunk_size=10),
            )
        logger.info("Day %s: uploaded %d clusters as batch %r.", day, len(clusters), batch_name)

    return clusters


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

async def main() -> list[list]:
    parser = argparse.ArgumentParser(description="k-means++ clustering with central/diverse sampling")
    parser.add_argument("--limit", type=int, default=1, help="Max days to process (default: 1, 0=all)")
    parser.add_argument("--dataset-name", default="posts_clustered_kmeans_001")
    parser.add_argument("--n-clusters", type=int, default=None, help="Number of k-means clusters (default: sqrt(n/2))")
    parser.add_argument("--sample-size", type=int, default=50, help="Max elements per sample per cluster (default: 50)")
    parser.add_argument("--collect-limit", type=int, default=0, help="Cap rows collected from S3, 0=no limit (dry-run)")
    parser.add_argument("--batch", default=None, help="Process only this specific batch name (overrides --limit).")
    parser.add_argument("--day", default=None, help="Process only batches for this day (YYYYMMDD). Overrides --limit.")
    parser.add_argument("--skip-upload", action="store_true", help="Skip S3 upload (for memory profiling)")
    parser.add_argument("--output-batch-name", default=None, help="Override auto-generated output batch name")
    args = parser.parse_args()

    if args.batch:
        m = _DAY_RE.search(args.batch)
        day = m.group(1) if m else "unknown"
        output_batch_name = args.output_batch_name or f"x-posts-kmeans-{day}"
        all_clusters = await _process_day(
            day,
            [args.batch],
            args.dataset_name,
            n_clusters=args.n_clusters,
            sample_size=args.sample_size,
            collect_limit=args.collect_limit,
            output_batch_name=output_batch_name,
            skip_upload=args.skip_upload,
        )
        return all_clusters

    all_batches = await _list_batches()
    days = _group_batches_by_day(all_batches)
    logger.info("Found %d batches across %d days.", len(all_batches), len(days))

    if args.day:
        if args.day not in days:
            logger.error("Day %s not found in available batches. Available: %s", args.day, sorted(days.keys()))
            return []
        days = {args.day: days[args.day]}

    sorted_days = sorted(days.keys())

    all_clusters: list[list] = []
    processed = 0
    for day in sorted_days:
        if args.limit > 0 and processed >= args.limit:
            break
        clusters = await _process_day(
            day,
            days[day],
            args.dataset_name,
            n_clusters=args.n_clusters,
            sample_size=args.sample_size,
            collect_limit=args.collect_limit,
            output_batch_name=args.output_batch_name,
            skip_upload=args.skip_upload,
        )
        if clusters:
            processed += 1
            all_clusters.extend(clusters)

    return all_clusters


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_mb = usage.ru_maxrss / 1024.0  # KB → MB on Linux
    logger.info("Peak RSS: %.1f MB", peak_mb)
