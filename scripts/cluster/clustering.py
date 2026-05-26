import argparse
import asyncio
import base64
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Any

import faiss
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.cluster.vq import kmeans2

from pydantic import BaseModel

from s3_data_tool import S3DataTool, DataItem, RawDuckFilter
from s3_data_tool.s3_utils import enumerate_batches  # not in __all__; import from submodule

logger = logging.getLogger(__name__)


class SmallClusterInfo(BaseModel):
    """Diagnostic info for one small cluster considered for merging."""
    size: int
    max_sim: float
    nearest_large_size: int
    row_indices: list[int] = []

    class Config:
        # Exclude row_indices from serialization by default — it can be large
        pass


class ClusterDiagnostics(BaseModel):
    """Diagnostics returned by cluster_knn_graph_with_diagnostics."""
    n_initial: int
    n_after_split: int
    n_after_merge: int
    total_items: int
    small_clusters: list[SmallClusterInfo]


def _decode_embeddings(encoded: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float32)


@dataclass
class _Collected:
    rows: list[DataItem] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)


async def _collect(batches: list[str] | None = None) -> _Collected:
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
    logger.info("Query returned %d rows, %d have embeddings.", total_rows, len(result.rows))
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


import re

_DAY_RE = re.compile(r"(\d{8})")  # extract YYYYMMDD from batch names


def _group_batches_by_day(batches: list[str]) -> dict[str, list[str]]:
    """Group batch names by the 8-digit YYYYMMDD date embedded in them."""
    days: dict[str, list[str]] = defaultdict(list)
    for batch in batches:
        if m := _DAY_RE.search(batch):
            days[m.group(1)].append(batch)
        else:
            logger.warning("Could not parse date from batch name %r; skipping.", batch)
    return dict(days)


def _build_knn_graph(
    matrix: np.ndarray,
    k: int,
    drop_frac: float = 0.9,  # e.g. 0.9 means drop the farthest 90%
) -> csr_matrix:
    """
    Build a symmetric pruned kNN graph over *matrix* using cosine similarity.

    Each row is connected to its *k* nearest neighbors (excluding itself),
    then the farthest `drop_frac` fraction of those edges are removed.
    """
    if not (0.0 <= drop_frac < 1.0):
        raise ValueError("drop_frac must be in [0.0, 1.0)")

    n, dim = matrix.shape

    matrix = matrix.copy()
    faiss.normalize_L2(matrix)  # in-place; enables cosine via inner product

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    # k+1 because FAISS returns the query point itself as the first neighbor
    similarities, neighbors = index.search(matrix, k + 1)
    similarities = similarities[:, 1:]  # drop self-match column
    neighbors = neighbors[:, 1:]

    # Keep only the closest fraction of neighbors for each node
    keep_k = max(1, int(np.ceil(k * (1.0 - drop_frac))))
    neighbors = neighbors[:, :keep_k]
    similarities = similarities[:, :keep_k]

    row_idx = np.repeat(np.arange(n), keep_k)
    col_idx = neighbors.ravel()

    # You can use ones(...) if you only care about connectivity.
    # Keeping similarity scores is more flexible.
    data = similarities.ravel().astype(np.float32)

    graph = csr_matrix((data, (row_idx, col_idx)), shape=(n, n))

    # Union symmetrization: keep edge if either direction kept it
    graph = graph.maximum(graph.T)
    graph.eliminate_zeros()
    return graph


def _compute_centroids(
    matrix: np.ndarray,    # (n, dim) float32
    labels: np.ndarray,    # (n,) int
    n_clusters: int,
) -> np.ndarray:           # (n_clusters, dim) float32
    """Compute mean embedding for each cluster. Handles non-contiguous labels."""
    dim = matrix.shape[1]
    centroids = np.zeros((n_clusters, dim), dtype=np.float32)
    np.add.at(centroids, labels, matrix)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)[:, np.newaxis]
    np.maximum(counts, 1.0, out=counts)  # guard against division by zero
    centroids /= counts
    return centroids


def _split_overlarge_clusters(
    matrix: np.ndarray,       # (n, dim)
    labels: np.ndarray,       # (n,) int
    max_size: int,
    k: int,
    drop_frac: float,
    split_k_scale: float,     # e.g. 0.5 → halve k for sub-clustering
) -> np.ndarray:              # (n,) int — new, possibly finer labels
    """Recursively re-cluster components exceeding *max_size*.

    Tries kNN + connected-components first.  Falls back to k-means when
    the kNN graph remains a single component even at low k.
    """
    unique_labels, compressed = np.unique(labels, return_inverse=True)
    n_comp = len(unique_labels)

    new_compressed = np.empty(len(compressed), dtype=np.int32)
    queue = deque()
    for cid in range(n_comp):
        idx = np.flatnonzero(compressed == cid)
        if len(idx) > max_size:
            queue.append((idx, k))
        else:
            new_compressed[idx] = cid

    oversized_count = len(queue)
    if oversized_count > 0:
        sizes = [len(item[0]) for item in queue]
        logger.info(
            "Splitting %d oversized clusters (sizes: %s)",
            oversized_count, sorted(sizes, reverse=True),
        )

    next_label = n_comp

    while queue:
        idx, cur_k = queue.popleft()

        if len(idx) <= max_size:
            continue

        sub_k = max(2, int(cur_k * split_k_scale))
        sub_matrix = matrix[idx]
        sub_graph = _build_knn_graph(sub_matrix, k=sub_k, drop_frac=drop_frac)
        sub_n, sub_labels = connected_components(sub_graph, directed=False)

        if sub_n == 1:
            if sub_k <= 3:
                # kNN with tiny k still yields one component — fall back to k-means
                n_means = max(2, int(np.ceil(len(idx) / max_size)))
                logger.info(
                    "kNN stuck (size=%d, k=%d); falling back to k-means (k=%d)",
                    len(idx), sub_k, n_means,
                )
                _, kmeans_labels = kmeans2(sub_matrix.astype(np.float64), n_means, minit="points")
                sub_n = n_means
                sub_labels = kmeans_labels.astype(np.int32)
            else:
                # Still one component — re-enqueue with reduced k for another attempt
                logger.info(
                    "Cluster (size=%d) still single component at k=%d; retrying",
                    len(idx), sub_k,
                )
                queue.append((idx, sub_k))
                continue

        if sub_n == 1:  # k-means also returned 1 cluster — accept
            new_compressed[idx] = next_label
            next_label += 1
            logger.info(
                "Cluster (size=%d) cannot be split further (k=%d)",
                len(idx), sub_k,
            )
        else:
            child_sizes = [int((sub_labels == sid).sum()) for sid in range(sub_n)]
            logger.info(
                "Split cluster (size=%d) into %d pieces: %s",
                len(idx), sub_n, sorted(child_sizes, reverse=True),
            )
            for sid in range(sub_n):
                child_mask = (sub_labels == sid)
                child_idx = idx[child_mask]
                if len(child_idx) > max_size and sub_k > 1:
                    queue.append((child_idx, sub_k))
                else:
                    new_compressed[child_idx] = next_label
                    next_label += 1

    _, result = np.unique(new_compressed, return_inverse=True)
    return result.astype(np.int32)


def _merge_undersized_clusters(
    matrix: np.ndarray,         # (n, dim)
    labels: np.ndarray,         # (n,) int
    min_size: int,
    outlier_threshold: float,   # cosine similarity below which small clusters are dropped
) -> tuple[np.ndarray, set, list[dict]]:   # (new_labels, outlier_label_indices, small_cluster_sims)
    """Merge clusters below *min_size* into nearest larger cluster.

    If *outlier_threshold* > 0, a small cluster whose maximum cosine similarity
    to any large cluster falls below the threshold is marked as an outlier
    (its label appears in the returned outlier set) instead of being merged.

    The third return value is a list of dicts describing each small cluster:
        {"size": int, "max_sim": float, "nearest_large_size": int}
    This data is used by diagnostic tooling to select an outlier_threshold.
    """
    unique_labels, compressed = np.unique(labels, return_inverse=True)
    n = len(unique_labels)

    if n <= 1:
        return labels, set(), []

    sizes = np.bincount(compressed, minlength=n)
    small = np.where(sizes < min_size)[0]
    large = np.where(sizes >= min_size)[0]

    if len(small) == 0:
        return labels, set(), []

    if len(large) == 0:
        if outlier_threshold > 0:
            # All clusters are small — compute pairwise similarity and keep
            # only those that are close to at least one other cluster
            centroids = _compute_centroids(matrix, compressed, n)
            centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
            sim = centroids_norm @ centroids_norm.T  # (n, n)
            np.fill_diagonal(sim, -1.0)  # ignore self
            max_sim = sim.max(axis=1)   # (n,) — best similarity to any other cluster
            keep_mask = max_sim >= outlier_threshold
            outliers = set(unique_labels[~keep_mask].tolist())
            if outliers:
                logger.info(
                    "Dropping %d outliers (all clusters below min_size=%d, threshold=%.2f)",
                    len(outliers), min_size, outlier_threshold,
                )
            keep_compressed = keep_mask[compressed]
            new_compressed = compressed[keep_compressed]
            new_unique, new_inverse = np.unique(new_compressed, return_inverse=True)
            sim_data = [{"size": int(sizes[i]), "max_sim": float(max_sim[i]),
                         "nearest_large_size": 0,
                         "row_indices": np.flatnonzero(compressed == i).tolist()}
                        for i in range(n)]
            return new_inverse.astype(np.int32), outliers, sim_data
        else:
            logger.warning(
                "All %d clusters are below min_cluster_size=%d; skipping merge.", n, min_size,
            )
            sim_data = [{"size": int(sizes[i]), "max_sim": 0.0,
                         "nearest_large_size": 0,
                         "row_indices": np.flatnonzero(compressed == i).tolist()}
                        for i in range(n)]
            return labels, set(), sim_data

    centroids = _compute_centroids(matrix, compressed, n)
    small_c = centroids[small]
    large_c = centroids[large]

    small_norm = small_c / np.linalg.norm(small_c, axis=1, keepdims=True)
    large_norm = large_c / np.linalg.norm(large_c, axis=1, keepdims=True)

    sims = small_norm @ large_norm.T          # (n_small, n_large)
    best_sim = sims.max(axis=1)               # (n_small,) — best match per small cluster
    best_large_idx = large[sims.argmax(axis=1)]  # target compressed label

    outlier_labels: set[int] = set()
    new_compressed = compressed.copy()

    for i, s_label in enumerate(small):
        if outlier_threshold > 0 and best_sim[i] < outlier_threshold:
            outlier_labels.add(int(unique_labels[s_label]))
        else:
            new_compressed[compressed == s_label] = best_large_idx[i]

    # Compress out dropped outlier positions
    if outlier_labels:
        # Remove items whose label is in outlier set
        outlier_mask = np.isin(labels, list(outlier_labels))
        keep_idx = np.flatnonzero(~outlier_mask)
        kept_labels = new_compressed[keep_idx] if len(keep_idx) > 0 else np.array([], dtype=new_compressed.dtype)
        _, result = np.unique(kept_labels, return_inverse=True) if len(kept_labels) > 0 else (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        logger.info(
            "Dropped %d items in %d outlier clusters (threshold=%.2f)",
            outlier_mask.sum(), len(outlier_labels), outlier_threshold,
        )
        sim_data = [{"size": int(sizes[s]), "max_sim": float(best_sim[i]),
                     "nearest_large_size": int(sizes[best_large_idx[i]]),
                     "row_indices": np.flatnonzero(compressed == s).tolist()}
                    for i, s in enumerate(small)]
        return result.astype(np.int32), outlier_labels, sim_data
    else:
        _, result = np.unique(new_compressed, return_inverse=True)
        sim_data = [{"size": int(sizes[s]), "max_sim": float(best_sim[i]),
                     "nearest_large_size": int(sizes[best_large_idx[i]]),
                     "row_indices": np.flatnonzero(compressed == s).tolist()}
                    for i, s in enumerate(small)]
        return result.astype(np.int32), set(), sim_data


def _merge_similar_clusters(
    matrix: np.ndarray,         # (n, dim)
    labels: np.ndarray,         # (n,) int
    merge_threshold: float,     # cosine similarity above which clusters are merged
) -> np.ndarray:                # (n,) int — new labels
    """Greedily merge clusters whose centroid cosine similarity exceeds *merge_threshold*.

    Repeatedly finds the most similar pair of clusters and merges them, updating
    centroids after each merge, until no pair exceeds the threshold.  This catches
    topically-related clusters that the kNN graph kept separate because their
    individual items were not each other's nearest neighbors.
    """
    unique_labels, compressed = np.unique(labels, return_inverse=True)
    n_clusters = len(unique_labels)

    if n_clusters <= 1:
        return labels

    sizes = np.bincount(compressed, minlength=n_clusters)
    centroids = _compute_centroids(matrix, compressed, n_clusters)
    centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Pairwise cosine similarity matrix (upper triangle only for efficiency)
    sim = centroids_norm @ centroids_norm.T  # (n_clusters, n_clusters)
    np.fill_diagonal(sim, -1.0)

    # Greedy merging: repeatedly merge the most similar pair
    while True:
        best_i, best_j = np.unravel_index(sim.argmax(), sim.shape)
        best_sim = sim[best_i, best_j]
        if best_sim < merge_threshold:
            break

        # Merge cluster j into cluster i
        mask_j = (compressed == best_j)
        compressed[mask_j] = best_i

        # Update centroid for i (weighted average)
        size_i, size_j = sizes[best_i], sizes[best_j]
        merged_centroid = ((centroids[best_i] * size_i + centroids[best_j] * size_j)
                           / (size_i + size_j))
        centroids[best_i] = merged_centroid
        sizes[best_i] = size_i + size_j

        # Invalidate cluster j
        sizes[best_j] = 0
        centroids[best_j] = 0.0
        sim[best_j, :] = -1.0
        sim[:, best_j] = -1.0

        # Recompute similarities for cluster i
        ci_norm = centroids[best_i] / np.linalg.norm(centroids[best_i])
        for k in range(n_clusters):
            if k == best_i or sizes[k] == 0:
                continue
            ck_norm = centroids[k] / np.linalg.norm(centroids[k])
            sim[best_i, k] = float(ci_norm @ ck_norm)
            sim[k, best_i] = sim[best_i, k]
        sim[best_i, best_i] = -1.0

        logger.debug("Merged clusters %d and %d (sim=%.3f)", best_i, best_j, best_sim)

    # Compact out invalidated clusters
    valid = sizes > 0
    if valid.all():
        return labels

    old_to_new = np.full(n_clusters, -1, dtype=np.int32)
    old_to_new[valid] = np.arange(valid.sum())
    _, result = np.unique(old_to_new[compressed], return_inverse=True)
    logger.info("Centroid merge: %d -> %d clusters (threshold=%.2f)",
                n_clusters, valid.sum(), merge_threshold)
    return result.astype(np.int32)


def cluster_knn_graph(
    rows: list,
    embeddings: list[np.ndarray],
    k: int | None = None,
    drop_frac: float = 0.9,
    min_cluster_size: int = 0,
    max_cluster_size: int = 0,
    split_k_scale: float = 0.5,
    outlier_threshold: float = 0.0,
    merge_threshold: float = 0.0,
) -> list[list[DataItem]]:
    """
    Cluster *rows* by building a kNN graph over their *embeddings* and
    extracting connected components.

    The number of output clusters is determined organically by the data's
    density structure — no target cluster count is required.

    Args:
        rows:              DataItem objects, parallel to *embeddings*.
        embeddings:        128-d float32 vectors, one per row.
        k:                 Neighbours per node.  Defaults to ceil(log2(n)).
        drop_frac:         Fraction of farthest neighbor edges to prune per node
                           (default 0.9).
        min_cluster_size:  Merge clusters smaller than this into nearest larger
                           cluster (default 0 = disabled).
        max_cluster_size:  Split clusters larger than this by re-clustering with
                           reduced k (default 0 = disabled).
        split_k_scale:     Multiplier for k when sub-clustering oversized
                           components (default 0.5).
        outlier_threshold: Cosine similarity floor. Small clusters below this
                           threshold vs all large clusters are dropped as
                           outliers.  Only active when min_cluster_size > 0
                           (default 0.0 = disabled).
        merge_threshold:   Cosine similarity above which any two clusters are
                           merged (greedy centroid-based).  Use to collapse
                           near-duplicate clusters that the kNN graph kept
                           separate (default 0.0 = disabled).

    Returns:
        list[list[DataItem]] — one inner list per discovered cluster,
        ordered largest-first. Outlier items (if any) are excluded.
    """
    if not rows:
        return []

    n = len(rows)

    if k is None:
        k = math.ceil(math.log2(n))
        logger.info(f"Auto-selected k={k} for n={n} embeddings")

    matrix = np.stack(embeddings).astype(np.float32)
    graph = _build_knn_graph(matrix, k=k, drop_frac=drop_frac)

    n_clusters, labels = connected_components(graph, directed=False)
    logger.info(
        f"Initial: {n_clusters} clusters (k={k}, n={n})",
    )

    # Post-processing: split → merge → centroid-merge → drop outliers
    if max_cluster_size > 0:
        labels = _split_overlarge_clusters(
            matrix, labels, max_cluster_size, k, drop_frac, split_k_scale,
        )
        logger.info("After split: %d labels", len(np.unique(labels)))

    outlier_labels: set[int] = set()
    if min_cluster_size > 0:
        labels, outlier_labels, _small_cluster_sims = _merge_undersized_clusters(
            matrix, labels, min_cluster_size, outlier_threshold,
        )
        logger.info("After merge: %d labels", len(np.unique(labels)))

    if merge_threshold > 0:
        labels = _merge_similar_clusters(
            matrix, labels, merge_threshold,
        )

    # Rebuild cluster lists — labels may be non-contiguous after processing
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    clusters: list[list] = [[] for _ in range(len(unique_labels))]
    for row, label_idx in zip(rows, inverse):
        clusters[label_idx].append(row)

    clusters.sort(key=len, reverse=True)

    if outlier_labels:
        logger.info(
            "Retained %d clusters, dropped %d outliers (%d items lost).",
            len(clusters), len(outlier_labels),
            n - sum(len(c) for c in clusters),
        )

    return clusters


def cluster_knn_graph_with_diagnostics(
    rows: list,
    embeddings: list[np.ndarray],
    k: int | None = None,
    drop_frac: float = 0.9,
    min_cluster_size: int = 0,
    max_cluster_size: int = 0,
    split_k_scale: float = 0.5,
    outlier_threshold: float = 0.0,
    merge_threshold: float = 0.0,
) -> tuple[list[list[DataItem]], ClusterDiagnostics]:
    """Same as cluster_knn_graph but returns per-cluster diagnostic info.

    The diagnostics payload includes similarity scores for every small cluster
    (those below *min_cluster_size*), enabling threshold selection without
    re-running clustering.

    Returns:
        (clusters, diagnostics) — clusters are the same as cluster_knn_graph;
        diagnostics carries ClusterDiagnostics with per-small-cluster similarity
        data, cluster counts, and total item count.
    """
    if not rows:
        return [], ClusterDiagnostics(
            n_initial=0, n_after_split=0, n_after_merge=0, total_items=0, small_clusters=[],
        )

    n = len(rows)

    if k is None:
        k = math.ceil(math.log2(n))
        logger.info(f"Auto-selected k={k} for n={n} embeddings")

    matrix = np.stack(embeddings).astype(np.float32)
    graph = _build_knn_graph(matrix, k=k, drop_frac=drop_frac)

    n_clusters, labels = connected_components(graph, directed=False)
    logger.info(f"Initial: {n_clusters} clusters (k={k}, n={n})")

    # Split
    n_after_split = n_clusters
    if max_cluster_size > 0:
        labels = _split_overlarge_clusters(
            matrix, labels, max_cluster_size, k, drop_frac, split_k_scale,
        )
        n_after_split = len(np.unique(labels))
        logger.info("After split: %d labels", n_after_split)

    # Merge — capture similarity data
    small_cluster_sims: list[dict] = []
    n_after_merge = n_after_split
    if min_cluster_size > 0:
        labels, _outlier_labels, small_cluster_sims = _merge_undersized_clusters(
            matrix, labels, min_cluster_size, outlier_threshold,
        )
        n_after_merge = len(np.unique(labels))
        logger.info("After merge: %d labels", n_after_merge)

    # Centroid-based merge of near-duplicate clusters
    if merge_threshold > 0:
        labels = _merge_similar_clusters(
            matrix, labels, merge_threshold,
        )

    # Rebuild cluster lists
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    clusters: list[list] = [[] for _ in range(len(unique_labels))]
    for row, label_idx in zip(rows, inverse):
        clusters[label_idx].append(row)

    clusters.sort(key=len, reverse=True)

    diagnostics = ClusterDiagnostics(
        n_initial=n_clusters,
        n_after_split=n_after_split,
        n_after_merge=n_after_merge,
        total_items=n,
        small_clusters=[SmallClusterInfo(**sim) for sim in small_cluster_sims],
    )

    return clusters, diagnostics


async def _get_cluster_iterator(
    clusters: list[list[DataItem]],
) -> AsyncIterator[dict[str, list[Any]]]:
    """Produce iterator of vectorized clusters.

    Each output is for one cluster.
    Each original features is stacked across all rows in each cluster.
    """

    buffer: dict[str, list[Any]] = defaultdict(list)
    for cluster in clusters:
        for row in cluster:
            for _column, _value in row.data.items():
                buffer[_column].append(_value)

        buffer.pop("embedding")
        buffer["original_ids"] = buffer.pop("id")
        yield buffer
        buffer = defaultdict(list)


async def _process_day(
    day: str,
    day_batches: list[str],
    dataset_name: str,
    k: int | None = None,
    drop_frac: float = 0.9,
    min_cluster_size: int = 0,
    max_cluster_size: int = 0,
    split_k_scale: float = 0.5,
    outlier_threshold: float = 0.0,
    merge_threshold: float = 0.0,
    output_batch_name: str | None = None,
) -> list[list[DataItem]]:
    """Collect, cluster, and upload one day's worth of data."""
    logger.info(
        "Day %s: processing %d batches (%s ... %s)",
        day, len(day_batches), day_batches[0], day_batches[-1],
    )

    collected = await _collect(batches=day_batches)

    if not collected.rows:
        logger.warning("Day %s: no embeddings found; skipping.", day)
        return []

    logger.info("Day %s: collected %d embeddings.", day, len(collected.rows))

    clusters = cluster_knn_graph(
        collected.rows, collected.embeddings,
        k=k, drop_frac=drop_frac,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        split_k_scale=split_k_scale,
        outlier_threshold=outlier_threshold,
        merge_threshold=merge_threshold,
    )

    for i, cluster in enumerate(clusters):
        logger.info("Day %s cluster %d: %d items", day, i, len(cluster))

    batch_name = output_batch_name or f"bsky-trending-{day}"
    async with S3DataTool().dataset_generator() as dataset_generator:
        await dataset_generator.from_async_iterator(
            _get_cluster_iterator(clusters),
            name=dataset_name,
            batch=batch_name,
            streaming_configs=S3DataTool.StreamingConfigs(chunk_size=100),
            deduplicate_on=["text", "source_id"],
        )

    logger.info("Day %s: uploaded %d clusters as batch %r.", day, len(clusters), batch_name)
    return clusters


async def main() -> list[list]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1, help="Max days to process (default: 1, 0=all)")
    parser.add_argument("--dataset-name", default="posts_clustered_001")
    parser.add_argument("--k", type=int, default=None, help="kNN neighbours per node (default: log2(n))")
    parser.add_argument("--drop-frac", type=float, default=0.95, help="Fraction of farthest edges to prune (default: 0.95)")
    parser.add_argument("--min-cluster-size", type=int, default=0,
                        help="Merge clusters smaller than this (default: 0 = disabled)")
    parser.add_argument("--max-cluster-size", type=int, default=0,
                        help="Split clusters larger than this (default: 0 = disabled)")
    parser.add_argument("--split-k-scale", type=float, default=0.5,
                        help="Scale factor for k when sub-clustering (default: 0.5)")
    parser.add_argument("--outlier-threshold", type=float, default=0.0,
                        help="Cosine similarity floor for small-cluster merging (default: 0.0 = disabled)")
    parser.add_argument("--merge-threshold", type=float, default=0.0,
                        help="Cosine similarity above which any two cluster centroids are merged (default: 0.0 = disabled)")
    parser.add_argument("--batch", default=None,
                        help="Process only this specific batch name (overrides --limit).")
    args = parser.parse_args()

    if args.batch:
        m = _DAY_RE.search(args.batch)
        day = m.group(1) if m else "unknown"
        output_batch_name = f"x-posts-clusters-{day}"
        all_clusters = await _process_day(
            day, [args.batch], args.dataset_name,
            k=args.k, drop_frac=args.drop_frac,
            min_cluster_size=args.min_cluster_size,
            max_cluster_size=args.max_cluster_size,
            split_k_scale=args.split_k_scale,
            outlier_threshold=args.outlier_threshold,
            merge_threshold=args.merge_threshold,
            output_batch_name=output_batch_name,
        )
        return all_clusters

    all_batches = await _list_batches()
    days = _group_batches_by_day(all_batches)
    logger.info(
        "Found %d batches across %d days.",
        len(all_batches), len(days),
    )

    sorted_days = sorted(days.keys())

    all_clusters: list[list] = []
    processed = 0
    for day in sorted_days:
        if args.limit > 0 and processed >= args.limit:
            break
        clusters = await _process_day(
            day, days[day], args.dataset_name,
            k=args.k, drop_frac=args.drop_frac,
            min_cluster_size=args.min_cluster_size,
            max_cluster_size=args.max_cluster_size,
            split_k_scale=args.split_k_scale,
            outlier_threshold=args.outlier_threshold,
            merge_threshold=args.merge_threshold,
        )
        if clusters:  # only count days that yielded results
            processed += 1
            all_clusters.extend(clusters)

    return all_clusters


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
