import argparse
import asyncio
import base64
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Any

import faiss
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from s3_data_tool import S3DataTool, DataItem

logger = logging.getLogger(__name__)


def _decode_embeddings(encoded: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float32)


@dataclass
class _Collected:
    rows: list[DataItem] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)


async def _collect() -> _Collected:
    result = _Collected()
    async with S3DataTool().filter_for_annotation(
        name="posts",
        annotator_name="clustering",
        base_columns=["text"],
        annotator_columns={"embeddings_128d": ["embedding"]},
    ) as generator:
        async for row in generator:
            encoded = row.data.get("embedding")
            if not encoded:
                continue
            result.rows.append(row)
            result.embeddings.append(_decode_embeddings(encoded))
    return result


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


def cluster_knn_graph(
    rows: list,
    embeddings: list[np.ndarray],
    k: int | None = None,
) -> list[list[DataItem]]:
    """
    Cluster *rows* by building a kNN graph over their *embeddings* and
    extracting connected components.

    The number of output clusters is determined organically by the data's
    density structure — no target cluster count is required.

    Args:
        rows:       DataItem objects, parallel to *embeddings*.
        embeddings: 128-d float32 vectors, one per row.
        k:          Neighbours per node.  Defaults to ceil(log2(n)), which
                    keeps the graph sparse while still connecting natural
                    neighbourhoods for corpus sizes from hundreds to millions.

    Returns:
        list[list[DataItem]] — one inner list per discovered cluster,
        ordered largest-first.
    """
    if not rows:
        return []

    n = len(rows)

    if k is None:
        # log2(n) is a well-established heuristic: sparse enough to avoid
        # merging distinct topics, dense enough to bridge legitimate clusters.
        k = 2
        logger.info(f"Auto-selected k={k} for n={n} embeddings")

    matrix = np.stack(embeddings).astype(np.float32)
    graph = _build_knn_graph(matrix, k=k)

    n_clusters, labels = connected_components(graph, directed=False)
    logger.info(
        f"Found {n_clusters} clusters (k={k}, n={n})",
    )

    clusters: list[list] = [[] for _ in range(n_clusters)]
    for row, label in zip(rows, labels):
        clusters[label].append(row)

    # Largest clusters first for convenient downstream inspection
    clusters.sort(key=len, reverse=True)
    return clusters


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


async def main() -> list[list]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dataset-name", default="posts_clustered_001")
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H")
    batch_name = f"bsky-trending-{timestamp}"

    collected = await _collect()

    if not collected.rows:
        logger.warning("No embeddings found; nothing to cluster.")
        return []

    logger.info("Collected %d embeddings.", len(collected.rows))

    clusters = cluster_knn_graph(collected.rows, collected.embeddings)

    for i, cluster in enumerate(clusters):
        logger.info("Cluster %d: %d items", i, len(cluster))

    async with S3DataTool().dataset_generator() as dataset_generator:
        # Add rows to the dataset named "example_dataset"
        await dataset_generator.from_async_iterator(
            _get_cluster_iterator(clusters),
            name=args.dataset_name,
            batch=batch_name,
            streaming_configs=S3DataTool.StreamingConfigs(chunk_size=10),
            deduplicate_on=["text", "source_id"],  # list of columns
        )

    return clusters


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
