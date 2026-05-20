import argparse
import asyncio
import base64
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import AsyncIterator, Any

import faiss
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from s3_data_tool import S3DataTool, DataItem, RawDuckFilter
from s3_data_tool.s3_utils import enumerate_batches

logger = logging.getLogger(__name__)

_DAY_RE = re.compile(r"(\d{8})")


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


def _group_batches_by_day(batches: list[str]) -> dict[str, list[str]]:
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
    drop_frac: float = 0.9,
) -> csr_matrix:
    """Build a symmetric pruned kNN graph using cosine similarity (FAISS IndexFlatIP).

    Each row connects to its *k* nearest neighbors (excluding itself),
    then the farthest `drop_frac` fraction of edges are pruned.
    """
    if not (0.0 <= drop_frac < 1.0):
        raise ValueError("drop_frac must be in [0.0, 1.0)")

    n, dim = matrix.shape

    matrix = matrix.copy()
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    similarities, neighbors = index.search(matrix, k + 1)
    similarities = similarities[:, 1:]
    neighbors = neighbors[:, 1:]

    keep_k = max(1, int(np.ceil(k * (1.0 - drop_frac))))
    neighbors = neighbors[:, :keep_k]
    similarities = similarities[:, :keep_k]

    row_idx = np.repeat(np.arange(n), keep_k)
    col_idx = neighbors.ravel()
    data = similarities.ravel().astype(np.float32)

    graph = csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
    graph = graph.maximum(graph.T)
    graph.eliminate_zeros()
    return graph


def cluster_knn_graph(
    rows: list,
    embeddings: list[np.ndarray],
    k: int | None = None,
    drop_frac: float = 0.9,
) -> list[list[DataItem]]:
    """Cluster *rows* by building a kNN graph and extracting connected components.

    Returns clusters ordered largest-first.
    """
    if not rows:
        return []

    n = len(rows)

    if k is None:
        k = math.ceil(math.log2(n))
        logger.info("Auto-selected k=%d for n=%d embeddings", k, n)

    matrix = np.stack(embeddings).astype(np.float32)
    graph = _build_knn_graph(matrix, k=k, drop_frac=drop_frac)

    n_clusters, labels = connected_components(graph, directed=False)
    logger.info("Found %d clusters (k=%d, n=%d)", n_clusters, k, n)

    unique_labels, inverse = np.unique(labels, return_inverse=True)
    clusters: list[list] = [[] for _ in range(len(unique_labels))]
    for row, label_idx in zip(rows, inverse):
        clusters[label_idx].append(row)

    clusters.sort(key=len, reverse=True)
    return clusters


MAX_TEXT_LEN = 10_000


async def _get_cluster_iterator(
    clusters: list[list[DataItem]],
) -> AsyncIterator[dict[str, list[Any]]]:
    buffer: dict[str, list[Any]] = defaultdict(list)
    for cluster in clusters:
        for row in cluster:
            for _column, _value in row.data.items():
                if _column == "text" and isinstance(_value, str) and len(_value) > MAX_TEXT_LEN:
                    _value = _value[:MAX_TEXT_LEN]
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
    output_batch_name: str | None = None,
) -> list[list[DataItem]]:
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
    )

    for i, cluster in enumerate(clusters):
        logger.info("Day %s cluster %d: %d items", day, i, len(cluster))

    batch_name = output_batch_name or f"bsky-trending-{day}"
    async with S3DataTool().dataset_generator() as dataset_generator:
        await dataset_generator.from_async_iterator(
            _get_cluster_iterator(clusters),
            name=dataset_name,
            batch=batch_name,
            streaming_configs=S3DataTool.StreamingConfigs(chunk_size=50),
            deduplicate_on=["text", "source_id"],
        )

    logger.info("Day %s: uploaded %d clusters as batch %r.", day, len(clusters), batch_name)
    return clusters


async def main() -> list[list]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1, help="Max days to process (default: 1, 0=all)")
    parser.add_argument("--dataset-name", default="posts_clustered_simplified_001")
    parser.add_argument("--k", type=int, default=None, help="kNN neighbours per node (default: log2(n))")
    parser.add_argument("--drop-frac", type=float, default=0.95, help="Fraction of farthest edges to prune (default: 0.95)")
    parser.add_argument("--batch", default=None, help="Process only this specific batch name (overrides --limit).")
    args = parser.parse_args()

    if args.batch:
        m = _DAY_RE.search(args.batch)
        day = m.group(1) if m else "unknown"
        output_batch_name = f"x-posts-clusters-{day}"
        all_clusters = await _process_day(
            day, [args.batch], args.dataset_name,
            k=args.k, drop_frac=args.drop_frac,
            output_batch_name=output_batch_name,
        )
        return all_clusters

    all_batches = await _list_batches()
    days = _group_batches_by_day(all_batches)
    logger.info("Found %d batches across %d days.", len(all_batches), len(days))

    sorted_days = sorted(days.keys())

    all_clusters: list[list] = []
    processed = 0
    for day in sorted_days:
        if args.limit > 0 and processed >= args.limit:
            break
        clusters = await _process_day(
            day, days[day], args.dataset_name,
            k=args.k, drop_frac=args.drop_frac,
        )
        if clusters:
            processed += 1
            all_clusters.extend(clusters)

    return all_clusters


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
