"""Cluster embedded claims using kNN-graph connected components.

Reads embedded claims from S3, clusters them using the same algorithm as the
first-level pipeline, and uploads meta-clusters back to S3.

Usage:
    uv run --env-file .env python scripts/cluster/cluster_claims.py \\
        --drop-frac 0.9 --min-cluster-size 2

    # For double-clustering: aim for ~10 clusters
    uv run --env-file .env python scripts/cluster/cluster_claims.py \\
        --drop-frac 0.8 --min-cluster-size 3 --outlier-threshold 0.1
"""

import argparse
import asyncio
import base64
import logging
import math
from collections import defaultdict
from typing import AsyncIterator

import numpy as np

from s3_data_tool import S3DataTool, DataItem

from scripts.cluster.clustering import (
    cluster_knn_graph,
    cluster_knn_graph_with_diagnostics,
    ClusterDiagnostics,
)

logger = logging.getLogger(__name__)


def _decode_embedding(encoded: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float32)


async def _collect_embedded_claims(
    source_dataset: str,
    batches: list[str] | None = None,
) -> tuple[list[DataItem], list[np.ndarray]]:
    """Read embedded claims from S3 and return rows + embeddings."""
    rows: list[DataItem] = []
    embeddings: list[np.ndarray] = []

    async with S3DataTool().filter_for_export(
        name=source_dataset,
        base_columns=["claim", "claim_id", "cluster_id", "original_ids",
                       "original_texts", "post_count", "embedding"],
    ) as generator:
        async for row in generator._iter_filtered_items(batches=batches):
            encoded = row.data.get("embedding")
            if not encoded:
                continue
            rows.append(row)
            embeddings.append(_decode_embedding(encoded))

    logger.info("Read %d embedded claims from %s.", len(rows), source_dataset)
    return rows, embeddings


async def _get_meta_cluster_iterator(
    clusters: list[list[DataItem]],
) -> AsyncIterator[dict]:
    """Yield one dict per meta-cluster with claims stacked into lists."""
    for cluster in clusters:
        out: dict = defaultdict(list)
        for row in cluster:
            for col, val in row.data.items():
                if col == "embedding":
                    continue  # drop embedding from output
                out[col].append(val)

        out["total_post_count"] = sum(out.get("post_count", []))
        out["num_claims"] = len(cluster)
        yield dict(out)


async def _process(
    source_dataset: str,
    target_dataset: str,
    k: int | None = None,
    drop_frac: float = 0.9,
    min_cluster_size: int = 0,
    max_cluster_size: int = 0,
    split_k_scale: float = 0.5,
    outlier_threshold: float = 0.0,
    merge_threshold: float = 0.0,
) -> list[list[DataItem]]:
    """Collect, cluster, and upload embedded claims."""
    rows, embeddings = await _collect_embedded_claims(source_dataset)

    if not rows:
        logger.warning("No embedded claims found in %s.", source_dataset)
        return []

    n = len(rows)
    if k is None:
        k = math.ceil(math.log2(n))
        logger.info("Auto-selected k=%d for n=%d claims.", k, n)

    clusters = cluster_knn_graph(
        rows, embeddings,
        k=k, drop_frac=drop_frac,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        split_k_scale=split_k_scale,
        outlier_threshold=outlier_threshold,
        merge_threshold=merge_threshold,
    )

    logger.info("Produced %d meta-clusters from %d claims.", len(clusters), n)
    for i, c in enumerate(clusters[:10]):
        logger.info("  Cluster %d: %d claims", i, len(c))
    if len(clusters) > 10:
        logger.info("  ... and %d more clusters", len(clusters) - 10)

    async with S3DataTool().dataset_generator() as dataset_generator:
        await dataset_generator.from_async_iterator(
            _get_meta_cluster_iterator(clusters),
            name=target_dataset,
            batch="bsky-meta-clusters-001",
            streaming_configs=S3DataTool.StreamingConfigs(chunk_size=100),
        )

    logger.info("Uploaded %d meta-clusters to %s.", len(clusters), target_dataset)
    return clusters


async def main():
    parser = argparse.ArgumentParser(description="Cluster embedded claims")
    parser.add_argument("--k", type=int, default=None,
                        help="kNN neighbours per node (default: log2(n))")
    parser.add_argument("--drop-frac", type=float, default=0.9,
                        help="Fraction of farthest edges to prune (default: 0.9)")
    parser.add_argument("--min-cluster-size", type=int, default=2,
                        help="Merge clusters smaller than this (default: 2)")
    parser.add_argument("--max-cluster-size", type=int, default=0,
                        help="Split clusters larger than this (default: 0 = disabled)")
    parser.add_argument("--split-k-scale", type=float, default=0.5,
                        help="Scale factor for k when sub-clustering")
    parser.add_argument("--outlier-threshold", type=float, default=0.0,
                        help="Cosine similarity floor for outlier detection")
    parser.add_argument("--merge-threshold", type=float, default=0.0,
                        help="Cosine similarity above which any two cluster centroids are merged (default: 0.0 = disabled)")
    parser.add_argument("--source-dataset", default="posts_claims_embedded_001")
    parser.add_argument("--target-dataset", default="posts_clustered_meta_001")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Also run with diagnostics and print summary")
    args = parser.parse_args()

    if args.diagnostics:
        rows, embeddings = await _collect_embedded_claims(args.source_dataset)
        if not rows:
            logger.warning("No embedded claims found.")
            return
        n = len(rows)
        k = args.k or math.ceil(math.log2(n))
        clusters, diag = cluster_knn_graph_with_diagnostics(
            rows, embeddings,
            k=k, drop_frac=args.drop_frac,
            min_cluster_size=args.min_cluster_size,
            max_cluster_size=args.max_cluster_size,
            split_k_scale=args.split_k_scale,
            outlier_threshold=args.outlier_threshold,
            merge_threshold=args.merge_threshold,
        )
        print(f"\nDiagnostics: {diag.n_initial} initial -> {diag.n_after_split} after split "
              f"-> {diag.n_after_merge} after merge, {diag.total_items} total items")
        print(f"Final clusters: {len(clusters)}")
        sizes = sorted([len(c) for c in clusters], reverse=True)
        print(f"Sizes: {sizes[:20]}{'...' if len(sizes) > 20 else ''}")
        if diag.small_clusters:
            sims = sorted([s.max_sim for s in diag.small_clusters])
            print(f"Small cluster sims: {[round(s, 3) for s in sims[:10]]}")
    else:
        await _process(
            source_dataset=args.source_dataset,
            target_dataset=args.target_dataset,
            k=args.k,
            drop_frac=args.drop_frac,
            min_cluster_size=args.min_cluster_size,
            max_cluster_size=args.max_cluster_size,
            split_k_scale=args.split_k_scale,
            outlier_threshold=args.outlier_threshold,
            merge_threshold=args.merge_threshold,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
