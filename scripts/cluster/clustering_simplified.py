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

import faiss
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from trafilatura.deduplication import Simhash

from s3_data_tool import S3DataTool, DataItem, RawDuckFilter
from s3_data_tool.s3_utils import enumerate_batches

logger = logging.getLogger(__name__)

_DAY_RE = re.compile(r"(\d{8})")

# Simhash is 64 bits; use the top 16 bits as a single LSH band.
_LSH_BAND_BITS = 16
_LSH_BAND_MASK = (1 << _LSH_BAND_BITS) - 1

MAX_TEXT_LEN = 10_000


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
        base_columns=["text", "at_uri"],
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
    """Select up to *sample_size* elements that maximize embedding diversity."""
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
#  DBSCAN clustering via FAISS HNSW approximate search
# ---------------------------------------------------------------------------

def cluster_dbscan(
    rows: list,
    embeddings: list[np.ndarray],
    eps: float = 0.7,
    min_samples: int = 5,
    min_cluster_size: int = 50,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 300,
) -> list[list[DataItem]]:
    """DBSCAN via FAISS HNSW approximate nearest-neighbor search.

    Uses k-NN search with threshold filtering (FAISS 1.13.2 batch *range_search*
    is buggy).  Core points (>= *min_samples* neighbours within *eps* inner
    product) form clusters via connected components on the symmetric core-core
    graph.  Border points are assigned to the nearest core cluster.  Noise
    points are discarded.

    Returns clusters ordered largest-first.
    """
    if not rows:
        return []

    n = len(rows)
    dim = 128

    # L2-normalize so inner product = cosine similarity.
    matrix = np.stack(embeddings).astype(np.float32)
    matrix_norm = matrix.copy()
    faiss.normalize_L2(matrix_norm)

    # Build HNSW index (METRIC_INNER_PRODUCT matches IndexFlatIP semantics).
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(matrix_norm)

    # Adaptive k — enough headroom for dense regions, bounded for memory.
    k_search = min(500, max(200, n // 1000))
    logger.info(
        "HNSW k-NN search: k=%d, efSearch=%d, M=%d, n=%d",
        k_search, ef_search, M, n,
    )

    similarities, neighbors = index.search(matrix_norm, k_search)
    # similarities: inner-products (n, k_search); neighbors: indices (n, k_search)

    # --- core / border / noise classification --------------------------------
    query_idx = np.arange(n)[:, None]          # (n, 1)
    valid = (similarities >= eps) & (neighbors != query_idx)

    neighbor_counts = valid.sum(axis=1)
    core_mask = neighbor_counts >= min_samples

    n_core = int(core_mask.sum())
    logger.info(
        "Core: %d/%d (%.1f%%), non-core: %d (%.1f%%)",
        n_core, n, 100.0 * n_core / n,
        n - n_core, 100.0 * (n - n_core) / n,
    )

    if n_core < 2:
        logger.warning("Fewer than 2 core points; returning empty clusters.")
        return []

    # --- build symmetric sparse graph on core-point *subgraph* ----------------
    core_indices = np.where(core_mask)[0]          # int64 indices in [0, n)
    core_to_sub = {int(old): new for new, old in enumerate(core_indices)}

    edge_mask = valid & core_mask[neighbors]       # (n, k_search) bool
    rows_coo, cols_slot = np.where(edge_mask)      # flat indices
    cols_coo = neighbors[rows_coo, cols_slot]      # neighbor indices in [0, n)
    vals_coo = similarities[rows_coo, cols_slot]

    # Keep only core ↔ core edges.
    both_core = core_mask[rows_coo] & core_mask[cols_coo]
    rows_coo = rows_coo[both_core]
    cols_coo = cols_coo[both_core]
    vals_coo = vals_coo[both_core]

    rows_sub = np.array([core_to_sub[int(r)] for r in rows_coo], dtype=np.int32)
    cols_sub = np.array([core_to_sub[int(c)] for c in cols_coo], dtype=np.int32)

    graph = csr_matrix(
        (vals_coo, (rows_sub, cols_sub)),
        shape=(n_core, n_core),
        dtype=np.float32,
    )
    graph = graph.maximum(graph.T)                 # symmetrize
    graph.eliminate_zeros()

    logger.info(
        "Core graph: %d nodes, %d edges (avg degree %.1f)",
        n_core, graph.nnz, graph.nnz / n_core,
    )

    # --- connected components on core subgraph --------------------------------
    n_components, sub_labels = connected_components(graph, directed=False)
    unique_labels = np.unique(sub_labels)
    n_real_components = len(unique_labels)

    # Build clusters from core points; record label for border assignment.
    clusters: list[list] = [[] for _ in range(n_real_components)]
    label_to_cidx = {int(lab): i for i, lab in enumerate(unique_labels)}
    core_to_cluster_label: dict[int, int] = {}
    for sub_idx, core_idx in enumerate(core_indices):
        lab = int(sub_labels[sub_idx])
        cidx = label_to_cidx[lab]
        clusters[cidx].append(rows[core_idx])
        core_to_cluster_label[int(core_idx)] = lab

    # --- assign border points to nearest core cluster -------------------------
    border_mask = ~core_mask & (neighbor_counts > 0)
    n_border_candidates = int(border_mask.sum())
    n_assigned = 0
    for i in np.where(border_mask)[0]:
        i = int(i)
        vi = np.where(valid[i])[0]
        if len(vi) == 0:
            continue
        # Prefer the most-similar core neighbor.
        order = vi[np.argsort(-similarities[i][vi])]
        for s_idx in order:
            nb = int(neighbors[i, s_idx])
            if core_mask[nb]:
                lab = core_to_cluster_label[nb]
                clusters[label_to_cidx[lab]].append(rows[i])
                n_assigned += 1
                break
        # else: true noise → discarded

    n_noise = n_border_candidates - n_assigned + int((~core_mask & (neighbor_counts == 0)).sum())
    logger.info(
        "Components: %d, border assigned: %d, noise discarded: %d",
        n_real_components, n_assigned, n_noise,
    )

    # --- filter by min cluster size, then sort largest-first and return --------
    clusters = [c for c in clusters if len(c) >= min_cluster_size]
    n_dropped = n_real_components - len(clusters)
    if n_dropped:
        logger.info("Dropped %d clusters below min_cluster_size=%d.", n_dropped, min_cluster_size)
    clusters.sort(key=len, reverse=True)
    return clusters


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
    """Yield one dict per cluster with full cluster data plus central / diverse samples."""
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

        buffer: dict[str, list[Any]] = {}

        buffer["cluster_size"] = [n]
        buffer["central_sample_texts"] = [r.data.get("text", "") for r in central_rows]
        buffer["central_sample_ids"] = [r.id for r in central_rows]
        buffer["diverse_sample_texts"] = [r.data.get("text", "") for r in diverse_rows]
        buffer["diverse_sample_ids"] = [r.id for r in diverse_rows]
        buffer["central_sample_at_uris"] = [r.data.get("at_uri", "") for r in central_rows]
        buffer["diverse_sample_at_uris"] = [r.data.get("at_uri", "") for r in diverse_rows]

        yield buffer


# ---------------------------------------------------------------------------
#  Per-day processing
# ---------------------------------------------------------------------------

async def _process_day(
    day: str,
    day_batches: list[str],
    dataset_name: str,
    eps: float = 0.7,
    min_samples: int = 5,
    min_cluster_size: int = 50,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 300,
    sample_size: int = 50,
    collect_limit: int = 0,
    output_batch_name: str | None = None,
    skip_upload: bool = False,
) -> list[list[DataItem]]:
    logger.info(
        "Day %s: processing %d batches (%s ... %s)",
        day, len(day_batches), day_batches[0], day_batches[-1],
    )

    collected = await _collect(batches=day_batches, collect_limit=collect_limit)

    if not collected.rows:
        logger.warning("Day %s: no embeddings found; skipping.", day)
        return []

    logger.info("Day %s: collected %d embeddings.", day, len(collected.rows))

    clusters = cluster_dbscan(
        collected.rows, collected.embeddings,
        eps=eps, min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        M=M, ef_construction=ef_construction, ef_search=ef_search,
    )

    if not clusters:
        logger.warning("Day %s: no clusters formed (all points discarded as noise).", day)
        return []

    # Compute centroids as cluster means and build index map for sampling.
    embeddings_matrix = np.stack(collected.embeddings).astype(np.float32)
    orig_idx_map = {row.id: i for i, row in enumerate(collected.rows)}
    centroids = []
    for cluster in clusters:
        indices = [orig_idx_map[row.id] for row in cluster]
        centroids.append(embeddings_matrix[indices].mean(axis=0))

    for i, cluster in enumerate(clusters):
        logger.info(
            "Day %s cluster %d: %d items (central=%d, diverse=%d)",
            day, i, len(cluster),
            min(sample_size, len(cluster)),
            min(sample_size, len(cluster)),
        )

    batch_name = output_batch_name or f"bsky-trending-{day}"

    if skip_upload:
        async for _ in _get_cluster_iterator(
            clusters, centroids, embeddings_matrix, orig_idx_map, sample_size,
        ):
            pass
        logger.info("Day %s: processed %d clusters (upload skipped).", day, len(clusters))
    else:
        async with S3DataTool().dataset_generator() as dataset_generator:
            await dataset_generator.from_async_iterator(
                _get_cluster_iterator(
                    clusters, centroids, embeddings_matrix, orig_idx_map, sample_size,
                ),
                name=dataset_name,
                batch=batch_name,
                streaming_configs=S3DataTool.StreamingConfigs(chunk_size=500),
            )
        logger.info("Day %s: uploaded %d clusters as batch %r.", day, len(clusters), batch_name)

    return clusters


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

async def main() -> list[list]:
    parser = argparse.ArgumentParser(
        description="DBSCAN clustering via FAISS HNSW with per-cluster subsampling"
    )
    parser.add_argument("--limit", type=int, default=1, help="Max days to process (default: 1, 0=all)")
    parser.add_argument("--dataset-name", default="posts_clustered_dbscan_001")
    parser.add_argument("--eps", type=float, default=0.9, help="Inner-product threshold (cosine similarity) for DBSCAN (default: 0.9)")
    parser.add_argument("--min-samples", type=int, default=100, help="Minimum neighbours for core points (default: 100)")
    parser.add_argument("--min-cluster-size", type=int, default=50, help="Drop clusters with fewer items than this (default: 50)")
    parser.add_argument("--M", type=int, default=32, help="HNSW out-degree (default: 32)")
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW build-time search width (default: 200)")
    parser.add_argument("--ef-search", type=int, default=300, help="HNSW query-time search width (default: 300)")
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
        output_batch_name = args.output_batch_name or f"x-posts-clusters-{day}"
        all_clusters = await _process_day(
            day, [args.batch], args.dataset_name,
            eps=args.eps, min_samples=args.min_samples,
            min_cluster_size=args.min_cluster_size,
            M=args.M, ef_construction=args.ef_construction,
            ef_search=args.ef_search,
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
            day, days[day], args.dataset_name,
            eps=args.eps, min_samples=args.min_samples,
            min_cluster_size=args.min_cluster_size,
            M=args.M, ef_construction=args.ef_construction,
            ef_search=args.ef_search,
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
    peak_mb = usage.ru_maxrss / 1024.0  # KB -> MB on Linux
    logger.info("Peak RSS: %.1f MB", peak_mb)
