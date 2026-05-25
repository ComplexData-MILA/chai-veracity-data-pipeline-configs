#!/usr/bin/env python3
"""
Offline DBSCAN clustering via FAISS HNSW — reads local parquet files, writes local JSONL.

Replaces the S3-dependent _collect() with a local DuckDB reader, and _process_day()
output with a local JSONL writer.  All clustering/sampling logic is unchanged from
clustering_simplified.py.

Usage:
  python clustering_offline.py \\
    --data-dir /path/to/data \\
    --output-dir /path/to/output \\
    --day 20260427 \\
    --dataset-name posts_clustered_dbscan_005_b \\
    --sample-size 50 --min-cluster-size 50
"""

import argparse
import asyncio
import base64
import json
import logging
import re
import resource
import uuid
from collections import defaultdict
from pathlib import Path

import duckdb
import faiss
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from trafilatura.deduplication import Simhash

logger = logging.getLogger(__name__)

_DAY_RE = re.compile(r"(\d{8})")
_LSH_BAND_BITS = 16
_LSH_BAND_MASK = (1 << _LSH_BAND_BITS) - 1


# ---------------------------------------------------------------------------
#  Local parquet listing
# ---------------------------------------------------------------------------

def list_batches(data_dir: Path) -> list[str]:
    posts_dir = data_dir / "datasets" / "posts"
    batches = set()
    for parquet_path in posts_dir.glob("bsky-*/merged.parquet"):
        batch_name = parquet_path.parent.name
        batches.add(batch_name)
    return sorted(batches)


def group_batches_by_day(batches: list[str]) -> dict[str, list[str]]:
    days: dict[str, list[str]] = defaultdict(list)
    for batch in batches:
        if m := _DAY_RE.search(batch):
            days[m.group(1)].append(batch)
        else:
            logger.warning("Could not parse date from batch name %r; skipping.", batch)
    return dict(days)


# ---------------------------------------------------------------------------
#  Local DuckDB reader (replaces S3DataTool.filter_for_export)
# ---------------------------------------------------------------------------

def _collect_local(
    data_dir: Path,
    batches: list[str],
    collect_limit: int = 0,
) -> tuple[list[dict], list[np.ndarray]]:
    """Collect rows with embeddings from local parquet files via DuckDB.

    Each returned row dict has keys: id, text, at_uri, _batch, embedding, _emb_idx.
    The _emb_idx field maps the row to its index in the embeddings list.
    Returns (rows, embeddings) in 1:1 correspondence.
    """
    posts_dir = data_dir / "datasets" / "posts"
    annot_dir = posts_dir / "annotations" / "embeddings_128d"

    base_paths = []
    annot_paths = []
    for batch in batches:
        bp = posts_dir / batch / "merged.parquet"
        if bp.exists():
            base_paths.append(str(bp))
        ap = annot_dir / batch / "merged.parquet"
        if ap.exists():
            annot_paths.append(str(ap))

    if not base_paths:
        logger.warning("No base parquet files found for batches.")
        return [], []

    base_paths_str = ", ".join(f"'{p}'" for p in base_paths)
    annot_paths_str = ", ".join(f"'{p}'" for p in annot_paths)

    limit_clause = f"LIMIT {collect_limit}" if collect_limit > 0 else ""

    query = f"""
        WITH
        base_filtered AS (
            SELECT id, ANY_VALUE(text) AS text, ANY_VALUE(at_uri) AS at_uri, ANY_VALUE(_batch) AS _batch
            FROM read_parquet([{base_paths_str}])
            GROUP BY id
        ),
        embeddings_128d_filtered AS (
            SELECT id, ANY_VALUE(embedding) AS embedding
            FROM read_parquet([{annot_paths_str}])
            WHERE embedding IS NOT NULL
            GROUP BY id
        )
        SELECT base_filtered.*, embeddings_128d_filtered.embedding
        FROM base_filtered
        JOIN embeddings_128d_filtered USING (id)
        {limit_clause}
    """

    logger.info("Running DuckDB query on %d base + %d annotation paths...",
                len(base_paths), len(annot_paths))

    rows: list[dict] = []
    embeddings: list[np.ndarray] = []

    conn = duckdb.connect()
    try:
        result = conn.execute(query)
        col_names = [d[0] for d in result.description]
        while True:
            chunk = result.fetchmany(1000)
            if not chunk:
                break
            for row in chunk:
                row_dict = dict(zip(col_names, row))
                emb_str = row_dict.pop("embedding", None)
                if not emb_str:
                    continue
                idx = len(rows)
                row_dict["_emb_idx"] = idx
                rows.append(row_dict)
                embeddings.append(_decode_embedding(emb_str))
    finally:
        conn.close()

    logger.info("Query returned %d rows with embeddings (limit=%s).",
                len(rows), collect_limit or "none")
    return rows, embeddings


def _decode_embedding(encoded: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float32)


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
    bucket_best: dict[int, int] = {}
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
) -> list[list[dict]]:
    if not rows:
        return []

    n = len(rows)
    dim = 128

    matrix = np.stack(embeddings).astype(np.float32)
    matrix_norm = matrix.copy()
    faiss.normalize_L2(matrix_norm)

    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(matrix_norm)

    k_search = min(500, max(200, n // 1000))
    logger.info("HNSW k-NN search: k=%d, efSearch=%d, M=%d, n=%d",
                k_search, ef_search, M, n)

    similarities, neighbors = index.search(matrix_norm, k_search)

    query_idx = np.arange(n)[:, None]
    valid = (similarities >= eps) & (neighbors != query_idx)

    neighbor_counts = valid.sum(axis=1)
    core_mask = neighbor_counts >= min_samples

    n_core = int(core_mask.sum())
    logger.info("Core: %d/%d (%.1f%%), non-core: %d (%.1f%%)",
                n_core, n, 100.0 * n_core / n,
                n - n_core, 100.0 * (n - n_core) / n)

    if n_core < 2:
        logger.warning("Fewer than 2 core points; returning empty clusters.")
        return []

    core_indices = np.where(core_mask)[0]
    core_to_sub = {int(old): new for new, old in enumerate(core_indices)}

    edge_mask = valid & core_mask[neighbors]
    rows_coo, cols_slot = np.where(edge_mask)
    cols_coo = neighbors[rows_coo, cols_slot]
    vals_coo = similarities[rows_coo, cols_slot]

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
    graph = graph.maximum(graph.T)
    graph.eliminate_zeros()

    logger.info("Core graph: %d nodes, %d edges (avg degree %.1f)",
                n_core, graph.nnz, graph.nnz / n_core)

    n_components, sub_labels = connected_components(graph, directed=False)
    unique_labels = np.unique(sub_labels)
    n_real_components = len(unique_labels)

    clusters: list[list] = [[] for _ in range(n_real_components)]
    label_to_cidx = {int(lab): i for i, lab in enumerate(unique_labels)}
    core_to_cluster_label: dict[int, int] = {}
    for sub_idx, core_idx in enumerate(core_indices):
        lab = int(sub_labels[sub_idx])
        cidx = label_to_cidx[lab]
        clusters[cidx].append(rows[core_idx])
        core_to_cluster_label[int(core_idx)] = lab

    border_mask = ~core_mask & (neighbor_counts > 0)
    n_border_candidates = int(border_mask.sum())
    n_assigned = 0
    for i in np.where(border_mask)[0]:
        i = int(i)
        vi = np.where(valid[i])[0]
        if len(vi) == 0:
            continue
        order = vi[np.argsort(-similarities[i][vi])]
        for s_idx in order:
            nb = int(neighbors[i, s_idx])
            if core_mask[nb]:
                lab = core_to_cluster_label[nb]
                clusters[label_to_cidx[lab]].append(rows[i])
                n_assigned += 1
                break

    n_noise = n_border_candidates - n_assigned + int((~core_mask & (neighbor_counts == 0)).sum())
    logger.info("Components: %d, border assigned: %d, noise discarded: %d",
                n_real_components, n_assigned, n_noise)

    clusters = [c for c in clusters if len(c) >= min_cluster_size]
    n_dropped = n_real_components - len(clusters)
    if n_dropped:
        logger.info("Dropped %d clusters below min_cluster_size=%d.", n_dropped, min_cluster_size)
    clusters.sort(key=len, reverse=True)
    return clusters


# ---------------------------------------------------------------------------
#  Local JSONL writer (replaces S3DataTool.dataset_generator)
# ---------------------------------------------------------------------------

def _write_clusters_jsonl(
    clusters: list[list[dict]],
    embeddings_matrix: np.ndarray,
    centroids: list[np.ndarray],
    sample_size: int,
    output_dir: Path,
    batch_name: str,
) -> int:
    """Write clusters as Newline-Delimited JSON to output_dir.

    Uses _emb_idx stored in each row dict to look up embeddings.
    Returns total rows written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    chunk_size = 500
    chunk_idx = 0
    total_rows = 0
    row_count = 0

    def _open_next_chunk() -> tuple:
        path = output_dir / f"{batch_name}_{run_id}_chunk_{chunk_idx:04d}.jsonl"
        return open(path, "w"), path

    fh, _ = _open_next_chunk()

    for ci, cluster in enumerate(clusters):
        n = len(cluster)
        emb_indices = [row["_emb_idx"] for row in cluster]
        cluster_embs = embeddings_matrix[emb_indices]
        centroid = centroids[ci]
        texts = [row.get("text", "") for row in cluster]

        central_idx = _sample_central(texts, cluster_embs, centroid, sample_size)
        diverse_idx = _sample_diverse(cluster_embs, centroid, sample_size)

        record = {
            "id": uuid.uuid4().hex[:16],
            "_batch": batch_name,
            "cluster_size": n,
            "central_sample_texts": [cluster[i].get("text", "") for i in central_idx],
            "central_sample_ids": [cluster[i].get("id", "") for i in central_idx],
            "central_sample_at_uris": [cluster[i].get("at_uri", "") for i in central_idx],
            "diverse_sample_texts": [cluster[i].get("text", "") for i in diverse_idx],
            "diverse_sample_ids": [cluster[i].get("id", "") for i in diverse_idx],
            "diverse_sample_at_uris": [cluster[i].get("at_uri", "") for i in diverse_idx],
        }

        fh.write(json.dumps(record) + "\n")
        row_count += 1
        total_rows += 1

        if row_count >= chunk_size:
            fh.close()
            chunk_idx += 1
            fh, _ = _open_next_chunk()
            row_count = 0

    fh.close()
    # Remove trailing empty chunk if present
    last_chunk = output_dir / f"{batch_name}_{run_id}_chunk_{chunk_idx:04d}.jsonl"
    if last_chunk.exists() and last_chunk.stat().st_size == 0:
        last_chunk.unlink()

    logger.info("Wrote %d cluster rows to %d chunks in %s",
                total_rows, chunk_idx + (0 if row_count == 0 else 1), output_dir)
    return total_rows


# ---------------------------------------------------------------------------
#  Per-day processing
# ---------------------------------------------------------------------------

async def process_day(
    data_dir: Path,
    output_dir: Path,
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
) -> int:
    logger.info("Day %s: processing %d batches (%s ... %s)",
                day, len(day_batches), day_batches[0], day_batches[-1])

    rows, embeddings = _collect_local(data_dir, day_batches, collect_limit)

    if not rows:
        logger.warning("Day %s: no embeddings found; skipping.", day)
        return 0

    logger.info("Day %s: collected %d embeddings.", day, len(rows))

    clusters = cluster_dbscan(
        rows, embeddings,
        eps=eps, min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        M=M, ef_construction=ef_construction,
        ef_search=ef_search,
    )

    if not clusters:
        logger.warning("Day %s: no clusters formed.", day)
        return 0

    embeddings_matrix = np.stack(embeddings).astype(np.float32)
    centroids = []
    for cluster in clusters:
        indices = [row["_emb_idx"] for row in cluster]
        centroids.append(embeddings_matrix[indices].mean(axis=0))

    for i, cluster in enumerate(clusters):
        logger.info("Day %s cluster %d: %d items (central=%d, diverse=%d)",
                    day, i, len(cluster),
                    min(sample_size, len(cluster)),
                    min(sample_size, len(cluster)))

    batch_name = f"bsky-trending-{day}"
    n_written = _write_clusters_jsonl(
        clusters, embeddings_matrix, centroids,
        sample_size, output_dir, batch_name,
    )

    logger.info("Day %s: wrote %d cluster records.", day, n_written)
    return n_written


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline DBSCAN clustering — local parquet in, JSONL out"
    )
    parser.add_argument("--data-dir", required=True, help="Local data root directory")
    parser.add_argument("--output-dir", required=True, help="Local output directory for JSONL")
    parser.add_argument("--dataset-name", default="posts_clustered_dbscan_005_b")
    parser.add_argument("--eps", type=float, default=0.9)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=300)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--collect-limit", type=int, default=0)
    parser.add_argument("--day", required=True, help="Day to process (YYYYMMDD)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not data_dir.exists():
        logger.error("Data directory %s does not exist.", data_dir)
        return

    batches = list_batches(data_dir)
    days = group_batches_by_day(batches)
    logger.info("Found %d batches across %d days.", len(batches), len(days))

    if args.day not in days:
        logger.error("Day %s not found. Available: %s", args.day, sorted(days.keys()))
        return

    day_batches = days[args.day]
    await process_day(
        data_dir, output_dir, args.day, day_batches, args.dataset_name,
        eps=args.eps, min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
        M=args.M, ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        sample_size=args.sample_size,
        collect_limit=args.collect_limit,
    )

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_mb = usage.ru_maxrss / 1024.0
    logger.info("Peak RSS: %.1f MB", peak_mb)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
