"""Sweep drop_frac, min_cluster_size, and outlier_threshold for clustering,
evaluating topic coherence via LLM-as-a-Judge.

Usage:
    uv run --env-file .env python scripts/analysis/tune_clustering.py \\
        --model-name $MODEL_NAME --limit 1

    uv run --env-file .env python scripts/analysis/tune_clustering.py \\
        --drop-frac 0.5,0.8,0.9 --min-cluster-size 2,3,5 \\
        --outlier-threshold 0.0,0.2,0.5 \\
        --sample-size 30 --max-concurrency 32
"""

import argparse
import asyncio
import itertools
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import openai
from scipy import stats as scipy_stats
from s3_data_tool import DataItem
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.cluster.clustering import (
    _collect,
    _group_batches_by_day,
    _list_batches,
    cluster_knn_graph_with_diagnostics,
    ClusterDiagnostics,
)
from scripts.analysis.tune_outlier_threshold import (
    _llm_judge_cluster,
    _load_template,
)
from scripts.analysis.cluster_eval_utils import (
    parse_grid,
    sample_clusters,
    build_results_df,
    plot_heatmap,
)

logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for clustering with LLM-as-a-Judge coherence evaluation",
    )
    parser.add_argument("--limit", type=int, default=1, help="Days to process (0=all)")
    parser.add_argument("--k", type=int, default=None, help="kNN neighbors (default: auto=ceil(log2(n)))")
    parser.add_argument("--drop-frac", default="0.0,0.5,0.8,0.9,0.95,0.99",
                        help="Comma-separated drop_frac values")
    parser.add_argument("--min-cluster-size", default="2,3,5,10",
                        help="Comma-separated min_cluster_size values")
    parser.add_argument("--max-cluster-size", type=int, default=50)
    parser.add_argument("--split-k-scale", type=float, default=0.5)
    parser.add_argument("--sample-size", type=int, default=20,
                        help="Clusters to sample per parameter combo for LLM evaluation")
    parser.add_argument("--model-name", default=None,
                        help="OpenAI model name (default: env MODEL_NAME)")
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--outlier-threshold", default="0.0",
                        help="Comma-separated outlier_threshold values")
    parser.add_argument("--n-judge-runs", type=int, default=1,
                        help="Number of independent sampling+judging runs per combo for confidence intervals")
    parser.add_argument("--output-dir", default="outputs/tune_clustering")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    drop_frac_values = parse_grid(args.drop_frac, float)
    min_cluster_size_values = parse_grid(args.min_cluster_size, int)
    outlier_threshold_values = parse_grid(args.outlier_threshold, float)

    for d in drop_frac_values:
        if not (0.0 <= d < 1.0):
            raise ValueError(f"drop_frac must be in [0, 1), got {d}")
    for m in min_cluster_size_values:
        if m < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {m}")
    for t in outlier_threshold_values:
        if not (0.0 <= t < 1.0):
            raise ValueError(f"outlier_threshold must be in [0, 1), got {t}")

    n_combos = len(drop_frac_values) * len(min_cluster_size_values) * len(outlier_threshold_values)
    logger.info("Grid: %d drop_frac x %d min_cluster_size x %d outlier_threshold = %d combinations",
                len(drop_frac_values), len(min_cluster_size_values),
                len(outlier_threshold_values), n_combos)

    # --- Collect data once across all days ---
    all_batches = await _list_batches()
    days_map = _group_batches_by_day(all_batches)
    sorted_days = sorted(days_map.keys())

    per_day_data: list[tuple[str, list[DataItem], list[np.ndarray], int]] = []
    # (day_str, rows, embeddings, k_value)

    processed = 0
    for day in sorted_days:
        if args.limit > 0 and processed >= args.limit:
            break
        collected = await _collect(batches=days_map[day])
        if not collected.rows:
            logger.info("Day %s: no embeddings found; skipping.", day)
            continue
        k_val = args.k or math.ceil(math.log2(len(collected.rows)))
        logger.info("Day %s: collected %d embeddings, k=%d.", day, len(collected.rows), k_val)
        per_day_data.append((day, collected.rows, collected.embeddings, k_val))
        processed += 1

    if not per_day_data:
        logger.error("No data collected. Check S3 connectivity and batch contents.")
        return

    # --- LLM setup ---
    model_name = args.model_name or os.environ["MODEL_NAME"]
    template = _load_template()
    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.max_concurrency)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSweeping {n_combos} combinations across {len(per_day_data)} day(s)...")
    print(f"LLM: {model_name}, concurrency={args.max_concurrency}, "
          f"sample_size={args.sample_size}, n_judge_runs={args.n_judge_runs}")

    # --- Sweep ---
    results: list[dict] = []

    param_grid = list(itertools.product(
        drop_frac_values, min_cluster_size_values, outlier_threshold_values,
    ))

    for combo_idx, (drop_frac, min_cluster_size, outlier_threshold) in enumerate(
        tqdm(param_grid, desc="Sweep", unit="combo"), start=1
    ):
        label = f"drop={drop_frac:.2f} min_size={min_cluster_size} thr={outlier_threshold:.2f}"
        logger.info("[%d/%d] %s", combo_idx, n_combos, label)

        # Re-cluster each day with this parameter combo
        all_clusters: list[list[DataItem]] = []
        all_diagnostics: list[ClusterDiagnostics] = []
        total_items = 0

        for day_str, rows, embeddings, k_val in per_day_data:
            clusters, diagnostics = cluster_knn_graph_with_diagnostics(
                rows, embeddings,
                k=k_val,
                drop_frac=drop_frac,
                min_cluster_size=min_cluster_size,
                max_cluster_size=args.max_cluster_size,
                split_k_scale=args.split_k_scale,
                outlier_threshold=outlier_threshold,
            )
            all_clusters.extend(clusters)
            all_diagnostics.append(diagnostics)
            total_items += diagnostics.total_items

        n_clusters = len(all_clusters)
        if n_clusters == 0:
            logger.warning("  -> 0 clusters produced; skipping LLM evaluation.")
            results.append({
                "drop_frac": drop_frac,
                "min_cluster_size": min_cluster_size,
                "outlier_threshold": outlier_threshold,
                "n_clusters": 0,
                "n_total_items": 0,
                "avg_cluster_size": 0,
                "std_cluster_size": 0,
                "n_sampled": 0,
                "n_judged": 0,
                "n_coherent": 0,
                "n_failed_llm": 0,
                "coherence_rate": None,
            })
            continue

        # --- Multi-run sampling + LLM judging ---
        run_samples: list[list[list[str]]] = []
        for run_idx in range(args.n_judge_runs):
            seed = args.seed + combo_idx + run_idx * 1000
            sampled = sample_clusters(all_clusters, args.sample_size, min_cluster_size, seed)
            if not sampled:
                logger.warning("  -> run %d: no clusters with non-empty text.", run_idx + 1)
            run_samples.append(sampled)

        # Build all LLM tasks across all runs: (run_idx, cluster_idx, coroutine)
        all_tasks = []
        for run_idx, samples in enumerate(run_samples):
            for ci, texts in enumerate(samples):
                all_tasks.append((
                    run_idx,
                    ci,
                    _llm_judge_cluster(client, model_name, texts, template, semaphore),
                ))

        if not all_tasks:
            logger.warning("  -> No samples with text across any run; skipping.")
            results.append({
                "drop_frac": drop_frac, "min_cluster_size": min_cluster_size,
                "outlier_threshold": outlier_threshold,
                "n_clusters": n_clusters, "n_total_items": total_items,
                "avg_cluster_size": 0, "std_cluster_size": 0,
                "n_sampled": 0, "n_judged": 0, "n_coherent": 0, "n_failed_llm": 0,
                "coherence_rate": None,
            })
            continue

        logger.info("  -> %d clusters, %d samples x %d runs = %d LLM calls.",
                    n_clusters, len(run_samples[0]) if run_samples[0] else 0,
                    args.n_judge_runs, len(all_tasks))

        # Execute all LLM calls concurrently
        judgments = await asyncio.gather(*[t[2] for t in all_tasks])

        # Group judgments by run
        per_run: dict[int, list[int | None]] = {i: [] for i in range(args.n_judge_runs)}
        for (run_idx, ci, _), judgment in zip(all_tasks, judgments):
            per_run[run_idx].append(judgment)

        # Compute per-run stats
        run_rates: list[float | None] = []
        total_judged = 0
        total_coherent = 0
        total_failed = 0
        total_noise = 0
        for run_idx in range(args.n_judge_runs):
            jlist = per_run[run_idx]
            n_co = sum(1 for j in jlist if j == 1)
            n_no = sum(1 for j in jlist if j == 0)
            n_ju = n_co + n_no
            n_fa = sum(1 for j in jlist if j is None)
            cr = n_co / n_ju * 100 if n_ju > 0 else None
            run_rates.append(cr)
            total_judged += n_ju
            total_coherent += n_co
            total_noise += n_no
            total_failed += n_fa

        valid_rates = [r for r in run_rates if r is not None]
        mean_cr = float(np.mean(valid_rates)) if valid_rates else None
        std_cr = float(np.std(valid_rates, ddof=1)) if len(valid_rates) >= 2 else None

        # t-distribution confidence interval on the mean
        n_runs_valid = len(valid_rates)
        sem_cr = None
        ci_low = None
        ci_high = None
        if std_cr is not None and n_runs_valid >= 2:
            sem_cr = std_cr / math.sqrt(n_runs_valid)
            t_crit = float(scipy_stats.t.ppf(0.975, df=n_runs_valid - 1))
            if mean_cr is not None:
                ci_low = round(max(0.0, mean_cr - t_crit * sem_cr), 2)
                ci_high = round(min(100.0, mean_cr + t_crit * sem_cr), 2)
            sem_cr = round(sem_cr, 2)

        # Cluster size distribution
        cluster_sizes = [len(c) for c in all_clusters]
        total_small = sum(1 for d in all_diagnostics for _ in d.small_clusters)

        result = {
            "drop_frac": drop_frac,
            "min_cluster_size": min_cluster_size,
            "outlier_threshold": outlier_threshold,
            "n_clusters": n_clusters,
            "n_total_items": total_items,
            "avg_cluster_size": round(float(np.mean(cluster_sizes)), 1) if cluster_sizes else 0,
            "std_cluster_size": round(float(np.std(cluster_sizes)), 1) if cluster_sizes else 0,
            "n_small_clusters": total_small,
            "n_judge_runs": args.n_judge_runs,
            "n_sampled_per_run": len(run_samples[0]) if run_samples[0] else 0,
            "n_judged": total_judged,
            "n_coherent": total_coherent,
            "n_noise": total_noise,
            "n_failed_llm": total_failed,
            "coherence_rate": round(mean_cr, 2) if mean_cr is not None else None,
            "coherence_std": round(std_cr, 2) if std_cr is not None else None,
            "coherence_sem": sem_cr,
            "coherence_ci_low": ci_low,
            "coherence_ci_high": ci_high,
            "run_coherence_rates": [round(r, 2) if r is not None else None for r in run_rates],
            "size_summary": {
                "min": int(np.min(cluster_sizes)) if cluster_sizes else 0,
                "max": int(np.max(cluster_sizes)) if cluster_sizes else 0,
                "median": float(np.median(cluster_sizes)) if cluster_sizes else 0.0,
                "mean": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            },
        }

        results.append(result)
        cr_str = f"{mean_cr:.1f}%" if mean_cr is not None else "N/A"
        ci_str = f" (95% CI: {ci_low:.1f}–{ci_high:.1f}%)" if ci_low is not None else ""
        logger.info("  -> coherence_rate=%s%s (n=%d runs, %d judged)",
                    cr_str, ci_str, args.n_judge_runs, total_judged)

    # --- Report ---
    df = build_results_df(results)
    print()
    print(df.to_string())

    # Save CSV
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path)
    logger.info("Saved results table to %s", csv_path)

    # Save JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results to %s", results_path)

    # Heatmaps: slice by outlier_threshold for 2D visualization
    valid_results = [r for r in results if r.get("coherence_rate") is not None]
    has_ci = any(r.get("coherence_sem") is not None for r in valid_results)

    for thr in outlier_threshold_values:
        thr_tag = f"{thr:.2f}".replace(".", "")

        thr_valid = [r for r in valid_results if r.get("outlier_threshold") == thr]
        if thr_valid:
            plot_heatmap(
                thr_valid,
                x_field="drop_frac", y_field="min_cluster_size",
                value_field="coherence_rate",
                title=f"Cluster Topic Coherence Rate (outlier_threshold={thr:.2f})",
                output_path=output_dir / f"coherence_heatmap_thr{thr_tag}.png",
                vmin=0, vmax=100,
                std_field="coherence_sem" if has_ci else None,
            )

        thr_results = [r for r in results if r.get("outlier_threshold") == thr]
        if thr_results:
            plot_heatmap(
                thr_results,
                x_field="drop_frac", y_field="min_cluster_size",
                value_field="n_clusters",
                title=f"Number of Clusters (outlier_threshold={thr:.2f})",
                output_path=output_dir / f"cluster_count_heatmap_thr{thr_tag}.png",
                fmt="d", unit="",
                vmin=0, vmax=max(r["n_clusters"] for r in thr_results),
                cmap="viridis",
            )

    print(f"\nOutput saved to {output_dir}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(main())
