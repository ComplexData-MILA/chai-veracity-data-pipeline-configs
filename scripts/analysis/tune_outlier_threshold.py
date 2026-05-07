"""Select optimal outlier_threshold for clustering.

Two modes:
  manual — export small-cluster candidates as JSONL for human review
  llm    — sample clusters near threshold boundaries, ask LLM to judge
           whether each is a coherent topic or noise, and report noise-rate
           estimates per threshold
  full   — run both modes

Usage:
    uv run --env-file .env python scripts/analysis/tune_outlier_threshold.py \\
        --mode manual --limit 1

    uv run --env-file .env python scripts/analysis/tune_outlier_threshold.py \\
        --mode llm --limit 1 --thresholds 0.2,0.3,0.4 --model-name $MODEL_NAME
"""

import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openai

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

logger = logging.getLogger(__name__)


def _load_template() -> str:
    template_path = Path(_project_root) / "templates" / "judge_cluster_coherence.txt"
    with open(template_path) as f:
        return f.read()


def _format_prompt(template: str, texts: list[str]) -> str:
    """Format the judge template with numbered posts."""
    posts = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))
    return template.format(posts=posts)


async def _collect_candidates(
    limit: int,
    k: int | None,
    drop_frac: float,
    min_cluster_size: int,
    max_cluster_size: int,
    split_k_scale: float,
) -> tuple[list[dict], ClusterDiagnostics]:
    """Collect one day, cluster with outlier_threshold=0, return texts + diag.

    Returns:
        List of dicts, one per small cluster, each with:
            max_sim, size, nearest_large_size, texts
    """
    all_batches = await _list_batches()
    days = _group_batches_by_day(all_batches)
    sorted_days = sorted(days.keys())

    candidates: list[dict] = []
    diagnostics = None

    processed = 0
    for day in sorted_days:
        if limit > 0 and processed >= limit:
            break

        collected = await _collect(batches=days[day])
        if not collected.rows:
            continue

        logger.info("Day %s: collected %d embeddings.", day, len(collected.rows))

        if k is None:
            k = math.ceil(math.log2(len(collected.rows)))
            logger.info("k not specified, using ceil(log2(n)) = %d.", k)

        clusters, diagnostics = cluster_knn_graph_with_diagnostics(
            collected.rows,
            collected.embeddings,
            k=k,
            drop_frac=drop_frac,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            split_k_scale=split_k_scale,
            outlier_threshold=0.0,  # merge all, drop none
        )

        # Build all-texts lookup from original rows (row index → text)
        all_texts_arr = [
            row.data.get("text", "") or "" for row in collected.rows
        ]

        # Build candidates from diagnostics
        for sim_info in diagnostics.small_clusters:
            texts = [
                all_texts_arr[idx] for idx in sim_info.row_indices
                if all_texts_arr[idx].strip()
            ]
            candidates.append({
                "size": sim_info.size,
                "max_sim": round(sim_info.max_sim, 4),
                "nearest_large_size": sim_info.nearest_large_size,
                "texts": texts,
            })

        processed += 1
        break  # one day is enough for tuning

    if diagnostics is None:
        raise RuntimeError("No data collected. Check S3 connectivity and batch contents.")

    return candidates, diagnostics


def _simulate_thresholds(
    diagnostics: ClusterDiagnostics,
    thresholds: list[float],
) -> dict[float, dict]:
    """For each threshold, count how many clusters/items would be dropped."""
    result = {}
    small = diagnostics.small_clusters
    for t in sorted(thresholds):
        dropped = [s for s in small if s.max_sim < t]
        n_clusters = len(dropped)
        n_items = sum(s.size for s in dropped)
        result[t] = {
            "n_dropped_clusters": n_clusters,
            "n_dropped_items": n_items,
            "pct_items": n_items / diagnostics.total_items * 100 if diagnostics.total_items else 0,
        }
    return result


def _export_manual(
    candidates: list[dict],
    diagnostics: ClusterDiagnostics,
    output_path: Path,
) -> None:
    """Write small-cluster candidates sorted by max_sim ascending."""
    if not candidates:
        logger.info("No small clusters to export.")
        return

    # Sort by max_sim ascending (worst outliers first)
    entries = sorted(candidates, key=lambda e: e["max_sim"])
    for i, e in enumerate(entries):
        e["rank"] = i

    with open(output_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Summary table
    print(f"\nWrote {len(entries)} candidates to {output_path}")
    print(f"\n{'Threshold':>10} {'Clusters':>10} {'Items':>10} {'% of Total':>12}")
    print("-" * 46)
    for t, stats in _simulate_thresholds(
        diagnostics, [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
    ).items():
        print(
            f"{t:>10.2f} {stats['n_dropped_clusters']:>10} "
            f"{stats['n_dropped_items']:>10} {stats['pct_items']:>11.1f}%",
        )
    print()


async def _llm_judge_cluster(
    client: openai.AsyncOpenAI,
    model_name: str,
    texts: list[str],
    template: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> int | None:
    """Ask LLM: is this cluster coherent (1) or noise (0)? Returns None on failure."""
    # Truncate to avoid token limits
    truncated = texts[:32]
    prompt = _format_prompt(template, truncated)

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1024,
                )
                content = response.choices[0].message.content
                if not content:
                    continue
                # Parse: split on last |, extract 0 or 1
                parts = content.rsplit("|", 1)
                if len(parts) == 2:
                    import re
                    m = re.search(r"\b([01])\b", parts[1])
                    if m:
                        return int(m.group(1))
                # Fallback: search whole content
                import re
                m = re.search(r"\b([01])\b", content)
                if m:
                    return int(m.group(1))
            except Exception:
                await asyncio.sleep(2 ** attempt)
    return None


async def _llm_evaluate(
    candidates: list[dict],
    diagnostics: ClusterDiagnostics,
    thresholds: list[float],
    model_name: str,
    max_concurrency: int,
    sample_size: int,
    output_dir: Path,
) -> dict[float, dict]:
    """LLM-judge clusters near each threshold boundary and report noise rates."""
    template = _load_template()
    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrency)

    # Build per-threshold samples: clusters in [threshold - 0.05, threshold + 0.05]
    print(f"\nLLM Judge ({model_name}, concurrency={max_concurrency})")
    print(f"{'Threshold':>10} {'Sampled':>8} {'Noise':>8} {'Noise%':>8} {'Est FP Items':>14}")
    print("-" * 54)

    results: dict[float, dict] = {}
    for t in sorted(thresholds):
        # Candidates near this threshold
        near = [(i, c) for i, c in enumerate(candidates)
                if abs(c["max_sim"] - t) <= 0.05]

        # Take up to sample_size
        if len(near) > sample_size:
            below = [(i, c) for i, c in near if c["max_sim"] < t]
            above = [(i, c) for i, c in near if c["max_sim"] >= t]
            n_below = min(sample_size // 2, len(below))
            n_above = sample_size - n_below
            if n_above > len(above):
                n_below = sample_size - len(above)
            rng = random.Random(18)
            sampled = rng.sample(below, n_below) + rng.sample(above, n_above)
        else:
            sampled = near

        # Run LLM judgments
        tasks = [
            _llm_judge_cluster(client, model_name, c["texts"], template, semaphore)
            for _, c in sampled
        ]
        judgments = await asyncio.gather(*tasks)

        n_noise = sum(1 for j in judgments if j == 0)
        n_coherent = sum(1 for j in judgments if j == 1)
        n_total = n_noise + n_coherent
        noise_pct = n_noise / n_total * 100 if n_total > 0 else 0

        # Estimate false positives at this threshold
        dropped_at_t = [c for c in candidates if c["max_sim"] < t]
        n_dropped_items = sum(c["size"] for c in dropped_at_t)
        est_fp = (1 - noise_pct / 100) * n_dropped_items if noise_pct > 0 else 0

        results[t] = {
            "sampled": n_total,
            "n_noise": n_noise,
            "noise_pct": noise_pct,
            "n_dropped_items": n_dropped_items,
            "est_fp_items": int(est_fp),
        }

        print(
            f"{t:>10.2f} {n_total:>8} {n_noise:>8} {noise_pct:>7.1f}% {int(est_fp):>14}",
        )

    print()

    results_path = output_dir / "llm_judge_results.json"
    serializable = {str(t): v for t, v in results.items()}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved LLM judge results to %s", results_path)

    return results


def _plot_llm_results(
    results: dict[float, dict],
    output_dir: Path,
) -> None:
    thresholds = sorted(results.keys())
    labels = [f"{t:.2f}" for t in thresholds]
    noise_pcts = [results[t]["noise_pct"] for t in thresholds]
    n_noises = [results[t]["n_noise"] for t in thresholds]
    n_sampleds = [results[t]["sampled"] for t in thresholds]
    n_dropped = [results[t]["n_dropped_items"] for t in thresholds]
    est_fps = [results[t]["est_fp_items"] for t in thresholds]

    x = np.arange(len(thresholds))
    width = 0.5

    # Chart A: Noise rate
    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    bars = ax_a.bar(x, noise_pcts, width, color="steelblue")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_xlabel("Outlier Threshold")
    ax_a.set_ylabel("Noise Rate (%)")
    ax_a.set_title("LLM Judge: Noise Rate by Threshold")
    ax_a.set_ylim(0, max(noise_pcts) * 1.25 if noise_pcts and max(noise_pcts) > 0 else 100)
    for bar, n_noise, n_sampled in zip(bars, n_noises, n_sampleds):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{n_noise}/{n_sampled}", ha="center", va="bottom", fontsize=9,
        )
    fig_a.tight_layout()
    fig_a.savefig(output_dir / "llm_noise_rate.png", dpi=150)
    logger.info("Saved noise rate chart to %s", output_dir / "llm_noise_rate.png")
    plt.close(fig_a)

    # Chart B: Dropped items impact
    fig_b, ax_b = plt.subplots(figsize=(8, 5))
    bar_width = 0.35
    ax_b.bar(x - bar_width / 2, n_dropped, bar_width, color="darkorange", label="Total dropped items")
    ax_b.bar(x + bar_width / 2, est_fps, bar_width, color="crimson", label="Est. false positives (coherent dropped)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_xlabel("Outlier Threshold")
    ax_b.set_ylabel("Number of Items")
    ax_b.set_title("Impact of Threshold on Dropped Items")
    ax_b.legend(fontsize=9)
    fig_b.tight_layout()
    fig_b.savefig(output_dir / "llm_false_positive_impact.png", dpi=150)
    logger.info("Saved false positive impact chart to %s", output_dir / "llm_false_positive_impact.png")
    plt.close(fig_b)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Tune outlier_threshold for clustering")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--drop-frac", type=float, default=0.95)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--max-cluster-size", type=int, default=50)
    parser.add_argument("--split-k-scale", type=float, default=0.5)
    parser.add_argument("--mode", choices=["manual", "llm", "full"], default="manual")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--output-dir", default="output/")
    args = parser.parse_args()

    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    print("Collecting candidates...")
    candidates, diagnostics = await _collect_candidates(
        limit=args.limit,
        k=args.k,
        drop_frac=args.drop_frac,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        split_k_scale=args.split_k_scale,
    )

    print(
        f"Diagnostics: {diagnostics.n_initial} initial → "
        f"{diagnostics.n_after_split} after split → "
        f"{diagnostics.n_after_merge} after merge, "
        f"{diagnostics.total_items} total items, "
        f"{len(candidates)} small clusters",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("manual", "full"):
        output_path = output_dir / "outlier_candidates.jsonl"
        _export_manual(candidates, diagnostics, output_path)

    if args.mode in ("llm", "full"):
        model_name = args.model_name or os.environ["MODEL_NAME"]
        results = await _llm_evaluate(
            candidates, diagnostics, thresholds,
            model_name, args.max_concurrency, args.sample_size,
            output_dir,
        )
        _plot_llm_results(results, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
