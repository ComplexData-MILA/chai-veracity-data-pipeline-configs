"""Shared utilities for clustering evaluation sweep scripts.

Extracted from tune_clustering.py to be reused by tune_double_clustering.py.
"""

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from s3_data_tool import DataItem


def parse_grid(arg_str: str, cls: type) -> list:
    return [cls(v.strip()) for v in arg_str.split(",")]


def sample_clusters(
    clusters: list[list[DataItem]],
    n_sample: int,
    min_cluster_size: int,
    seed: int = 42,
    text_field: str = "text",
) -> list[list[str]]:
    """Sample clusters stratified by size for LLM evaluation.

    Returns up to *n_sample* clusters as lists of text strings.
    Clusters with no non-empty text are excluded.
    """
    # Build (texts, size) for each cluster with at least one non-empty text
    valid: list[list[str]] = []
    for cluster in clusters:
        texts = [
            row.data.get(text_field, "") or ""
            for row in cluster
        ]
        texts = [t for t in texts if t.strip()]
        if not texts:
            continue
        valid.append(texts)

    if not valid:
        return []

    if len(valid) <= n_sample:
        return [t[:32] for t in valid]

    rng = random.Random(seed)

    # Stratify into three size bins
    small_bin: list[list[str]] = []   # [min_cluster_size, min_cluster_size + 2]
    medium_bin: list[list[str]] = []  # (min_cluster_size + 2, 10)
    large_bin: list[list[str]] = []   # >= 10

    small_max = min_cluster_size + 2
    for texts in valid:
        sz = len(texts)
        if sz <= small_max:
            small_bin.append(texts)
        elif sz < 10:
            medium_bin.append(texts)
        else:
            large_bin.append(texts)

    bins = [small_bin, medium_bin, large_bin]

    # Proportional allocation
    total = len(valid)
    allocated = []
    remaining = n_sample
    for bin_clusters in bins:
        n = max(1, round(n_sample * len(bin_clusters) / total)) if bin_clusters else 0
        n = min(n, len(bin_clusters), remaining)
        allocated.append(n)
        remaining -= n

    # Distribute leftover to largest bin with spare capacity
    if remaining > 0:
        for i in sorted(range(len(bins)), key=lambda i: len(bins[i]), reverse=True):
            available = len(bins[i]) - allocated[i]
            take = min(remaining, available)
            allocated[i] += take
            remaining -= take
            if remaining == 0:
                break

    sampled: list[list[str]] = []
    for bin_clusters, n_alloc in zip(bins, allocated):
        if n_alloc > 0 and bin_clusters:
            chosen = rng.sample(bin_clusters, min(n_alloc, len(bin_clusters)))
            sampled.extend(chosen)

    return [t[:32] for t in sampled]


def build_results_df(results: list[dict]) -> pd.DataFrame:
    """Build a pandas DataFrame from results, sorted by coherence_rate descending."""
    has_ci = any(r.get("coherence_ci_low") is not None for r in results)
    rows = []
    for r in results:
        row = {
            "drop_frac": r["drop_frac"],
            "min_cluster_size": r["min_cluster_size"],
            "outlier_threshold": r["outlier_threshold"],
            "n_clusters": r["n_clusters"],
            "avg_cluster_size": r.get("avg_cluster_size", 0),
            "std_cluster_size": r.get("std_cluster_size", 0),
            "n_sampled": r.get("n_sampled_per_run", r.get("n_sampled", 0)),
            "n_judged": r["n_judged"],
            "n_coherent": r["n_coherent"],
            "n_noise": r.get("n_noise", 0),
            "n_failed_llm": r["n_failed_llm"],
            "coherence_rate": r.get("coherence_rate"),
        }
        if has_ci:
            row["coherence_ci_low"] = r.get("coherence_ci_low")
            row["coherence_ci_high"] = r.get("coherence_ci_high")
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values("coherence_rate", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    return df


def plot_heatmap(
    results: list[dict],
    x_field: str,
    y_field: str,
    value_field: str,
    title: str,
    output_path: Path,
    fmt: str = ".1f",
    cmap: str = "RdYlGn",
    vmin: float = 0,
    vmax: float = 100,
    unit: str = "%",
    std_field: str | None = None,
    std_label: str = "sem",
) -> None:
    x_vals = sorted(set(r[x_field] for r in results))
    y_vals = sorted(set(r[y_field] for r in results))

    lookup = {(r[x_field], r[y_field]): r[value_field] for r in results}
    matrix = np.full((len(y_vals), len(x_vals)), np.nan)
    for j, x in enumerate(x_vals):
        for i, y in enumerate(y_vals):
            matrix[i, j] = lookup.get((x, y), np.nan)

    # Build ± annotation lookup if requested
    ann_lookup: dict[tuple, float] = {}
    if std_field:
        for r in results:
            key = (r[x_field], r[y_field])
            std_val = r.get(std_field)
            if std_val is not None:
                ann_lookup[key] = std_val

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(x_vals)))
    ax.set_yticks(range(len(y_vals)))
    ax.set_xticklabels([str(x) for x in x_vals])
    ax.set_yticklabels([str(y) for y in y_vals])
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.set_title(title)

    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            val = matrix[i, j]
            if not np.isnan(val):
                disp = f"{int(val):{fmt}}" if fmt == "d" else f"{val:{fmt}}"
                key = (x_vals[j], y_vals[i])
                std_val = ann_lookup.get(key) if std_field else None
                if std_val is not None:
                    disp += f"\n±{std_val:.1f}{unit}"
                ax.text(j, i, disp, ha="center", va="center",
                        color="black" if 30 < val < 70 else "white",
                        fontsize=8)

    plt.colorbar(im, ax=ax, label=value_field)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
