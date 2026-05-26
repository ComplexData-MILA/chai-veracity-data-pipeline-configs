"""Merge per-dataset comparison results into a single combined report.

Reads individual result.json files from subdirectories (one per dataset, as
produced by compare_dataset_quality.py in single-dataset mode) and produces
the combined table, CSV, JSON, and stacked bar chart.

Usage:
    uv run python scripts/analysis/merge_dataset_comparisons.py \
        --input-dir outputs/compare_dataset_quality_split \
        --output-dir outputs/compare_dataset_quality_merged
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _mean_ci(values: list[float]) -> tuple[float, float, float, float, float]:
    """Return (mean, std, sem, ci_low, ci_high) for a list of per-run percentages."""
    from math import sqrt
    from scipy import stats as scipy_stats

    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(values))
    if n >= 2:
        std = float(np.std(values, ddof=1))
        sem = std / sqrt(n)
        t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
        ci_low = max(0.0, mean - t_crit * sem)
        ci_high = min(100.0, mean + t_crit * sem)
    else:
        std = float("nan")
        sem = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    return mean, std, sem, ci_low, ci_high


def _print_table(results: list[dict]) -> pd.DataFrame:
    """Build DataFrame and print to stdout."""
    rows = []
    for r in results:
        row = {
            "dataset": r["dataset"],
            "n": r["n_examples"],
            "judged": r["total_judged"],
            "failed": r["total_failed"],
            "class_0_%": r["pct_0_mean"],
            "class_0_CI": (
                f"[{r['pct_0_ci_low']}, {r['pct_0_ci_high']}]"
                if r["pct_0_ci_low"] is not None else "-"
            ),
            "class_1_%": r["pct_1_mean"],
            "class_1_CI": (
                f"[{r['pct_1_ci_low']}, {r['pct_1_ci_high']}]"
                if r["pct_1_ci_low"] is not None else "-"
            ),
            "class_2_%": r["pct_2_mean"],
            "class_2_CI": (
                f"[{r['pct_2_ci_low']}, {r['pct_2_ci_high']}]"
                if r["pct_2_ci_low"] is not None else "-"
            ),
            "feasible_%": r["pct_feasible_mean"],
            "feasible_CI": (
                f"[{r['pct_feasible_ci_low']}, {r['pct_feasible_ci_high']}]"
                if r["pct_feasible_ci_low"] is not None else "-"
            ),
        }
        rows.append(row)
    df = pd.DataFrame(rows)

    print()
    print("=" * 100)
    print("Dataset Quality Comparison — LLM-as-a-Judge Feasibility Labels")
    print("=" * 100)
    print(df.to_string(index=False))
    print()
    return df


def _plot(results: list[dict], output_dir: Path) -> None:
    """Plot stacked bar chart of feasibility class distribution per dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))

    dataset_names = [r["dataset"] for r in results]
    x = np.arange(len(dataset_names))
    width = 0.55

    pct_1 = [r["pct_1_mean"] for r in results]
    pct_2 = [r["pct_2_mean"] for r in results]
    n_judge_runs = results[0].get("n_judge_runs", 1) if results else 1

    ax.bar(x, pct_1, width, label="Class 1 (ambiguous)", color="#7CB5EC")
    ax.bar(x, pct_2, width, bottom=pct_1, label="Class 2 (clear)", color="#2E75B6")

    feasible_means = [r["pct_feasible_mean"] for r in results]
    yerr_low = []
    yerr_high = []
    for r in results:
        ci_low = r.get("pct_feasible_ci_low")
        ci_high = r.get("pct_feasible_ci_high")
        mean = r["pct_feasible_mean"]
        if ci_low is not None and ci_high is not None:
            yerr_low.append(mean - ci_low)
            yerr_high.append(ci_high - mean)
        else:
            yerr_low.append(0)
            yerr_high.append(0)

    ax.errorbar(
        x, feasible_means,
        yerr=[yerr_low, yerr_high],
        fmt="none", ecolor="black", capsize=6, capthick=1.5, linewidth=1.5,
        label=f"95% CI (t-dist, n={n_judge_runs} runs)",
    )

    for i, (m1, m2, mf) in enumerate(zip(pct_1, pct_2, feasible_means)):
        if m1 is not None and m1 > 6:
            ax.text(i, m1 / 2, f"{m1:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        if m2 is not None and m2 > 6:
            ax.text(i, m1 + m2 / 2, f"{m2:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        r = results[i]
        ci_low = r.get("pct_feasible_ci_low")
        ci_high = r.get("pct_feasible_ci_high")
        ci_str = ""
        if ci_low is not None and ci_high is not None:
            ci_str = f"  [{ci_low:.1f}, {ci_high:.1f}]"
        ax.text(i, mf + 2.5, f"{mf:.1f}%{ci_str}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, fontsize=10)
    ax.set_ylabel("Percentage of examples (%)", fontsize=12)
    ax.set_title(
        "Feasibility Distribution by Dataset\n(LLM-as-a-Judge feasibility filter)",
        fontsize=13,
    )
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = output_dir / "feasibility_comparison.png"
    fig.savefig(plot_path, dpi=150)
    logger.info("Saved plot to %s", plot_path)
    pdf_path = output_dir / "feasibility_comparison.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    logger.info("Saved PDF to %s", pdf_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-dataset comparison results into a combined report",
    )
    parser.add_argument(
        "--input-dir", default="outputs/compare_dataset_quality_split",
        help="Directory containing per-dataset result subdirectories",
    )
    parser.add_argument(
        "--output-dir", default="outputs/compare_dataset_quality_merged",
        help="Output directory for merged CSV, JSON, and plot",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-dataset results
    results = []
    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        json_path = subdir / "results.json"
        if not json_path.exists():
            logger.warning("No results.json in %s, skipping", subdir)
            continue
        with open(json_path) as f:
            dataset_results = json.load(f)
        if isinstance(dataset_results, list) and len(dataset_results) == 1:
            results.append(dataset_results[0])
        elif isinstance(dataset_results, list):
            results.extend(dataset_results)
        else:
            results.append(dataset_results)
        logger.info("Loaded %s", json_path)

    if not results:
        raise ValueError(f"No result.json files found under {input_dir}")

    # Recompute CIs in case per-run data is available (merging across runs)
    _recompute_cis(results)

    # Print table
    df = _print_table(results)

    # Save CSV
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV to %s", csv_path)

    # Save JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved JSON to %s", json_path)

    # Plot
    _plot(results, output_dir)

    print(f"Merged output saved to {output_dir}/")


def _recompute_cis(results: list[dict]) -> None:
    """Recompute CIs from per-run data (handles single-run results gracefully)."""
    for r in results:
        if r.get("pct_feasible_mean") is not None:
            continue  # Already computed
        runs_0 = r.get("run_pct_0", [])
        runs_1 = r.get("run_pct_1", [])
        runs_2 = r.get("run_pct_2", [])
        runs_feas = r.get("run_pct_feasible", [])
        for key, runs in [("0", runs_0), ("1", runs_1), ("2", runs_2), ("feasible", runs_feas)]:
            mean, std, sem, ci_low, ci_high = _mean_ci(runs)
            r[f"pct_{key}_mean"] = round(mean, 2) if not (mean != mean) else None
            r[f"pct_{key}_std"] = round(std, 2) if not (std != std) else None
            r[f"pct_{key}_ci_low"] = round(ci_low, 2) if not (ci_low != ci_low) else None
            r[f"pct_{key}_ci_high"] = round(ci_high, 2) if not (ci_high != ci_high) else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
