"""Generate a feasibility comparison PDF from saved result JSON files.

Reads one or more results.json files (from compare_dataset_quality.py or
merge_dataset_comparisons.py) and produces a stacked bar chart as PDF.

Usage:
    # From a single-dataset result:
    uv run python scripts/analysis/plot_from_json.py \
        outputs/compare_dataset_quality_split/fever/results.json

    # From merged results:
    uv run python scripts/analysis/plot_from_json.py \
        outputs/compare_dataset_quality_split_merged/results.json

    # From multiple per-dataset files:
    uv run python scripts/analysis/plot_from_json.py \
        outputs/compare_dataset_quality_split/*/results.json \
        -o outputs/my_plot.pdf
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _plot(results: list[dict], output_path: Path) -> None:
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
    fig.savefig(output_path, bbox_inches="tight")
    logger.info("Saved PDF to %s", output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate feasibility comparison PDF from saved JSON results",
    )
    parser.add_argument(
        "json_files", nargs="+", type=Path,
        help="One or more results.json files to plot",
    )
    parser.add_argument(
        "-o", "--output", default=None, type=Path,
        help="Output PDF path (default: feasibility_comparison.pdf in cwd)",
    )
    args = parser.parse_args()

    results: list[dict] = []
    for path in args.json_files:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)
        logger.info("Loaded %s", path)

    if not results:
        raise ValueError("No results loaded")

    output_path = args.output or Path("feasibility_comparison.pdf")
    _plot(results, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
