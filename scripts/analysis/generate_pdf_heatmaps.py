"""One-off: read results.csv from the current directory and produce PDF heatmaps.

Usage (run from within an output folder):
    python generate_pdf_heatmaps.py

Or point it at a specific directory:
    python generate_pdf_heatmaps.py --input-dir outputs/tune_clustering
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_heatmap(
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
    fig.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close(fig)


def _has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PDF heatmaps from results.csv"
    )
    parser.add_argument(
        "--input-dir", default=".",
        help="Directory containing results.csv (default: current directory)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    csv_path = input_dir / "results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Convert to list of dicts
    results: list[dict] = df.to_dict("records")

    # Detect outlier_threshold column
    has_thr = _has_col(df, "outlier_threshold")
    has_ci = _has_col(df, "coherence_ci_low") and _has_col(df, "coherence_ci_high")

    # If CI bounds are present, compute a half-width for ± annotation
    if has_ci:
        for r in results:
            low = r.get("coherence_ci_low")
            high = r.get("coherence_ci_high")
            if low is not None and high is not None and r.get("coherence_rate") is not None:
                r["ci_halfwidth"] = round((float(high) - float(low)) / 2, 1)
            else:
                r["ci_halfwidth"] = None

    # Determine threshold values to iterate over
    if has_thr:
        thr_values = sorted(set(r["outlier_threshold"] for r in results))
        print(f"Detected outlier_threshold values: {thr_values}")
    else:
        thr_values = [None]
        print("No outlier_threshold column; producing single set of heatmaps.")

    # Filter to rows with a coherence_rate
    valid = [r for r in results if r.get("coherence_rate") is not None]

    for thr in thr_values:
        if has_thr:
            thr_tag = f"{float(thr):.2f}".replace(".", "")
            thr_valid = [r for r in valid if r.get("outlier_threshold") == thr]
            thr_all = [r for r in results if r.get("outlier_threshold") == thr]
            title_suffix = f" (outlier_threshold={float(thr):.2f})"
            fname_suffix = f"_thr{thr_tag}"
        else:
            thr_valid = valid
            thr_all = results
            title_suffix = ""
            fname_suffix = ""

        if thr_valid:
            _plot_heatmap(
                thr_valid,
                x_field="drop_frac", y_field="min_cluster_size",
                value_field="coherence_rate",
                title=f"Cluster Topic Coherence Rate{title_suffix}",
                output_path=input_dir / f"coherence_heatmap{fname_suffix}.pdf",
                vmin=0, vmax=100,
                std_field="ci_halfwidth" if has_ci else None,
            )

        if thr_all:
            cluster_counts = [r["n_clusters"] for r in thr_all if r.get("n_clusters") is not None]
            _plot_heatmap(
                thr_all,
                x_field="drop_frac", y_field="min_cluster_size",
                value_field="n_clusters",
                title=f"Number of Clusters{title_suffix}",
                output_path=input_dir / f"cluster_count_heatmap{fname_suffix}.pdf",
                fmt="d", unit="",
                vmin=0, vmax=max(cluster_counts) if cluster_counts else 1,
                cmap="viridis",
            )

    print("Done.")


if __name__ == "__main__":
    main()
