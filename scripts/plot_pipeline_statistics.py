"""Plot per-day row counts across data-pipeline stages.

Usage:
    uv run --env-file .env python scripts/plot_pipeline_statistics.py [--regenerate]

Produces PDF + PNG + JSON in outputs/dataset_statistics/.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.environ["S3_BUCKET"]
PREFIX = os.environ.get("S3_PREFIX", "datasets")
OUTDIR = "outputs/dataset_statistics"

COLORS = {
    "raw": "#1A3A5C",
    "annotated": "#2E75B6",
    "clustered": "#7CB5EC",
    "summarized": "#2E75B6",
    "outliers_dropped": "#999999",
}

matplotlib.use("Agg")


# ── Date helpers ───────────────────────────────────────────────────────────

def parse_date(day_str: str) -> datetime:
    return datetime.strptime(day_str, "%Y%m%d")


def day_label(day_str: str) -> str:
    """Only label Sundays so the x-axis isn't crowded."""
    dt = parse_date(day_str)
    dow = dt.strftime("%a").upper()
    if dow == "SUN":
        return f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}\n({dow})"
    return ""


# ── DuckDB ─────────────────────────────────────────────────────────────────

def duckdb_conn():
    import duckdb

    conn = duckdb.connect()
    s3_endpoint = os.environ["S3_ENDPOINT_URL"]
    host = s3_endpoint.removeprefix("https://").rstrip("/").removeprefix("http://").rstrip("/")
    use_ssl = s3_endpoint.startswith("https://")
    conn.execute(f"""
        SET s3_access_key_id='{os.environ["S3_ACCESS_KEY"]}';
        SET s3_secret_access_key='{os.environ["S3_SECRET_KEY"]}';
        SET s3_endpoint='{host}';
        SET s3_use_ssl={str(use_ssl).lower()};
        SET s3_url_style='path';
    """)
    return conn


# ── Path-based date extraction (datasets 1–3) ──────────────────────────────

def _extract_day_generic(path: str, prefix: str) -> str:
    for part in path.split("/"):
        if part.startswith(prefix):
            return part[len(prefix):][:8]
    return "unknown"


def extract_day_posts(path: str) -> str:
    return _extract_day_generic(path, "bsky-jetstream-")


def extract_day_clustering(path: str) -> str:
    return _extract_day_generic(path, "bsky-trending-")


# ── Data collection ────────────────────────────────────────────────────────

def _count_rows(paths: list[str], conn) -> int:
    if not paths:
        return 0
    paths_sql = ", ".join(f"'{p}'" for p in paths)
    result = conn.execute(
        f"SELECT count(*) FROM read_parquet([{paths_sql}])"
    ).fetchone()
    return result[0] if result else 0


def collect_posts(s3_client, dataset: str) -> list[dict]:
    """Collect per-day row counts for raw posts (dataset 1)."""
    sys.path.insert(0, "/mnt/s3-data-tool")
    from s3_data_tool.s3_utils import enumerate_parquet_paths_sync

    paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, dataset)
    by_day = defaultdict(list)
    for p in paths:
        by_day[extract_day_posts(p)].append(p)

    conn = duckdb_conn()
    result = []
    for day in sorted(by_day):
        result.append({"day": day, "count": _count_rows(by_day[day], conn)})
    conn.close()
    return result


def collect_annotated(s3_client, dataset: str, annotator: str) -> list[dict]:
    """Collect per-day row counts for annotated posts (dataset 2)."""
    sys.path.insert(0, "/mnt/s3-data-tool")
    from s3_data_tool.s3_utils import enumerate_parquet_paths_sync

    paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, dataset, annotator)
    by_day = defaultdict(list)
    for p in paths:
        by_day[extract_day_posts(p)].append(p)

    conn = duckdb_conn()
    result = []
    for day in sorted(by_day):
        result.append({"day": day, "count": _count_rows(by_day[day], conn)})
    conn.close()
    return result


def collect_clustered(s3_client, dataset: str) -> list[dict]:
    """Collect per-day row counts for clustering output (dataset 3)."""
    sys.path.insert(0, "/mnt/s3-data-tool")
    from s3_data_tool.s3_utils import enumerate_parquet_paths_sync

    paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, dataset)
    by_day = defaultdict(list)
    for p in paths:
        by_day[extract_day_clustering(p)].append(p)

    conn = duckdb_conn()
    result = []
    for day in sorted(by_day):
        result.append({"day": day, "count": _count_rows(by_day[day], conn)})
    conn.close()
    return result


def collect_summarized(s3_client, dataset: str) -> list[dict]:
    """Collect per-day row counts for summarization output (dataset 4).

    Uses the ``date`` column because a single summarization batch covers
    multiple source-data days.
    """
    sys.path.insert(0, "/mnt/s3-data-tool")
    from s3_data_tool.s3_utils import enumerate_parquet_paths_sync

    paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, dataset)
    if not paths:
        return []

    paths_sql = ", ".join(f"'{p}'" for p in paths)
    conn = duckdb_conn()
    rows = conn.execute(f"""
        SELECT REPLACE(date, '"', '') AS d, count(*) AS cnt
        FROM read_parquet([{paths_sql}])
        GROUP BY d
        ORDER BY d
    """).fetchall()
    conn.close()
    return [{"day": row[0], "count": row[1]} for row in rows]


def collect_outliers_dropped(s3_client, dataset: str, annotator: str,
                             clustering_dataset: str) -> list[dict]:
    """Per-day outliers dropped = label-2 annotations − sum of cluster_size."""
    sys.path.insert(0, "/mnt/s3-data-tool")
    from s3_data_tool.s3_utils import enumerate_parquet_paths_sync

    # Per-day label-2 counts from annotator
    annot_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, dataset, annotator)
    annot_by_day = defaultdict(list)
    for p in annot_paths:
        annot_by_day[extract_day_posts(p)].append(p)

    # Per-day cluster_size sums from clustering output
    clust_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, clustering_dataset)
    clust_by_day = defaultdict(list)
    for p in clust_paths:
        clust_by_day[extract_day_clustering(p)].append(p)

    conn = duckdb_conn()

    label2_by_day: dict[str, int] = {}
    for day in sorted(annot_by_day):
        paths_sql = ", ".join(f"'{p}'" for p in annot_by_day[day])
        row = conn.execute(f"""
            SELECT count(*) FROM read_parquet([{paths_sql}])
            WHERE json_extract_string(classifier_label, '$') = 'LABEL_2'
        """).fetchone()
        label2_by_day[day] = row[0] if row else 0

    cluster_size_by_day: dict[str, int] = {}
    for day in sorted(clust_by_day):
        paths_sql = ", ".join(f"'{p}'" for p in clust_by_day[day])
        row = conn.execute(f"""
            SELECT sum(CAST(TRIM(cluster_size, '[]') AS BIGINT))
            FROM read_parquet([{paths_sql}])
        """).fetchone()
        cluster_size_by_day[day] = row[0] if row and row[0] else 0

    conn.close()

    all_days = sorted(set(label2_by_day.keys()) | set(cluster_size_by_day.keys()))
    result = []
    for day in all_days:
        outliers = label2_by_day.get(day, 0) - cluster_size_by_day.get(day, 0)
        result.append({"day": day, "count": outliers})
    return result


def collect_all(s3_client, raw_dataset: str, annotator: str,
                clustering_dataset: str, summarization_dataset: str) -> dict:
    """Collect data for all five pipeline stages."""
    return {
        "raw": collect_posts(s3_client, raw_dataset),
        "annotated": collect_annotated(s3_client, raw_dataset, annotator),
        "clustered": collect_clustered(s3_client, clustering_dataset),
        "summarized": collect_summarized(s3_client, summarization_dataset),
        "outliers_dropped": collect_outliers_dropped(
            s3_client, raw_dataset, annotator, clustering_dataset,
        ),
    }


# ── Plot ───────────────────────────────────────────────────────────────────

def _scatter_lines(ax, x_vals: list[int], y_vals: list[int], color: str,
                   label: str):
    """Plot scatter points joined by lines."""
    if not x_vals:
        return

    ax.scatter(x_vals, y_vals, marker="o", s=48, facecolor="white",
               edgecolor=color, linewidths=1.0, zorder=10, label=label)

    for i in range(len(x_vals) - 1):
        ax.plot([x_vals[i], x_vals[i + 1]], [y_vals[i], y_vals[i + 1]],
                color=color, linewidth=1.0, alpha=0.8, zorder=9)


def _add_break_marks(ax_upper, ax_lower):
    """Draw slanted cut-out lines between the two subplots.

    Uses markers in Axes coordinates so the lines keep their angle
    independent of scales or figure size (matplotlib docs pattern).
    """
    d = 0.5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color="k", mec="k", mew=1, clip_on=False)
    ax_upper.plot([0, 1], [0, 0], transform=ax_upper.transAxes, **kwargs)
    ax_lower.plot([0, 1], [1, 1], transform=ax_lower.transAxes, **kwargs)


def _set_y_ticks_no_boundary(ax, is_upper: bool):
    """Set y-tick labels, blanking the one adjacent to the cut so labels
    from the two panels don't collide."""
    y_min, y_max = ax.get_ylim()
    ticks = ax.get_yticks()
    visible = [t for t in ticks if y_min < t < y_max]
    labels = []
    for t in visible:
        if t >= 1e6:
            s = f"{t / 1e6:.1f}".rstrip("0").rstrip(".")
            labels.append(f"{s}M")
        elif t >= 1e3:
            s = f"{t / 1e3:.1f}".rstrip("0").rstrip(".")
            labels.append(f"{s}K")
        else:
            labels.append(f"{t:.0f}")
    ax.set_yticks(visible)
    ax.set_yticklabels(labels)


def _auto_break_point(upper_vals: list[int], lower_vals: list[int]) -> tuple[float, float]:
    """Return (lower_top, upper_bottom) — independent y-limits per subplot."""
    max_lower = max(lower_vals) if lower_vals else 0
    min_upper = min((v for v in upper_vals if v > 0), default=0)
    max_upper = max(upper_vals) if upper_vals else 0

    if max_lower == 0:
        lower_top = 10
    else:
        lower_top = max_lower * 1.2

    if max_upper == 0:
        return lower_top, lower_top * 2

    if min_upper == 0:
        upper_bottom = lower_top * 1.5
    else:
        upper_bottom = min_upper * 0.70

    # Ensure a visible gap between the two subplots
    if upper_bottom <= lower_top:
        upper_bottom = lower_top * 1.5

    return lower_top, upper_bottom


def plot(all_data: dict):
    plt.rcParams.update({"font.size": 16})

    raw_data = all_data["raw"]
    annotated_data = all_data["annotated"]
    clustered_data = all_data["clustered"]
    summarized_data = all_data["summarized"]
    outliers_data = all_data.get("outliers_dropped", [])

    # Build unified x-axis from the union of all days
    all_days_set: set[str] = set()
    for d in raw_data:
        all_days_set.add(d["day"])
    for d in annotated_data:
        all_days_set.add(d["day"])
    for d in clustered_data:
        all_days_set.add(d["day"])
    for d in summarized_data:
        all_days_set.add(d["day"])
    all_days = sorted(all_days_set)

    x = list(range(len(all_days)))
    x_labels = [day_label(d) for d in all_days]

    def _lookup(data: list[dict]) -> dict[str, int]:
        return {d["day"]: d["count"] for d in data}

    raw_lu = _lookup(raw_data)
    annot_lu = _lookup(annotated_data)
    clust_lu = _lookup(clustered_data)
    summ_lu = _lookup(summarized_data)
    outlier_lu = _lookup(outliers_data)

    raw_y = [raw_lu.get(d, 0) for d in all_days]
    annot_y = [annot_lu.get(d, 0) for d in all_days]
    clust_y = [clust_lu.get(d, 0) for d in all_days]
    summ_y = [summ_lu.get(d, 0) for d in all_days]
    outlier_y = [outlier_lu.get(d, 0) for d in all_days]

    fig, (ax_upper, ax_lower) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 6),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05},
    )
    fig.patch.set_facecolor("white")

    # ── Upper subplot: raw + annotated posts + outliers dropped ──
    _scatter_lines(ax_upper, x, raw_y, COLORS["raw"], "Total")
    _scatter_lines(ax_upper, x, annot_y, COLORS["annotated"], "Annotated")
    _scatter_lines(ax_upper, x, outlier_y, COLORS["outliers_dropped"], "Outliers dropped")

    all_upper_vals = [v for v in raw_y + annot_y + outlier_y if v > 0]
    max_upper = max(all_upper_vals) if all_upper_vals else 1000

    # ── Lower subplot: clusters + summaries ──
    _scatter_lines(ax_lower, x, clust_y, COLORS["clustered"], "DBSCAN Clusters")

    bar_width = 0.55
    bar_x = [x[i] for i, v in enumerate(summ_y) if v > 0]
    bar_y_vals = [summ_y[i] for i, v in enumerate(summ_y) if v > 0]
    if bar_x:
        ax_lower.bar(bar_x, bar_y_vals, bar_width, color=COLORS["summarized"],
                     alpha=0.7, zorder=1, label="Claims Extracted")

    all_lower_vals = [v for v in clust_y + summ_y if v > 0]

    # ── Auto-compute independent y-limits per subplot ──
    lower_top, upper_bottom = _auto_break_point(all_upper_vals, all_lower_vals)
    ax_upper.set_ylim(upper_bottom, max_upper * 1.15)
    ax_lower.set_ylim(0, lower_top)

    # ── Axis labels ──
    fig.supylabel("Count", x=0.04)
    ax_upper.set_ylabel("")
    ax_lower.set_ylabel("")

    # ── X-axis ticks (on lower subplot) ──
    ax_upper.xaxis.tick_top()
    ax_upper.tick_params(labeltop=False)
    ax_lower.set_xticks(x)
    ax_lower.set_xticklabels(x_labels, linespacing=1.2)
    ax_lower.set_xlim(-0.6, len(x) - 0.4)

    # ── Hide inner spines & draw break marks ──
    ax_upper.spines["bottom"].set_visible(False)
    ax_lower.spines["top"].set_visible(False)
    ax_upper.tick_params(bottom=False)

    _add_break_marks(ax_upper, ax_lower)

    # ── Clean up y-tick labels at the boundary ──
    _set_y_ticks_no_boundary(ax_upper, is_upper=True)
    _set_y_ticks_no_boundary(ax_lower, is_upper=False)

    # ── Grid ──
    for ax_obj in (ax_upper, ax_lower):
        ax_obj.grid(axis="y", alpha=0.3)
        ax_obj.grid(axis="x", alpha=0.15)

    # ── Legend ──
    handles_upper, labels_upper = ax_upper.get_legend_handles_labels()
    handles_lower, labels_lower = ax_lower.get_legend_handles_labels()
    fig.legend(handles_upper + handles_lower, labels_upper + labels_lower,
               loc="upper center", ncol=4, frameon=True,
               fontsize=14, framealpha=0.9, bbox_to_anchor=(0.5, 1.01))

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.96)

    os.makedirs(OUTDIR, exist_ok=True)
    base = f"{OUTDIR}/pipeline_statistics"
    fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(f"{base}.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {base}.png, {base}.pdf")
    plt.close(fig)


# ── Filtering ──────────────────────────────────────────────────────────────

def filter_date_range(all_data: dict, start: str | None, end: str | None) -> dict:
    if start is None and end is None:
        return all_data

    def _filter(data: list[dict]) -> list[dict]:
        filtered = []
        for d in data:
            if start is not None and d["day"] < start.replace("-", ""):
                continue
            if end is not None and d["day"] > end.replace("-", ""):
                continue
            filtered.append(d)
        return filtered

    return {key: _filter(val) for key, val in all_data.items()}


def data_json_path() -> str:
    return f"{OUTDIR}/pipeline_statistics_data.json"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot per-day row counts across data-pipeline stages."
    )
    parser.add_argument("--raw-dataset", default="posts",
                        help="Raw dataset name (default: posts)")
    parser.add_argument("--annotator", default="feasibility_classifier_003",
                        help="Annotation filter name (default: feasibility_classifier_003)")
    parser.add_argument("--clustering-dataset", default="posts_clustered_dbscan_002",
                        help="Clustering output dataset (default: posts_clustered_dbscan_002)")
    parser.add_argument("--summarization-dataset", default="posts_summarized_005_diverse",
                        help="Summarization output dataset (default: posts_summarized_005_diverse)")
    parser.add_argument("--start", metavar="YYYY-MM-DD",
                        help="Start date (inclusive) for the plot x-axis.")
    parser.add_argument("--end", metavar="YYYY-MM-DD",
                        help="End date (inclusive) for the plot x-axis.")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate plots from cached JSON (no S3 reads).")
    args = parser.parse_args()

    if args.regenerate:
        json_path = data_json_path()
        if not os.path.exists(json_path):
            sys.exit(f"Data file not found: {json_path}. Run without --regenerate first.")
        with open(json_path) as f:
            all_data = json.load(f)
    else:
        import boto3

        session = boto3.Session(
            aws_access_key_id=os.environ["S3_ACCESS_KEY"],
            aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        )
        kwargs = {}
        if os.environ.get("S3_ENDPOINT_URL"):
            kwargs["endpoint_url"] = os.environ["S3_ENDPOINT_URL"]
        s3_client = session.client("s3", **kwargs)

        all_data = collect_all(
            s3_client,
            args.raw_dataset,
            args.annotator,
            args.clustering_dataset,
            args.summarization_dataset,
        )

        os.makedirs(OUTDIR, exist_ok=True)
        json_path = data_json_path()
        with open(json_path, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"Saved data: {json_path}")

    all_data = filter_date_range(all_data, args.start, args.end)
    plot(all_data)


if __name__ == "__main__":
    main()
