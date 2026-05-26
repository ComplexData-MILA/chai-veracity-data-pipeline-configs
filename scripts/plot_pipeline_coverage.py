"""Combined pipeline statistics and annotation coverage plot.

Usage:
    uv run --env-file .env python scripts/plot_pipeline_coverage.py [--regenerate]
    uv run --env-file .env python scripts/plot_pipeline_coverage.py --start 2026-04-28 --end 2026-05-19

Left y-axis (broken): scatter+lines for raw, annotated, clusters, summaries.
Right y-axis: 100% stacked bars for feasibility class 0/1/2.

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
    "summarized": "#FF6B35",
    "LABEL_0": "#999999",
    "LABEL_1": "#7CB5EC",
    "LABEL_2": "#2E75B6",
}

LABEL_NAMES = {
    "LABEL_0": "Class 0",
    "LABEL_1": "Class 1",
    "LABEL_2": "Class 2",
}

matplotlib.use("Agg")


# ── Date helpers ───────────────────────────────────────────────────────────

def parse_date(day_str: str) -> datetime:
    return datetime.strptime(day_str, "%Y%m%d")


def day_label(day_str: str) -> str:
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


# ── Path helpers ───────────────────────────────────────────────────────────

def _extract_day_generic(path: str, prefix: str) -> str:
    for part in path.split("/"):
        if part.startswith(prefix):
            return part[len(prefix):][:8]
    return "unknown"


def extract_day_posts(path: str) -> str:
    return _extract_day_generic(path, "bsky-jetstream-")


def extract_day_clustering(path: str) -> str:
    return _extract_day_generic(path, "bsky-trending-")


# ── Data collection (S3) ───────────────────────────────────────────────────

def _count_rows(paths: list[str], conn) -> int:
    if not paths:
        return 0
    paths_sql = ", ".join(f"'{p}'" for p in paths)
    result = conn.execute(
        f"SELECT count(*) FROM read_parquet([{paths_sql}])"
    ).fetchone()
    return result[0] if result else 0


def collect_pipeline_data(s3_client, raw_dataset: str, annotator: str,
                          clustering_dataset: str, summarization_dataset: str) -> dict:
    """Collect per-day row counts for all four pipeline stages."""
    sys.path.insert(0, "/mnt/s3-data-tool")
    from s3_data_tool.s3_utils import enumerate_parquet_paths_sync

    # Raw posts
    raw_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, raw_dataset)
    raw_by_day = defaultdict(list)
    for p in raw_paths:
        raw_by_day[extract_day_posts(p)].append(p)

    # Annotated posts
    annot_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, raw_dataset, annotator)
    annot_by_day = defaultdict(list)
    for p in annot_paths:
        annot_by_day[extract_day_posts(p)].append(p)

    # Clustered
    clust_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, clustering_dataset)
    clust_by_day = defaultdict(list)
    for p in clust_paths:
        clust_by_day[extract_day_clustering(p)].append(p)

    # Summarized
    summ_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, summarization_dataset)

    conn = duckdb_conn()

    def _build(data: defaultdict, conn) -> list[dict]:
        result = []
        for day in sorted(data):
            result.append({"day": day, "count": _count_rows(data[day], conn)})
        return result

    raw_data = _build(raw_by_day, conn)
    annot_data = _build(annot_by_day, conn)
    clust_data = _build(clust_by_day, conn)

    # Summarized uses a ``date`` column
    summ_data = []
    if summ_paths:
        paths_sql = ", ".join(f"'{p}'" for p in summ_paths)
        rows = conn.execute(f"""
            SELECT REPLACE(date, '"', '') AS d, count(*) AS cnt
            FROM read_parquet([{paths_sql}])
            GROUP BY d
            ORDER BY d
        """).fetchall()
        summ_data = [{"day": row[0], "count": row[1]} for row in rows]

    conn.close()
    return {
        "raw": raw_data,
        "annotated": annot_data,
        "clustered": clust_data,
        "summarized": summ_data,
    }


def collect_annotation_data(s3_client, annotator: str) -> list[dict]:
    """Collect per-day total/annotated counts and label distribution."""
    sys.path.insert(0, "/mnt/s3-data-tool")
    from s3_data_tool.s3_utils import enumerate_parquet_paths_sync

    base_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, "posts")
    annot_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, "posts", annotator)

    base_by_day = defaultdict(list)
    for p in base_paths:
        base_by_day[extract_day_posts(p)].append(p)

    annot_by_day = defaultdict(list)
    for p in annot_paths:
        annot_by_day[extract_day_posts(p)].append(p)

    all_days = sorted(set(base_by_day.keys()))

    conn = duckdb_conn()
    days_data = []
    for day in all_days:
        total = _count_rows(base_by_day[day], conn)
        annotated = _count_rows(annot_by_day[day], conn)
        labels = {}
        if annotated and annot_by_day[day]:
            paths_sql = ", ".join(f"'{p}'" for p in annot_by_day[day])
            rows = conn.execute(f"""
                SELECT json_extract_string(classifier_label, '$') AS label, count(*) AS cnt
                FROM read_parquet([{paths_sql}])
                GROUP BY label
                ORDER BY label
            """).fetchall()
            labels = {row[0]: row[1] for row in rows}
        days_data.append({
            "day": day,
            "total": total,
            "annotated": annotated,
            "labels": labels,
        })
    conn.close()
    return days_data


# ── Data merge ─────────────────────────────────────────────────────────────

def merge_data(pipeline_data: dict, annotation_data: list[dict]) -> list[dict]:
    """Merge pipeline and annotation data on day key into unified rows."""
    annot_lu = {d["day"]: d for d in annotation_data}

    raw_lu = {d["day"]: d["count"] for d in pipeline_data["raw"]}
    annot_count_lu = {d["day"]: d["count"] for d in pipeline_data["annotated"]}
    clust_lu = {d["day"]: d["count"] for d in pipeline_data["clustered"]}
    summ_lu = {d["day"]: d["count"] for d in pipeline_data["summarized"]}

    all_days = set(raw_lu) | set(annot_count_lu) | set(clust_lu) | set(summ_lu) | set(annot_lu)
    all_days = sorted(all_days)

    merged = []
    for day in all_days:
        entry = {
            "day": day,
            "raw": raw_lu.get(day, 0),
            "annotated": annot_count_lu.get(day, 0),
            "clustered": clust_lu.get(day, 0),
            "summarized": summ_lu.get(day, 0),
        }
        ad = annot_lu.get(day, {})
        entry["total"] = ad.get("total", 0)
        entry["labels"] = ad.get("labels", {})
        merged.append(entry)
    return merged


# ── Filtering ──────────────────────────────────────────────────────────────

def filter_date_range(merged: list[dict], start: str | None, end: str | None) -> list[dict]:
    if start is None and end is None:
        return merged
    filtered = []
    for d in merged:
        if start is not None and d["day"] < start.replace("-", ""):
            continue
        if end is not None and d["day"] > end.replace("-", ""):
            continue
        filtered.append(d)
    return filtered


# ── Plot ───────────────────────────────────────────────────────────────────

def _scatter_lines(ax, x_vals: list[int], y_vals: list[int], color: str,
                   label: str):
    if not x_vals:
        return
    ax.scatter(x_vals, y_vals, marker="o", s=48, facecolor="white",
               edgecolor=color, linewidths=1.0, zorder=10, label=label)
    for i in range(len(x_vals) - 1):
        ax.plot([x_vals[i], x_vals[i + 1]], [y_vals[i], y_vals[i + 1]],
                color=color, linewidth=1.0, alpha=0.8, zorder=9)


def _add_break_marks(ax_upper, ax_lower):
    d = 0.5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color="k", mec="k", mew=1, clip_on=False)
    ax_upper.plot([0, 1], [0, 0], transform=ax_upper.transAxes, **kwargs)
    ax_lower.plot([0, 1], [1, 1], transform=ax_lower.transAxes, **kwargs)


def _set_y_ticks_no_boundary(ax):
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
        upper_bottom = min_upper * 0.85

    if upper_bottom <= lower_top:
        upper_bottom = lower_top * 1.5

    return lower_top, upper_bottom


def _draw_stacked_bars(ax, x: list[int], merged: list[dict], bar_width: float,
                       label_order: list[str], colors: dict, label_names: dict):
    """Draw 100% stacked bars for label distribution on a twin axis."""
    bottom = [0.0] * len(x)
    for label in label_order:
        pcts = []
        for d in merged:
            day_total = sum(d["labels"].values())
            count = d["labels"].get(label, 0)
            pcts.append(count / day_total * 100 if day_total > 0 else 0.0)
        ax.bar(x, pcts, bar_width, bottom=bottom, color=colors[label],
               label=label_names[label], alpha=0.85, zorder=1)
        bottom = [b + p for b, p in zip(bottom, pcts)]


def plot(merged: list[dict]):
    plt.rcParams.update({"font.size": 16})

    days = [d["day"] for d in merged]
    x = list(range(len(days)))
    x_labels = [day_label(d) for d in days]

    raw_y = [d["raw"] for d in merged]
    annot_y = [d["annotated"] for d in merged]
    clust_y = [d["clustered"] for d in merged]
    summ_y = [d["summarized"] for d in merged]

    # ── Create broken-axis subplots ──
    fig, (ax_upper, ax_lower) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 6),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05},
    )
    fig.patch.set_facecolor("white")

    # ── Upper subplot: raw + annotated (scatter+lines) ──
    _scatter_lines(ax_upper, x, raw_y, COLORS["raw"], "Raw posts")
    _scatter_lines(ax_upper, x, annot_y, COLORS["annotated"], "Annotated posts")
    all_upper_vals = [v for v in raw_y + annot_y if v > 0]

    # ── Lower subplot: clusters + summaries (scatter+lines; summaries use scatter now) ──
    _scatter_lines(ax_lower, x, clust_y, COLORS["clustered"], "Clusters")
    _scatter_lines(ax_lower, x, summ_y, COLORS["summarized"], "Summaries")
    all_lower_vals = [v for v in clust_y + summ_y if v > 0]

    # ── Auto-compute y-axis break ──
    lower_top, upper_bottom = _auto_break_point(all_upper_vals, all_lower_vals)
    max_upper = max(all_upper_vals) if all_upper_vals else 1000
    ax_upper.set_ylim(upper_bottom, max_upper * 1.15)
    ax_lower.set_ylim(0, lower_top)

    # ── Right y-axis: 100% stacked bars on lower subplot only ──
    bar_width = 0.55
    label_order = ["LABEL_0", "LABEL_1", "LABEL_2"]

    ax_lower_right = ax_lower.twinx()
    _draw_stacked_bars(ax_lower_right, x, merged, bar_width, label_order, COLORS, LABEL_NAMES)
    ax_lower_right.set_ylim(0, 100)
    ax_lower_right.set_ylabel("Feasibility Label %", color="#555555", labelpad=12)
    ax_lower_right.tick_params(axis="y", labelcolor="#555555")

    # ── Axis labels ──
    fig.supylabel("Count", x=0.04)
    ax_upper.set_ylabel("")
    ax_lower.set_ylabel("")

    # ── X-axis ticks ──
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
    _set_y_ticks_no_boundary(ax_upper)
    _set_y_ticks_no_boundary(ax_lower)

    # ── Grid ──
    for ax_obj in (ax_upper, ax_lower):
        ax_obj.grid(axis="y", alpha=0.3)
        ax_obj.grid(axis="x", alpha=0.15)

    # ── Legend: two rows — first row for scatters, second row for class bars ──
    handles_upper, labels_upper = ax_upper.get_legend_handles_labels()
    handles_lower, labels_lower = ax_lower.get_legend_handles_labels()
    handles_right, labels_right = ax_lower_right.get_legend_handles_labels()

    scatter_handles = handles_upper + handles_lower
    scatter_labels = labels_upper + labels_lower
    bar_handles = handles_right
    bar_labels = labels_right

    # Use ncol=4 so 4 scatters fill row 1, 3 bars wrap to row 2
    fig.legend(
        scatter_handles + bar_handles,
        scatter_labels + bar_labels,
        loc="upper center", ncol=max(len(scatter_handles), len(bar_handles)),
        frameon=True, fontsize=14, framealpha=0.9, bbox_to_anchor=(0.5, 1.01),
    )

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.90)

    os.makedirs(OUTDIR, exist_ok=True)
    base = f"{OUTDIR}/pipeline_coverage"
    fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(f"{base}.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {base}.png, {base}.pdf")
    plt.close(fig)


# ── File paths ─────────────────────────────────────────────────────────────

def pipeline_json_path() -> str:
    return f"{OUTDIR}/pipeline_statistics_data.json"


def annotation_json_path(annotator: str) -> str:
    return f"{OUTDIR}/annotation_coverage_{annotator}_data.json"


def merged_json_path() -> str:
    return f"{OUTDIR}/pipeline_coverage_data.json"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined pipeline statistics + annotation coverage plot."
    )
    parser.add_argument("--raw-dataset", default="posts")
    parser.add_argument("--annotator", default="feasibility_classifier_003")
    parser.add_argument("--clustering-dataset", default="posts_clustered_dbscan_002")
    parser.add_argument("--summarization-dataset", default="posts_summarized_005_diverse")
    parser.add_argument("--start", metavar="YYYY-MM-DD")
    parser.add_argument("--end", metavar="YYYY-MM-DD")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate plots from cached JSON (no S3 reads).")
    args = parser.parse_args()

    pipeline_path = pipeline_json_path()
    annot_path = annotation_json_path(args.annotator)
    merged_path = merged_json_path()

    if args.regenerate:
        # Load merged JSON if available, otherwise load individual caches and merge
        if os.path.exists(merged_path):
            with open(merged_path) as f:
                merged = json.load(f)
            print(f"Loaded merged data: {merged_path}")
        else:
            if not os.path.exists(pipeline_path):
                sys.exit(f"Pipeline data not found: {pipeline_path}")
            if not os.path.exists(annot_path):
                sys.exit(f"Annotation data not found: {annot_path}")
            with open(pipeline_path) as f:
                pipeline_data = json.load(f)
            with open(annot_path) as f:
                annotation_data = json.load(f)
            merged = merge_data(pipeline_data, annotation_data)
            with open(merged_path, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"Migrated and saved merged data: {merged_path}")
    else:
        # Try loading from caches first; fall back to S3
        if os.path.exists(merged_path):
            with open(merged_path) as f:
                merged = json.load(f)
            print(f"Loaded merged data: {merged_path}")
        elif os.path.exists(pipeline_path) and os.path.exists(annot_path):
            print("Loading from existing cached data (one-time migration)...")
            with open(pipeline_path) as f:
                pipeline_data = json.load(f)
            with open(annot_path) as f:
                annotation_data = json.load(f)
            merged = merge_data(pipeline_data, annotation_data)
            with open(merged_path, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"Migrated and saved merged data: {merged_path}")
        else:
            print("Fetching data from S3...")
            import boto3

            session = boto3.Session(
                aws_access_key_id=os.environ["S3_ACCESS_KEY"],
                aws_secret_access_key=os.environ["S3_SECRET_KEY"],
            )
            kwargs = {}
            if os.environ.get("S3_ENDPOINT_URL"):
                kwargs["endpoint_url"] = os.environ["S3_ENDPOINT_URL"]
            s3_client = session.client("s3", **kwargs)

            pipeline_data = collect_pipeline_data(
                s3_client, args.raw_dataset, args.annotator,
                args.clustering_dataset, args.summarization_dataset,
            )
            annotation_data = collect_annotation_data(s3_client, args.annotator)

            os.makedirs(OUTDIR, exist_ok=True)
            with open(pipeline_path, "w") as f:
                json.dump(pipeline_data, f, indent=2)
            with open(annot_path, "w") as f:
                json.dump(annotation_data, f, indent=2)

            merged = merge_data(pipeline_data, annotation_data)
            with open(merged_path, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"Fetched and saved all data.")

    merged = filter_date_range(merged, args.start, args.end)
    if not merged:
        sys.exit("No data in selected date range.")
    print(f"Plotting {len(merged)} days ({merged[0]['day']} – {merged[-1]['day']})")
    plot(merged)


if __name__ == "__main__":
    main()
