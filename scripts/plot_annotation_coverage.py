"""Plot per-day annotation coverage and label distribution.

Usage:
    uv run --env-file .env python scripts/plot_annotation_coverage.py [--annotator NAME]

    # Regenerate from cached JSON (no S3 access needed):
    uv run --env-file .env python scripts/plot_annotation_coverage.py --regenerate

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
COLORS = {"LABEL_0": "#999999", "LABEL_1": "#7CB5EC", "LABEL_2": "#2E75B6"}

matplotlib.use("Agg")


def extract_day(path: str) -> str:
    for part in path.split("/"):
        if part.startswith("bsky-jetstream-"):
            return part[len("bsky-jetstream-"):][:8]
    return "unknown"


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


def count_rows(paths: list[str], conn) -> int:
    if not paths:
        return 0
    paths_sql = ", ".join(f"'{p}'" for p in paths)
    result = conn.execute(
        f"SELECT count(*) FROM read_parquet([{paths_sql}])"
    ).fetchone()
    return result[0] if result else 0


def query_labels(paths: list[str], conn) -> dict[str, int]:
    if not paths:
        return {}
    paths_sql = ", ".join(f"'{p}'" for p in paths)
    result = conn.execute(f"""
        SELECT json_extract_string(classifier_label, '$') AS label, count(*) AS cnt
        FROM read_parquet([{paths_sql}])
        GROUP BY label
        ORDER BY label
    """).fetchall()
    return {row[0]: row[1] for row in result}


def parse_date(day_str: str) -> datetime:
    return datetime.strptime(day_str, "%Y%m%d")


def day_label(day_str: str) -> str:
    """Only label Sundays, others stay blank so the x-axis isn't crowded."""
    dt = parse_date(day_str)
    dow = dt.strftime("%a").upper()
    if dow == "SUN":
        return f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}\n({dow})"
    return ""


def collect_data(s3_client, annotator: str) -> list[dict]:
    """Collect all data needed for the plot."""
    sys.path.insert(0, "/mnt/s3-data-tool")
    from s3_data_tool.s3_utils import enumerate_parquet_paths_sync

    base_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, "posts")
    annot_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, "posts", annotator)

    base_by_day = defaultdict(list)
    for p in base_paths:
        base_by_day[extract_day(p)].append(p)

    annot_by_day = defaultdict(list)
    for p in annot_paths:
        annot_by_day[extract_day(p)].append(p)

    all_days = sorted(set(base_by_day.keys()))

    conn = duckdb_conn()
    days_data = []
    for day in all_days:
        total = count_rows(base_by_day[day], conn)
        annotated = count_rows(annot_by_day[day], conn)
        labels = query_labels(annot_by_day[day], conn) if annotated else {}
        days_data.append({
            "day": day,
            "total": total,
            "annotated": annotated,
            "labels": labels,
        })
    conn.close()
    return days_data


def plot(annotator: str, days_data: list[dict]):
    plt.rcParams.update({"font.size": 18})

    days = [d["day"] for d in days_data]
    totals = [d["total"] for d in days_data]
    annotated = [d["annotated"] for d in days_data]
    x = list(range(len(days)))

    x_labels = [day_label(d) for d in days]

    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # ── Left axis: Total & Annotated scatter + lines ──
    total_color = "#333333"
    annot_color = "#2E75B6"

    ax1.scatter(x, totals, marker="o", s=48, facecolor="white",
                edgecolor=total_color, linewidths=1.0, zorder=10, label="Total")
    ax1.scatter(x, annotated, marker="o", s=48, facecolor="white",
                edgecolor=annot_color, linewidths=1.0, zorder=10, label="Annotated")

    for i in range(len(x) - 1):
        ax1.plot([x[i], x[i+1]], [totals[i], totals[i+1]],
                 color=total_color, linewidth=1.0, alpha=0.8, zorder=9)
        ax1.plot([x[i], x[i+1]], [annotated[i], annotated[i+1]],
                 color=annot_color, linewidth=1.0, alpha=0.8, zorder=9)

    ax1.set_ylabel("Number of Posts", color=total_color, labelpad=12)
    ax1.tick_params(axis="y", labelcolor=total_color)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))

    def _yfmt(v, _):
        if v == 0:
            return "0"
        s = f"{v/1e6:.1f}".rstrip("0").rstrip(".")
        return f"{s}M"
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_yfmt))
    ax1.set_ylim(0, max(totals) * 1.08)
    ax1.patch.set_visible(False)

    # ── Right axis: 100% stacked bars ──
    ax2 = ax1.twinx()
    ax1.set_zorder(2)
    ax2.set_zorder(1)
    bar_width = 0.55
    bottom = [0.0] * len(days)
    label_order = ["LABEL_0", "LABEL_1", "LABEL_2"]
    label_names = {
        "LABEL_0": "Class 0 (not feasible)",
        "LABEL_1": "Class 1 (ambiguous)",
        "LABEL_2": "Class 2 (clear)",
    }

    for label in label_order:
        pcts = []
        for d in days_data:
            day_total = sum(d["labels"].values())
            count = d["labels"].get(label, 0)
            pcts.append(count / day_total * 100 if day_total > 0 else 0.0)
        ax2.bar(x, pcts, bar_width, bottom=bottom, color=COLORS[label],
                label=label_names[label], alpha=0.85, zorder=1)
        bottom = [b + p for b, p in zip(bottom, pcts)]

    ax2.set_ylabel("Feasibility Label %", color="#555555", labelpad=12)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="#555555")

    # ── X-axis ──
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, linespacing=1.2)
    ax1.set_xlim(-0.6, len(x) - 0.4)

    # ── Grid ──
    ax2.grid(axis="y", alpha=0.3)

    # ── Legend ──
    lines1, labels1 = ax1.get_legend_handles_labels()
    bars2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + bars2[::-1], labels1 + labels2[::-1],
               loc="upper center", ncol=5, frameon=True,
               fontsize=14, framealpha=0.9, bbox_to_anchor=(0.5, 1.01))

#    ax1.set_title("Annotation coverage & label distribution", fontweight="bold", y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(OUTDIR, exist_ok=True)
    base = f"{OUTDIR}/annotation_coverage_{annotator}"
    fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(f"{base}.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {base}.png, {base}.pdf")
    plt.close(fig)


def filter_date_range(days_data: list[dict], start: str | None, end: str | None) -> list[dict]:
    """Filter days_data to the inclusive [start, end] date range."""
    if start is None and end is None:
        return days_data
    filtered = []
    for d in days_data:
        if start is not None and d["day"] < start.replace("-", ""):
            continue
        if end is not None and d["day"] > end.replace("-", ""):
            continue
        filtered.append(d)
    return filtered


def data_json_path(annotator: str) -> str:
    return f"{OUTDIR}/annotation_coverage_{annotator}_data.json"


def main():
    parser = argparse.ArgumentParser(
        description="Plot annotation coverage and label distribution by day."
    )
    parser.add_argument("--annotator", default="feasibility_classifier_003",
                        help="Annotator name (default: feasibility_classifier_003)")
    parser.add_argument("--start", metavar="YYYY-MM-DD",
                        help="Start date (inclusive) for the plot x-axis.")
    parser.add_argument("--end", metavar="YYYY-MM-DD",
                        help="End date (inclusive) for the plot x-axis.")
    parser.add_argument("--regenerate",
                        action="store_true",
                        help="Regenerate plots from cached JSON (no S3 reads).")
    args = parser.parse_args()

    if args.regenerate:
        json_path = data_json_path(args.annotator)
        if not os.path.exists(json_path):
            sys.exit(f"Data file not found: {json_path}. Run without --regenerate first.")
        with open(json_path) as f:
            days_data = json.load(f)
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

        days_data = collect_data(s3_client, args.annotator)

        os.makedirs(OUTDIR, exist_ok=True)
        json_path = data_json_path(args.annotator)
        with open(json_path, "w") as f:
            json.dump(days_data, f, indent=2)
        print(f"Saved data: {json_path}")

    days_data = filter_date_range(days_data, args.start, args.end)
    plot(args.annotator, days_data)


if __name__ == "__main__":
    main()
