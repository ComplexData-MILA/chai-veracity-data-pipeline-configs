"""Per-day label distribution for a given annotator."""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/mnt/s3-data-tool")

import duckdb
from dotenv import load_dotenv
from s3_data_tool.s3_utils import enumerate_parquet_paths_sync
import boto3

load_dotenv()

BUCKET = os.environ["S3_BUCKET"]
PREFIX = os.environ.get("S3_PREFIX", "datasets")
ANNOTATOR = "feasibility_classifier_003"


def extract_day(path: str) -> str:
    for part in path.split("/"):
        if part.startswith("bsky-jetstream-"):
            return part[len("bsky-jetstream-"):][:8]
    return "unknown"


def query_label_counts(paths: list[str]) -> dict[str, int]:
    """Count rows per classifier_label in the given parquet paths."""
    if not paths:
        return {}
    paths_sql = ", ".join(f"'{p}'" for p in paths)
    conn = duckdb.connect()
    try:
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

        # classifier_label is stored as JSON-encoded string ('"LABEL_0"').
        # Use json_extract_string to get the bare label value.
        result = conn.execute(f"""
            SELECT json_extract_string(classifier_label, '$') AS label, count(*) AS cnt
            FROM read_parquet([{paths_sql}])
            GROUP BY label
            ORDER BY label
        """).fetchall()
        return {row[0]: row[1] for row in result}
    finally:
        conn.close()


def main():
    session = boto3.Session(
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    )
    kwargs = {}
    if os.environ.get("S3_ENDPOINT_URL"):
        kwargs["endpoint_url"] = os.environ["S3_ENDPOINT_URL"]
    s3_client = session.client("s3", **kwargs)

    annot_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, "posts", ANNOTATOR)
    annot_by_day = defaultdict(list)
    for p in annot_paths:
        day = extract_day(p)
        annot_by_day[day].append(p)

    all_labels = ["LABEL_0", "LABEL_1", "LABEL_2"]

    print(f"{'Day':<10} {'Annotated':>12} {'LABEL_0':>9} {'LABEL_1':>9} {'LABEL_2':>9}")
    print("-" * 55)

    grand_total = 0
    grand_counts = defaultdict(int)

    for day in sorted(annot_by_day.keys()):
        paths = annot_by_day[day]
        counts = query_label_counts(paths)
        day_total = sum(counts.values())
        grand_total += day_total

        parts = [f"{day:<10} {day_total:>12,}"]
        for label in all_labels:
            c = counts.get(label, 0)
            grand_counts[label] += c
            pct = c / day_total * 100 if day_total > 0 else 0.0
            parts.append(f"{pct:>8.2f}%")
        print("".join(parts))

    print("-" * 55)
    overall = [f"{'TOTAL':<10} {grand_total:>12,}"]
    for label in all_labels:
        pct = grand_counts[label] / grand_total * 100 if grand_total > 0 else 0.0
        overall.append(f"{pct:>8.2f}%")
    print("".join(overall))


if __name__ == "__main__":
    main()
