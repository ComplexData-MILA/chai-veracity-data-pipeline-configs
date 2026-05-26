"""Day-by-day annotation coverage analysis for a given annotator."""

import asyncio
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


def count_rows_in_paths(paths: list[str]) -> int:
    """Count total rows across all given parquet paths using DuckDB."""
    if not paths:
        return 0
    paths_sql = ", ".join(f"'{p}'" for p in paths)
    conn = duckdb.connect()
    try:
        s3_endpoint = os.environ["S3_ENDPOINT_URL"]
        endpoint_host = s3_endpoint.removeprefix("https://").rstrip("/")
        endpoint_host = endpoint_host.removeprefix("http://").rstrip("/")
        use_ssl = s3_endpoint.startswith("https://")

        conn.execute(f"""
            SET s3_access_key_id='{os.environ["S3_ACCESS_KEY"]}';
            SET s3_secret_access_key='{os.environ["S3_SECRET_KEY"]}';
            SET s3_endpoint='{endpoint_host}';
            SET s3_use_ssl={str(use_ssl).lower()};
            SET s3_url_style='path';
        """)
        result = conn.execute(
            f"SELECT count(*) FROM read_parquet([{paths_sql}])"
        ).fetchone()
        return result[0] if result else 0
    finally:
        conn.close()


def extract_day(path: str) -> str:
    """Extract YYYYMMDD from a parquet path like .../bsky-jetstream-YYYYMMDD-HH/..."""
    for part in path.split("/"):
        if part.startswith("bsky-jetstream-"):
            return part[len("bsky-jetstream-"):][:8]
    return "unknown"


def main():
    session = boto3.Session(
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    )
    kwargs = {}
    if os.environ.get("S3_ENDPOINT_URL"):
        kwargs["endpoint_url"] = os.environ["S3_ENDPOINT_URL"]
    s3_client = session.client("s3", **kwargs)

    # Get all base paths grouped by day
    base_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, "posts")
    base_by_day = defaultdict(list)
    for p in base_paths:
        day = extract_day(p)
        base_by_day[day].append(p)

    # Get all annotation paths grouped by day
    annot_paths = enumerate_parquet_paths_sync(s3_client, BUCKET, PREFIX, "posts", ANNOTATOR)
    annot_by_day = defaultdict(list)
    for p in annot_paths:
        day = extract_day(p)
        annot_by_day[day].append(p)

    all_days = sorted(set(base_by_day.keys()) | set(annot_by_day.keys()))

    print(f"{'Day':<10} {'Batch':>5} {'Total':>12} {'Annotated':>12} {'Merged%':>8} {'Status'}")
    print("-" * 75)

    grand_total = 0
    grand_annotated = 0

    for day in all_days:
        base = base_by_day.get(day, [])
        annot = annot_by_day.get(day, [])
        n_batches = len(base)

        total = count_rows_in_paths(base) if base else 0
        annotated = count_rows_in_paths(annot) if annot else 0
        pct = annotated / total * 100 if total > 0 else 0

        grand_total += total
        grand_annotated += annotated

        if pct == 0:
            status = "not annotated"
        elif pct >= 99.9:
            status = "complete"
        elif pct > 50:
            status = f"partial ({len(annot)}/{n_batches} batches)"
        else:
            status = f"sparse ({len(annot)}/{n_batches} batches)"

        print(f"{day:<10} {n_batches:>5} {total:>12,} {annotated:>12,} {pct:>7.2f}% {status}")

    print("-" * 75)
    overall = grand_annotated / grand_total * 100 if grand_total > 0 else 0
    print(f"{'TOTAL':<10} {len(base_paths):>5} {grand_total:>12,} {grand_annotated:>12,} {overall:>7.2f}%")


if __name__ == "__main__":
    main()
