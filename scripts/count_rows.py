"""Count total rows and annotated rows in the "posts" dataset."""

import asyncio
import os
import sys

# Add s3-data-tool to path
sys.path.insert(0, "/mnt/s3-data-tool")

import duckdb
from dotenv import load_dotenv

from s3_data_tool.s3_utils import enumerate_parquet_paths_sync
import boto3

load_dotenv()


def count_parquet_rows(s3_client, bucket: str, paths: list[str]) -> int:
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
            SET s3_access_key_id='{os.environ['S3_ACCESS_KEY']}';
            SET s3_secret_access_key='{os.environ['S3_SECRET_KEY']}';
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


def main():
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "datasets")
    endpoint_url = os.environ.get("S3_ENDPOINT_URL") or None
    access_key = os.environ["S3_ACCESS_KEY"]
    secret_key = os.environ["S3_SECRET_KEY"]

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    s3_client = session.client("s3", **kwargs)

    # 1. Total rows in "posts" (base dataset)
    base_paths = enumerate_parquet_paths_sync(s3_client, bucket, prefix, "posts")
    print(f"Base parquet files found: {len(base_paths)}")
    for p in base_paths:
        print(f"  {p}")

    if base_paths:
        total_rows = count_parquet_rows(s3_client, bucket, base_paths)
        print(f"\nTotal rows in 'posts': {total_rows:,}")
    else:
        print("\nNo base parquet files found for 'posts'")
        total_rows = 0

    # 2. Annotated rows by feasibility_classifier_003
    annotator = "feasibility_classifier_003"
    annot_paths = enumerate_parquet_paths_sync(s3_client, bucket, prefix, "posts", annotator)
    print(f"\nAnnotation parquet files for '{annotator}': {len(annot_paths)} (of {len(base_paths)} total batches)")
    for p in annot_paths:
        print(f"  {p}")

    if annot_paths:
        annotated_rows = count_parquet_rows(s3_client, bucket, annot_paths)
        print(f"\nAnnotated rows: {annotated_rows:,}")
        if total_rows > 0:
            pct = annotated_rows / total_rows * 100
            print(f"Coverage: {pct:.2f}%")
    else:
        print("\nNo annotation parquet files found")
        annotated_rows = 0

    # Also check for partial/temp JSONL files that haven't been merged yet
    print("\n--- Checking for unmerged annotation JSONL files ---")
    annot_prefix = f"{prefix}/posts/annotations/{annotator}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    jsonl_count = 0
    jsonl_rows_estimate = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=annot_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".jsonl"):
                size = obj["Size"]
                jsonl_count += 1
                print(f"  {key} ({size:,} bytes)")
                # Rough estimate: ~1 row per line
                if size > 0:
                    try:
                        response = s3_client.get_object(Bucket=bucket, Key=key)
                        body = response["Body"].read().decode("utf-8")
                        n = len([l for l in body.strip().split("\n") if l.strip()])
                        jsonl_rows_estimate += n
                    except Exception:
                        pass

    if jsonl_count > 0:
        print(f"\nUnmerged JSONL files: {jsonl_count} (est. ~{jsonl_rows_estimate:,} rows)")
        print(f"Total annotated (merged + unmerged est.): {annotated_rows + jsonl_rows_estimate:,}")
    else:
        print("No unmerged JSONL files found.")

    # Check annotation manifests for completeness info
    print("\n--- Annotation manifests ---")
    manifest_count = 0
    total_manifest_rows = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=annot_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "manifest" in key.lower():
                manifest_count += 1
                try:
                    response = s3_client.get_object(Bucket=bucket, Key=key)
                    body = response["Body"].read().decode("utf-8")
                    import json
                    data = json.loads(body)
                    total_manifest_rows += data.get("num_annotated", 0)
                except Exception:
                    pass
    print(f"  Manifests found: {manifest_count}")
    print(f"  Total num_annotated across manifests: {total_manifest_rows:,}")


if __name__ == "__main__":
    main()
