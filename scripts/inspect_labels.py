"""Inspect the actual schema and sample rows from annotation parquet."""

import json
import os
import sys

sys.path.insert(0, "/mnt/s3-data-tool")

import duckdb
from dotenv import load_dotenv
from s3_data_tool.s3_utils import enumerate_parquet_paths_sync
import boto3

load_dotenv()

BUCKET = os.environ["S3_BUCKET"]
PREFIX = os.environ.get("S3_PREFIX", "datasets")
ANNOTATOR = "feasibility_classifier_003"


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

    # Pick the first path
    path = annot_paths[0]
    print(f"Inspecting: {path}\n")

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

    # Show schema
    result = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
    print("Schema:")
    for row in result:
        print(f"  {row[0]:30s} {row[1]}")

    # Show 5 sample rows
    result = conn.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 5").fetchall()
    cols = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
    col_names = [c[0] for c in cols]

    print("\nSample rows:")
    for row in result:
        print()
        for name, val in zip(col_names, row):
            val_str = str(val)
            if len(val_str) > 120:
                val_str = val_str[:120] + "..."
            print(f"  {name}: {val_str}")

    conn.close()


if __name__ == "__main__":
    main()
