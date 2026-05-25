#!/usr/bin/env python3
"""Download all posts parquet + embeddings_128d annotations from S3 to local disk.

Writes to DATA_DIR (default: ./data) preserving the s3 prefix structure:
  data/datasets/posts/{batch}/merged.parquet
  data/datasets/posts/annotations/embeddings_128d/{batch}/merged.parquet
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import aioboto3


async def download_batch(s3_client, bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        # Check size matches to skip re-download
        head = await s3_client.head_object(Bucket=bucket, Key=key)
        if dest.stat().st_size == head["ContentLength"]:
            return  # already downloaded
    await s3_client.download_file(bucket, key, str(dest))


async def list_all_keys(s3_client, bucket: str, prefix: str) -> list[dict]:
    """List all objects under a prefix."""
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj)
    return keys


async def main() -> None:
    parser = argparse.ArgumentParser(description="Download S3 data for offline HPC processing")
    parser.add_argument("--data-dir", default="./data", help="Local data root directory")
    parser.add_argument("--dry-run", action="store_true", help="List files only, don't download")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()

    # Read same env vars as S3DataTool
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "")
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "datasets")
    access_key = os.environ.get("S3_ACCESS_KEY", "")
    secret_key = os.environ.get("S3_SECRET_KEY", "")

    session = aioboto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    async with session.client("s3", **kwargs) as s3_client:
        # List everything under prefix/posts/
        search_prefix = f"{prefix}/posts/"
        print(f"Listing objects under s3://{bucket}/{search_prefix} ...")
        all_objects = await list_all_keys(s3_client, bucket, search_prefix)

        # Filter: only merged.parquet files + annotation_manifest.json (needed for structure)
        targets = []
        total_size = 0
        for obj in all_objects:
            key = obj["Key"]
            if key.endswith("merged.parquet"):
                targets.append((key, obj["Size"]))
                total_size += obj["Size"]

        print(f"Found {len(targets)} merged.parquet files, total {total_size/1024/1024/1024:.2f} GB")

        if args.dry_run:
            for key, size in targets:
                print(f"  {key}  ({size/1024/1024:.1f} MB)")
            return

        for i, (key, size) in enumerate(targets):
            dest = data_dir / key
            perc = (i + 1) / len(targets) * 100
            print(f"[{i+1}/{len(targets)}] {key} ({size/1024/1024:.1f} MB) [{perc:.0f}%]")
            await download_batch(s3_client, bucket, key, dest)

        print(f"\nDone. Data downloaded to {data_dir}")


if __name__ == "__main__":
    asyncio.run(main())
