#!/usr/bin/env python3
"""
Upload a single day's JSONL output to S3 using the same S3DataTool format
as the original pipeline.

Usage:
  python upload_single_day.py \
    --output-dir /path/to/output/20260427 \
    --dataset-name posts_clustered_dbscan_005_b \
    --batch bsky-trending-20260427
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import aioboto3


async def upload_jsonl_files(
    output_dir: Path,
    dataset_name: str,
    batch_name: str,
) -> int:
    """Upload all .jsonl files from output_dir to S3 under dataset/batch."""
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "datasets")
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "")
    access_key = os.environ.get("S3_ACCESS_KEY", "")
    secret_key = os.environ.get("S3_SECRET_KEY", "")

    session = aioboto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    s3_kwargs = {}
    if endpoint_url:
        s3_kwargs["endpoint_url"] = endpoint_url

    # Collect all JSONL files
    jsonl_files = sorted(output_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {output_dir}")
        return 0

    total_uploaded = 0
    async with session.client("s3", **s3_kwargs) as s3_client:
        for jsonl_path in jsonl_files:
            key = f"{prefix}/{dataset_name}/{batch_name}/{jsonl_path.name}"
            print(f"  Uploading {jsonl_path.name} -> s3://{bucket}/{key}  "
                  f"({jsonl_path.stat().st_size / 1024:.1f} KB)")
            await s3_client.upload_file(str(jsonl_path), bucket, key)
            total_uploaded += 1

    # Also upload a run manifest so the dataset generator can find the data
    manifest = {
        "run_id": batch_name,
        "completed": True,
        "chunk_count": total_uploaded,
    }
    manifest_key = f"{prefix}/{dataset_name}/{batch_name}/run_manifest.json"
    async with session.client("s3", **s3_kwargs) as s3_client:
        await s3_client.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=json.dumps(manifest).encode("utf-8"),
        )
    print(f"  Uploaded manifest -> s3://{bucket}/{manifest_key}")

    return total_uploaded


async def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a single day's JSONL output to S3")
    parser.add_argument("--output-dir", required=True, help="Directory containing JSONL files")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--batch", required=True, help="Output batch name")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    n = await upload_jsonl_files(output_dir, args.dataset_name, args.batch)
    print(f"Uploaded {n} files for batch {args.batch}")


if __name__ == "__main__":
    asyncio.run(main())
