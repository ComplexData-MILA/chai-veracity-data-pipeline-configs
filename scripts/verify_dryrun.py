"""Verify dry run embeddings: check S3 for results and validate structure."""

import asyncio
import json
import os
import sys

import aioboto3

from s3_data_tool.s3_utils import (
    enumerate_batches,
    enumerate_parquet_paths,
    read_parquet_rows,
    annotation_manifest_key,
    read_annotation_manifest,
)


async def main():
    annotator_name = sys.argv[1] if len(sys.argv) > 1 else "embeddings_128d_dryrun"
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "datasets")
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "")
    access_key = os.environ["S3_ACCESS_KEY"]
    secret_key = os.environ["S3_SECRET_KEY"]

    session = aioboto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    async with session.client("s3", **kwargs) as s3_client:
        # Check for annotation paths
        annotator_paths = await enumerate_parquet_paths(
            s3_client, bucket, prefix, "posts", annotator_name
        )
        print(f"Found {len(annotator_paths)} merged parquet files for '{annotator_name}':")
        for p in annotator_paths:
            print(f"  {p}")

        # Check for temp chunks too
        temp_prefix = f"{prefix}/posts/annotations/{annotator_name}/"
        paginator = s3_client.get_paginator("list_objects_v2")
        all_objects = []
        async for page in paginator.paginate(Bucket=bucket, Prefix=temp_prefix):
            for obj in page.get("Contents", []):
                all_objects.append(obj["Key"])

        jsonl_chunks = [k for k in all_objects if k.endswith(".jsonl")]
        manifests = [k for k in all_objects if k.endswith("annotation_manifest.json")]
        print(f"\nTemp JSONL chunks (.temp/): {len(jsonl_chunks)}")
        for c in jsonl_chunks:
            print(f"  {c}")
        print(f"Annotation manifests: {len(manifests)}")
        for m in manifests:
            print(f"  {m}")
            # Read manifest contents
            try:
                response = await s3_client.get_object(Bucket=bucket, Key=m)
                body = await response["Body"].read()
                manifest = json.loads(body)
                print(f"    -> {json.dumps(manifest)}")
            except Exception as e:
                print(f"    -> Error reading: {e}")

        # Read sample rows if parquet exists
        if annotator_paths:
            print("\n--- Sample rows from first parquet ---")
            rows = await read_parquet_rows(s3_client, bucket, annotator_paths[0])
            print(f"Total rows in first parquet: {len(rows)}")
            if rows:
                print(f"Sample row keys: {list(rows[0].keys())}")
                print(f"Sample row (first 200 chars): {json.dumps(rows[0])[:200]}")
                # Check embedding field
                for row in rows[:5]:
                    has_embedding = "embedding" in row
                    embedding_len = len(row.get("embedding", "")) if has_embedding else 0
                    print(f"  id={row.get('id')[:30]}... has_embedding={has_embedding} embedding_b64_len={embedding_len} dimensions={row.get('dimensions')}")

        if not annotator_paths and not jsonl_chunks:
            print("\nNo results found yet. The job may not have completed or produced output.")


if __name__ == "__main__":
    asyncio.run(main())
