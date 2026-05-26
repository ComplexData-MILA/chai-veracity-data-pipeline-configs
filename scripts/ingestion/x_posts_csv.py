"""
Ingest X/Twitter posts from a CSV file into the dataset.

Batch names are generated automatically as ``x-posts-YYYYMMDD-HH`` based on the
time the script is run.
"""

import argparse
import asyncio
import csv
from datetime import datetime
from typing import Any, AsyncGenerator


async def _iter_csv(path: str) -> AsyncGenerator[dict[str, Any], None]:
    """Read CSV rows and yield one dict per post."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "uri": f"https://x.com/{row['source_username']}/status/{row['tweet_id']}",
                "text": row.get("text"),
                "raw": dict(row),
            }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to a tweets CSV file")
    parser.add_argument("--dataset-name", default="posts")
    args = parser.parse_args()

    from s3_data_tool import S3DataTool

    timestamp = datetime.now().strftime("%Y%m%d-%H")
    batch_name = f"x-posts-{timestamp}"

    async with S3DataTool().dataset_generator() as dataset_generator:
        await dataset_generator.from_async_iterator(
            _iter_csv(args.csv_path),
            name=args.dataset_name,
            batch=batch_name,
            streaming_configs=S3DataTool.StreamingConfigs(chunk_size=100),
            deduplicate_on=["uri"],
        )


if __name__ == "__main__":
    asyncio.run(main())
