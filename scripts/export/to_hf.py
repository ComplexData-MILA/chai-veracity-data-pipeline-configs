import argparse
import asyncio
import os

from datasets import Dataset
from s3_data_tool import S3DataTool


async def main(hub_path: str | None = None, private: bool = False):
    async with S3DataTool().filter_for_annotation(
        name="posts_summarized_001",
        annotator_name="export_001",
        base_columns=["id", "cluster_id", "original_ids", "synopsis", "topic"],
    ) as generator:
        rows = [row.data async for row in generator]

    if not rows:
        print("No rows to export.")
        return

    ds = Dataset.from_list(rows)
    print(f"Created dataset with {len(ds)} rows, columns: {ds.column_names}")

    if hub_path:
        ds.push_to_hub(hub_path, private=private)
        print(f"Pushed dataset to {hub_path}")
    else:
        print("No --hub-path provided; dataset not pushed.")
        print(ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export S3 annotation data to a HuggingFace dataset.")
    parser.add_argument(
        "--hub-path",
        default=os.environ.get("HF_HUB_PATH"),
        help="HuggingFace Hub repo path (e.g. 'my-org/my-dataset'). "
             "Can also set via HF_HUB_PATH env var. If omitted, dataset is printed but not pushed.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=os.environ.get("HF_PRIVATE", "").lower() in ("1", "true", "yes"),
        help="Create a private repo on the Hub. Can also set via HF_PRIVATE=true env var.",
    )
    args = parser.parse_args()
    asyncio.run(main(hub_path=args.hub_path, private=args.private))
