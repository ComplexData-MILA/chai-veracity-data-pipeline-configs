import argparse
import asyncio
import os

from datasets import Dataset
from s3_data_tool import S3DataTool, RawDuckFilter


async def main(hub_path: str | None = None, private: bool = False):
    async with S3DataTool().filter_for_annotation(
        annotator_name="export_001", # placeholder to prevent concurrent runs
        name="posts",
        base_columns=["text"],
        annotator_columns={
            "feasibility_classifier_001": ["classifier_label", "classifier_probs"]
        },
        annotator_filters={
            "feasibility_classifier_001": RawDuckFilter(
                sql="classifier_label = '\"LABEL_1\"'",  # raw values are JSON-encoded strings: "LABEL_1"
            ),
        },
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
    parser = argparse.ArgumentParser(
        description="Export S3 annotation data to a HuggingFace dataset."
    )
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
