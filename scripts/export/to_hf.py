import argparse
import asyncio
import os

from datasets import Dataset
from pydantic import BaseModel
from s3_data_tool import S3DataTool, FilterNode, RawDuckFilter


class ExportConfig(BaseModel):
    """Configuration for a single export mode."""

    name: str
    base_columns: list[str]
    annotator_columns: dict[str, list[str]] = {}
    annotator_filters: dict[str, FilterNode] = {}
    base_filter: FilterNode | None = None


EXPORT_MODES: dict[str, ExportConfig] = {
    "classifier_hits": ExportConfig(
        name="posts",
        base_columns=["created_at", "at_uri", "uri", "text"],
        annotator_columns={
            "feasibility_classifier_001": ["classifier_label", "classifier_probs"],
        },
        annotator_filters={
            "feasibility_classifier_001": RawDuckFilter(
                sql="classifier_label = '\"LABEL_1\"'",
            ),
        },
    ),
    "raw_clusters": ExportConfig(
        name="posts_clustered_004",
        base_columns=["id", "original_ids", "text"],
    ),
    "clustered": ExportConfig(
        name="posts_summarized_003_b",
        base_columns=[
            "claim",
            "cluster_id",
            "id",
            "original_ids",
            "original_texts",
            "post_count",
        ],
        base_filter=RawDuckFilter(sql="post_count >= 2"),
    ),
}


async def main(mode: str, hub_path: str | None = None, private: bool = False):
    config = EXPORT_MODES[mode]

    async with S3DataTool().filter_for_annotation(
        annotator_name="export_001",  # placeholder to prevent concurrent runs
        name=config.name,
        base_columns=config.base_columns,
        annotator_columns=config.annotator_columns or None,
        annotator_filters=config.annotator_filters or None,
        base_filter=config.base_filter,
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
        "--mode",
        default="clustered",
        choices=list(EXPORT_MODES.keys()),
        help="Export mode selecting which dataset/columns/filters to use. "
        f"Available: {', '.join(EXPORT_MODES)}.",
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
    asyncio.run(main(mode=args.mode, hub_path=args.hub_path, private=args.private))
