"""Fetch labeled data from S3, deduplicate, split, and cache locally."""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from s3_data_tool import S3DataTool, RawDuckFilter


def _parse_classes(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


async def stream_and_collect(
    max_rows: int | None,
    classes: list[int],
) -> list[dict]:
    """Stream labeled rows from S3, deduplicate inline, balance classes.

    Deduplication happens while streaming (not after). If *max_rows* is set,
    stops when every class has at least ``max_rows // len(classes)`` unique
    items. After streaming, majority classes are randomly downsampled to match
    the minority class.
    """
    seen: set[str] = set()
    buckets: dict[int, list[dict]] = {c: [] for c in classes}

    per_class_limit = None
    if max_rows is not None:
        per_class_limit = max_rows // len(classes)

    async with S3DataTool().filter_for_annotation(
        name="posts",
        annotator_name="dry_run_data_export_001",
        base_columns=["id", "text"],
        annotator_columns={"feasibility_llm_judge_001": ["is_feasible"]},
        annotator_filters={
            "feasibility_llm_judge_001": RawDuckFilter(sql="is_feasible IS NOT NULL"),
        },
    ) as annotator_view:
        with tqdm() as pbar:
            async for _row in annotator_view:
                data = _row.data
                print(data)
                label = data.get("is_feasible")

                if label not in buckets:
                    continue

                text = data["text"]
                if text in seen:
                    continue

                seen.add(text)
                buckets[label].append(data)
                pbar.update()

                if per_class_limit is not None and all(
                    len(b) >= per_class_limit for b in buckets.values()
                ):
                    break

    # Balance: downsample majority classes to match the minority
    min_count = min(len(b) for b in buckets.values())
    balanced: list[dict] = []
    for bucket in buckets.values():
        if len(bucket) > min_count:
            bucket = random.sample(bucket, min_count)
        balanced.extend(bucket)

    random.shuffle(balanced)
    return balanced


def split_and_save(
    rows: list[dict],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    """Stratified split and save to JSONL files."""
    texts = [r["text"] for r in rows]
    labels = [r["is_feasible"] for r in rows]

    # First split: train vs rest
    train_texts, rest_texts, train_labels, rest_labels = train_test_split(
        texts,
        labels,
        test_size=1 - train_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test from rest
    val_frac = val_ratio / (1 - train_ratio)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        rest_texts,
        rest_labels,
        test_size=1 - val_frac,
        stratify=rest_labels,
        random_state=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, txts, lbls in [
        ("train", train_texts, train_labels),
        ("val", val_texts, val_labels),
        ("test", test_texts, test_labels),
    ]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for text, label in zip(txts, lbls):
                f.write(json.dumps({"text": text, "is_feasible": label}) + "\n")

    for name, lbls in [
        ("train", train_labels),
        ("val", val_labels),
        ("test", test_labels),
    ]:
        dist = {i: lbls.count(i) for i in sorted(set(lbls))}
        print(f"{name}: {len(lbls)} rows  label_dist={dist}")


async def main():
    parser = argparse.ArgumentParser(description="Prepare classifier training data")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument(
        "--classes",
        type=_parse_classes,
        required=True,
        help="Comma-separated label values to include (e.g. 0,1)",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expandvars("$SCRATCH/20260331-chai-veracity/classifier_data"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Fetching data (max_rows={args.max_rows}, classes={args.classes})...")
    rows = await stream_and_collect(args.max_rows, args.classes)
    print(f"Collected {len(rows)} rows")

    class_counts: dict[int, int] = {}
    for r in rows:
        lbl = r["is_feasible"]
        class_counts[lbl] = class_counts.get(lbl, 0) + 1
    print(f"Class distribution: {class_counts}")

    print(
        f"Splitting ({args.train_ratio:.0%}/{args.val_ratio:.0%}/{1 - args.train_ratio - args.val_ratio:.0%})..."
    )
    split_and_save(rows, output_dir, args.train_ratio, args.val_ratio, args.seed)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
