"""Fetch a handful of clusters from S3 and save locally for prompt iteration."""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from s3_data_tool import S3DataTool

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset", default="posts_clustered_dbscan_002")
    parser.add_argument("--text-column", default="central_sample_texts")
    parser.add_argument("--clusters-per-batch", type=int, default=3)
    parser.add_argument("--max-texts-per-cluster", type=int, default=32)
    parser.add_argument("--batches", nargs="+",
                        default=[f"bsky-trending-202605{i:02d}" for i in range(3, 11)])
    parser.add_argument("--out", default="data/sample_clusters.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    batch_counts: dict[str, int] = {b: 0 for b in args.batches}

    async with S3DataTool().filter_for_export(
        name=args.input_dataset,
        base_columns=[args.text_column],
    ) as generator:
        async for item in generator:
            if item.batch not in batch_counts:
                continue
            if batch_counts[item.batch] >= args.clusters_per_batch:
                # Check if all batches are full
                if all(c >= args.clusters_per_batch for c in batch_counts.values()):
                    break
                continue

            texts = item.data.get(args.text_column, [])
            if not texts:
                continue

            texts = texts[:args.max_texts_per_cluster]
            samples.append({
                "batch": item.batch,
                "cluster_id": item.id,
                "texts": texts,
            })
            batch_counts[item.batch] += 1
            print(f"[{item.batch}] cluster={item.id}  texts={len(texts)}")

    out_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(samples)} clusters to {out_path}")
    for b in args.batches:
        print(f"  {b}: {batch_counts[b]}")


if __name__ == "__main__":
    asyncio.run(main())
