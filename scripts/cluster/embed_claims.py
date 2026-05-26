import argparse
import asyncio
import logging
from typing import AsyncIterator

import backoff
import numpy as np
import openai

from s3_data_tool import S3DataTool, DataItem

logger = logging.getLogger(__name__)


def _encode_embedding(embedding: list[float]) -> str:
    import base64
    data = np.array(embedding).astype(np.float32).tobytes()
    return base64.b64encode(data).decode("utf-8")


async def _read_claims(
    source_dataset: str,
    source_filter: str | None,
) -> list[dict]:
    """Read claims from S3. Returns list of dicts with claim + metadata."""
    claims: list[dict] = []
    async with S3DataTool().filter_for_export(
        name=source_dataset,
        base_columns=["claim", "cluster_id", "id", "original_ids", "original_texts", "post_count"],
    ) as generator:
        async for row in generator:
            data = row.data
            if source_filter:
                pc = data.get("post_count", 0)
                if pc < 2:
                    continue
            claims.append({
                "claim": data["claim"],
                "cluster_id": data["cluster_id"],
                "claim_id": data["id"],
                "original_ids": data.get("original_ids", []),
                "original_texts": data.get("original_texts", []),
                "post_count": data.get("post_count", 0),
            })
    return claims


@backoff.on_exception(backoff.expo, [openai.APIConnectionError])
async def _embed_batch(
    items: list[dict],
    model_name: str,
    dimensions: int,
    oai_client: openai.AsyncOpenAI,
) -> list[dict]:
    """Embed a batch of claim texts. Returns items with embedding added."""
    texts = [item["claim"] for item in items]
    response = await oai_client.embeddings.create(
        model=model_name,
        input=texts,
        dimensions=dimensions,
    )
    for item, resp in zip(items, response.data):
        item["embedding"] = _encode_embedding(resp.embedding)
        item["embedding_dim"] = len(resp.embedding)
    return items


async def _embed_and_yield(
    claims: list[dict],
    model_name: str,
    dimensions: int,
    batch_size: int,
    batch_timeout: float,
    oai_client: openai.AsyncOpenAI,
) -> AsyncIterator[dict]:
    """Embed claims in batches and yield enriched items one at a time."""
    buffer: list[dict] = []

    async def flush():
        nonlocal buffer
        if not buffer:
            return
        batch = buffer
        buffer = []
        embedded = await _embed_batch(batch, model_name, dimensions, oai_client)
        for item in embedded:
            yield item

    for claim in claims:
        buffer.append(claim)
        if len(buffer) >= batch_size:
            async for item in flush():
                yield item

    async for item in flush():
        yield item


async def main():
    parser = argparse.ArgumentParser(description="Embed claims from summarized dataset")
    parser.add_argument("--model-name", default=None,
                        help="Model name (default: env MODEL_NAME)")
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batch-timeout", type=float, default=1.0)
    parser.add_argument("--source-dataset", default="posts_summarized_003_b")
    parser.add_argument("--target-dataset", default="posts_claims_embedded_001")
    parser.add_argument("--max-concurrency", type=int, default=36)
    parser.add_argument("--source-filter", default="post_count >= 2",
                        help="DuckDB SQL filter applied client-side (default: post_count >= 2)")
    parser.add_argument("--max-claims", type=int, default=0,
                        help="Max claims to process (0 = all)")
    args = parser.parse_args()

    model_name = args.model_name or __import__("os").environ["MODEL_NAME"]
    oai_client = openai.AsyncOpenAI()

    logger.info("Reading claims from %s (filter: %s)...", args.source_dataset, args.source_filter)
    claims = await _read_claims(args.source_dataset, args.source_filter)
    logger.info("Read %d claims.", len(claims))

    if args.max_claims > 0:
        claims = claims[:args.max_claims]
        logger.info("Limited to %d claims.", len(claims))

    if not claims:
        logger.warning("No claims to embed.")
        return

    logger.info(
        "Embedding %d claims with model=%s dims=%d batch_size=%d",
        len(claims), model_name, args.dimensions, args.batch_size,
    )

    embedded_count = 0
    async with S3DataTool().dataset_generator() as dataset_generator:
        await dataset_generator.from_async_iterator(
            _embed_and_yield(
                claims, model_name, args.dimensions,
                args.batch_size, args.batch_timeout, oai_client,
            ),
            name=args.target_dataset,
            batch="bsky-claims-embedded-001",
            streaming_configs=S3DataTool.StreamingConfigs(chunk_size=100),
            deduplicate_on=["claim_id"],
        )
        embedded_count = len(claims)

    logger.info("Done. Embedded %d claims into %s.", embedded_count, args.target_dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
