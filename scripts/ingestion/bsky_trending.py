"""
Fetch bluesky trending posts and add to dataset.

Batch names are generated automatically.
"""

import argparse
import asyncio
from datetime import datetime
from typing import Any, AsyncGenerator

import httpx

from s3_data_tool import S3DataTool

BLUESKY_API_BASE = "https://public.api.bsky.app/xrpc/app.bsky.feed.getFeed"
BLUESKY_FEED_URI = (
    "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot"
)


def _extract_text(feed_item: dict[str, Any]) -> dict[str, Any]:
    """Extract text and id from bsky feed item."""
    return {
        "uri": feed_item["post"]["uri"],
        "text": feed_item["post"]["record"].get("text", None),
        "langs": feed_item["post"]["record"].get("langs", []),
        "raw": feed_item,
    }


async def _fetch_bluesky_page(
    client: httpx.AsyncClient, limit: int, cursor: str | None = None
):
    params = {"feed": BLUESKY_FEED_URI, "limit": limit}
    if cursor:
        params["cursor"] = cursor

    response = await client.get(BLUESKY_API_BASE, params=params, timeout=60)
    response.raise_for_status()

    data: dict[str, Any] = response.json()
    feed: list[dict[str, Any]] = data.get("feed", [])
    next_cursor = data.get("cursor")
    return list(map(_extract_text, feed)), next_cursor


async def _get_iterator(
    client: httpx.AsyncClient, limit: int
) -> AsyncGenerator[dict[str, Any], None]:
    """Iterator for bsky feed, yielding one item at a time."""
    cursor = None
    count = 0

    while count < limit:
        feed, cursor = await _fetch_bluesky_page(client, limit=limit, cursor=cursor)
        for item in feed:
            count += 1
            yield item

        if not cursor:
            break


async def main():
    """Generate from bluesky top/trending posts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dataset-name", default="posts")
    args = parser.parse_args()

    httpx_client = httpx.AsyncClient()
    timestamp = datetime.now().strftime("%Y%m%d-%H")
    batch_name = f"bsky-trending-{timestamp}"

    # Load secrets from env.
    async with S3DataTool().dataset_generator() as dataset_generator:
        # Add rows to the dataset named "example_dataset"
        await dataset_generator.from_async_iterator(
            _get_iterator(httpx_client, limit=args.limit),
            name=args.dataset_name,
            batch=batch_name,
            streaming_configs=S3DataTool.StreamingConfigs(chunk_size=100),
            deduplicate_on=["text", "source_id"],  # list of columns
        )


if __name__ == "__main__":
    asyncio.run(main())
