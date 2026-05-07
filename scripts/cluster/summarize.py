import argparse
import asyncio
import logging
import re
from datetime import datetime
from typing import AsyncIterator, Any

import backoff
import openai
import yaml
from pydantic import BaseModel

from s3_data_tool import S3DataTool, DataItem

logger = logging.getLogger(__name__)

with open("templates/summarize_cluster.txt") as f:
    TEMPLATE = f.read()


class Claim(BaseModel):
    claim: str
    post_indices: list[int]


class ClusterSummary(BaseModel):
    claims: list[Claim]


_CLAIM_BLOCK_RE = re.compile(
    r'- claim:\s*"([^"]*)"\s*\n\s*post_indices:\s*\[([^\]]*)\]'
)


def _parse_output(output: str) -> ClusterSummary:
    """Parse LLM output, discarding incomplete trailing entries."""
    output = output.removeprefix("```").removeprefix("yaml").removesuffix("```").strip()
    try:
        data = yaml.safe_load(output)
        return ClusterSummary.model_validate(data)
    except Exception:
        pass

    # Fallback: regex-extract only complete claim blocks (bounded by - and ])
    claims: list[dict[str, Any]] = []
    for m in _CLAIM_BLOCK_RE.finditer(output):
        claim_text = m.group(1)
        indices_str = m.group(2)
        try:
            indices = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
        except ValueError:
            continue
        claims.append({"claim": claim_text, "post_indices": indices})

    return ClusterSummary.model_validate({"claims": claims})


@backoff.on_exception(backoff.expo, [openai.APIConnectionError])
async def _generate(
    texts: list[str],
    model_name: str,
    oai_client: openai.AsyncOpenAI,
    max_tokens: int,
) -> ClusterSummary:
    """Call LLM to extract claims from a cluster. Raises on failure."""
    numbered = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    prompt = TEMPLATE.format(posts=numbered)
    response = await oai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    output = response.choices[0].message.content
    assert output is not None

    return _parse_output(output)


async def _summarize_with_retries(
    texts: list[str],
    model_name: str,
    oai_client: openai.AsyncOpenAI,
    max_retries: int,
    max_tokens: int,
) -> ClusterSummary:
    exceptions = []
    for _ in range(max_retries):
        try:
            return await _generate(
                texts, model_name=model_name, oai_client=oai_client, max_tokens=max_tokens
            )
        except Exception as e:
            exceptions.append(e)
    raise RuntimeError(exceptions)


async def _get_topic_iterator(
    model_name: str,
    oai_client: openai.AsyncOpenAI,
    max_concurrency: int,
    max_retries: int,
    max_tokens: int,
) -> AsyncIterator[dict[str, Any]]:
    """Stream clusters, summarize each, yield one row per topic."""
    sem = asyncio.Semaphore(max_concurrency)

    async def _process(item: DataItem) -> list[dict[str, Any]]:
        texts = item.data.get("text", [])
        original_ids = item.data.get("original_ids", [])
        if not texts:
            return []

        async with sem:
            result = await _summarize_with_retries(
                texts, model_name, oai_client, max_retries, max_tokens
            )

        rows: list[dict[str, Any]] = []
        for claim in result.claims:
            claim_ids = [
                original_ids[i]
                for i in claim.post_indices
                if i < len(original_ids)
            ]
            rows.append({
                "cluster_id": item.id,
                "claim": claim.claim,
                "post_count": len(claim_ids),
                "original_ids": claim_ids,
            })
        return rows

    async with S3DataTool().filter_for_export(
        name="posts_clustered_002",
        base_columns=["text", "original_ids"],
    ) as generator:
        tasks = []
        async for item in generator:
            tasks.append(asyncio.create_task(_process(item)))

        for task in asyncio.as_completed(tasks):
            for row in await task:
                yield row


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_concurrency", type=int, default=16)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--dataset-name", default="posts_summarized_001_dry_run")
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H")
    batch_name = f"bsky-summarize-{timestamp}"

    oai_client = openai.AsyncOpenAI()

    async with S3DataTool().dataset_generator() as dataset_generator:
        await dataset_generator.from_async_iterator(
            _get_topic_iterator(
                model_name=args.model_name,
                oai_client=oai_client,
                max_concurrency=args.max_concurrency,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
            ),
            name=args.dataset_name,
            batch=batch_name,
            streaming_configs=S3DataTool.StreamingConfigs(chunk_size=10),
            deduplicate_on=["cluster_id", "claim"],
        )

    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
