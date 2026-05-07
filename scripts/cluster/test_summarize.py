"""Test harness for the claim-extraction prompt using real S3 clusters.

Pulls real clusters from posts_clustered_002, runs the new prompt, prints and
writes results as they arrive.
"""

import asyncio
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".llm-test.env")

TEMPLATE_PATH = Path(__file__).resolve().parent.parent.parent / "templates" / "summarize_cluster.txt"
TEMPLATE = TEMPLATE_PATH.read_text()

MAX_CLUSTERS = 3
MAX_POSTS_PER_CLUSTER = 15
OUT_PATH = Path("/tmp/claim_extraction_test.txt")


async def main():
    from s3_data_tool import S3DataTool

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B")

    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    # Clear output file
    OUT_PATH.write_text("")

    async with S3DataTool().filter_for_export(
        name="posts_clustered_002",
        base_columns=["text", "original_ids"],
    ) as generator:
        count = 0
        async for item in generator:
            texts = item.data.get("text", [])
            original_ids = item.data.get("original_ids", [])
            if not texts:
                continue

            texts = texts[:MAX_POSTS_PER_CLUSTER]
            original_ids = original_ids[:MAX_POSTS_PER_CLUSTER]

            numbered = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
            prompt = TEMPLATE.format(posts=numbered)

            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            raw = response.choices[0].message.content or "(empty)"

            block = []
            block.append(f"{'='*80}")
            block.append(f"CLUSTER: {item.id}  ({len(texts)} posts)")
            block.append(f"{'='*80}")
            block.append("")
            block.append("INPUT POSTS:")
            for i, (t, oid) in enumerate(zip(texts, original_ids)):
                block.append(f"  [{i}] (id={oid}) {t}")
            block.append("")
            block.append("LLM OUTPUT:")
            block.append(raw)
            block.append("")

            chunk = "\n".join(block)
            print(chunk)

            with open(OUT_PATH, "a") as f:
                f.write(chunk)

            count += 1
            if count >= MAX_CLUSTERS:
                break

    print(f"\nResults also written to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
