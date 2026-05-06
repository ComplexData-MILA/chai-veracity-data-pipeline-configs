import argparse
import asyncio
import re

import backoff
import openai
from s3_data_tool import Annotation, DataItem, S3DataTool

with open("templates/feasibility_filter.txt") as template_file:
    TEMPLATE = template_file.read()


def _parse_verdict(output: str) -> tuple[str, int]:
    """Extract explanation and feasibility rating from model output.

    The model is prompted to output: explanation | rating (0-2).
    Parsing is lenient: the rating is the first 0, 1, or 2 found after
    the last pipe delimiter, extracted via regex.
    """
    parts = output.split("|")
    if len(parts) < 2:
        raise ValueError(
            f"Output missing '|' delimiter. Output: {output[:200]}"
        )
    explanation = "|".join(parts[:-1])
    verdict_str = parts[-1]
    match = re.search(r"\b([0-2])\b", verdict_str)
    if match is None:
        raise ValueError(
            f"Could not parse feasibility rating from: {verdict_str[:200]}"
        )
    return explanation, int(match.group(1))


async def _generate(
    item: DataItem, model_name: str, oai_client: openai.AsyncOpenAI
) -> Annotation:
    """Annotate using LLM and template.

    This function is designed to raise exceptions eagerly.
    """
    if len(item.data["text"].strip()) == 0:
        return Annotation(data={"explanation": "Empty text."})

    prompt = TEMPLATE.format(text=item.data["text"])
    response = await oai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=16384,
    )
    output = response.choices[0].message.content
    if output is None:
        finish_reason = response.choices[0].finish_reason
        reasoning = getattr(
            response.choices[0].message, "reasoning_content", None
        )
        raise ValueError(
            f"Model returned no content. finish_reason={finish_reason}, "
            f"reasoning_content present: {reasoning is not None}"
        )

    explanation, verdict_int = _parse_verdict(output)

    return Annotation(
        data={"is_feasible": verdict_int, "explanation": explanation},
        metadata={"model_name": model_name},
    )


async def annotate(
    item: DataItem, model_name: str, oai_client: openai.AsyncOpenAI, max_retries: int
) -> Annotation:
    """Annotate- retry if inner generation loop does not work."""
    exceptions = []
    for _ in range(max_retries):
        try:
            return await _generate(item, model_name=model_name, oai_client=oai_client)
        except Exception as e:
            exceptions.append(e)

    raise RuntimeError(exceptions)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_concurrency", type=int, default=128)
    parser.add_argument("--max_retries", type=int, default=6)
    args = parser.parse_args()
    oai_client = openai.AsyncOpenAI()

    async with S3DataTool().filter_for_annotation(
        name="posts",
        annotator_name="feasibility_llm_judge_001",
        base_columns=["text"],
    ) as annotator_view:
        await annotator_view.annotate(
            lambda item: annotate(
                item,
                model_name=args.model_name,
                oai_client=oai_client,
                max_retries=args.max_retries,
            ),
            max_concurrency=args.max_concurrency,
            streaming_configs=S3DataTool.StreamingConfigs(chunk_size=10),
        )


if __name__ == "__main__":
    asyncio.run(main())
