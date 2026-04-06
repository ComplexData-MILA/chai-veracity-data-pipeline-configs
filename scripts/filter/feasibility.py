import argparse
import asyncio

import backoff
import openai
from s3_data_tool import Annotation, DataItem, S3DataTool

with open("scripts/filter/template.txt") as template_file:
    TEMPLATE = template_file.read()

@backoff.on_exception(backoff.expo, [openai.APIConnectionError])
async def _generate(
    item: DataItem, model_name: str, oai_client: openai.AsyncOpenAI
) -> Annotation:
    """Annotate using LLM and template.

    This function is designed to raise exceptions eagerly.
    """
    prompt = TEMPLATE.format(text=item.data["text"])
    response = await oai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    output = response.choices[0].message.content
    assert output is not None

    *_explanations, _verdict = output.split("|")
    explanation = "\n".join(_explanations)
    verdict_int = int(_verdict)

    return Annotation(data={"is_feasible": verdict_int, "explanation": explanation})


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
        annotator_name="feasibility_001",
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
