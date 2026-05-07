import argparse
import asyncio
import base64

import backoff
import numpy as np
import openai

from s3_data_tool import Annotation, DataItem, RawDuckFilter, S3DataTool

from scripts.utils.batch_coordinator import BatchCoordinator


def _encode_embeddings(embedding: list[float]) -> str:
    """Represent embedding vector as compact base64 string."""
    data = np.array(embedding).astype(np.float32).tobytes()
    return base64.b64encode(data).decode("utf-8")


def _decode_embeddings(encoded: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float32)


def get_is_valid(item: DataItem) -> bool:
    """Return whether the given item can be annotated."""
    return len(item.data["text"].strip()) > 0


class EmbeddingBatchCoordinator(BatchCoordinator):
    """Batched embedding coordinator using OpenAI-compatible API."""

    def __init__(
        self,
        oai_client: openai.AsyncOpenAI,
        model_name: str,
        dimensions: int,
        batch_size: int = 256,
        timeout: float = 1.0,
    ):
        super().__init__(batch_size=batch_size, timeout=timeout)
        self.oai_client = oai_client
        self.model_name = model_name
        self.dimensions = dimensions

    @backoff.on_exception(backoff.expo, [openai.APIConnectionError])
    async def _send_batch(
        self,
        batch: list[tuple[DataItem, asyncio.Future[Annotation]]],
    ):
        items = [item for item, _ in batch]
        futures = [future for _, future in batch]

        response = await self.oai_client.embeddings.create(
            model=self.model_name,
            input=[item.data["text"] for item in items],
            dimensions=self.dimensions,
        )

        annotations = [
            Annotation(
                data={
                    "embedding": _encode_embeddings(resp.embedding),
                    "dimensions": self.dimensions,
                },
                metadata={
                    "model_name": self.model_name,
                    "embedding_dimensions": len(resp.embedding),
                },
            )
            for resp in response.data
        ]

        for future, annotation in zip(futures, annotations):
            future.set_result(annotation)

    async def annotate(self, item: DataItem) -> Annotation:
        if not get_is_valid(item):
            raise ValueError(f"Item value invalid: {item.data}")
        return await super().annotate(item)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_concurrency", type=int, default=128)
    parser.add_argument(
        "--dimensions",
        type=int,
        default=128,
        help=(
            "Matryoshka embedding dimensions. If None, uses full embedding size. "
            "Example values: 128, 256, 512, 1024 depending on model support."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of texts to batch into a single API request.",
    )
    parser.add_argument(
        "--batch_timeout",
        type=float,
        default=1.0,
        help="Seconds to wait before flushing an incomplete batch.",
    )
    args = parser.parse_args()
    oai_client = openai.AsyncOpenAI()

    annotator_name = f"embeddings_{args.dimensions}d"

    coordinator = EmbeddingBatchCoordinator(
        oai_client=oai_client,
        model_name=args.model_name,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
        timeout=args.batch_timeout,
    )
    coordinator.start()

    async with (
        S3DataTool().filter_for_annotation(
            name="posts",
            annotator_name=annotator_name,
            base_columns=["text"],
            annotator_columns={"feasibility_classifier_001": ["classifier_label"]},
            annotator_filters={
                "feasibility_classifier_001": RawDuckFilter(
                    sql="classifier_label = '\"LABEL_1\"'",  # raw values are JSON-encoded strings: "LABEL_1"
                ),
            },
            base_filter=RawDuckFilter(sql="length(text) > 0"),
            fraction=0.1,
        ) as annotator_view
    ):
        await annotator_view.annotate(
            lambda item: coordinator.annotate(item),
            max_concurrency=args.max_concurrency,
            streaming_configs=S3DataTool.StreamingConfigs(chunk_size=10),
        )

    await coordinator.close()


if __name__ == "__main__":
    asyncio.run(main())
