import argparse
import asyncio
import base64
from dataclasses import dataclass, field

import backoff
import openai
import numpy as np

from s3_data_tool import AllFilter, Annotation, DataItem, S3DataTool, RawDuckFilter


def _encode_embeddings(embedding: list[float]) -> str:
    """Represent embedding vector as compact base64 string."""
    data = np.array(embedding).astype(np.float32).tobytes()
    return base64.b64encode(data).decode("utf-8")


def _decode_embeddings(encoded: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float32)


def get_is_valid(item: DataItem) -> bool:
    """Return whether the given item can be annotated."""
    return len(item.data["text"].strip()) > 0


@dataclass
class BatchCoordinator:
    """Batches embedding requests up to batch_size or timeout."""

    oai_client: openai.AsyncOpenAI
    model_name: str
    dimensions: int
    batch_size: int = 256
    timeout: float = 1.0

    _queue: asyncio.Queue[tuple[DataItem, asyncio.Future[Annotation]]] = field(
        default_factory=asyncio.Queue
    )
    _flush_task: asyncio.Task[None] | None = None

    def start(self):
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self):
        while True:
            batch: list[tuple[DataItem, asyncio.Future[Annotation]]] = []

            # Wait for first item, then fill batch until size limit or deadline
            try:
                item_and_future = await asyncio.wait_for(
                    self._queue.get(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                continue
            batch.append(item_and_future)

            while len(batch) < self.batch_size:
                try:
                    remaining = self.timeout
                    item_and_future = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch.append(item_and_future)
                except asyncio.TimeoutError:
                    break

            await self._send_batch(batch)

    async def _send_batch(
        self,
        batch: list[tuple[DataItem, asyncio.Future[Annotation]]],
    ):
        items = [item for item, _ in batch]
        futures = [future for _, future in batch]

        try:
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

        except Exception as e:
            for future in futures:
                future.set_exception(e)

    async def annotate(self, item: DataItem) -> Annotation:
        future: asyncio.Future[Annotation] = asyncio.get_event_loop().create_future()
        if not get_is_valid(item):
            raise ValueError(f"Item value invalid: {item.data}")
        await self._queue.put((item, future))
        return await future

    async def close(self):
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass


@backoff.on_exception(backoff.expo, [openai.APIConnectionError])
async def _generate(
    item: DataItem,
    model_name: str,
    oai_client: openai.AsyncOpenAI,
    dimensions: int | None,
) -> Annotation:
    """Generate embedding for text.

    This function is designed to raise exceptions eagerly.
    """
    if dimensions is not None:
        response = await oai_client.embeddings.create(
            model=model_name,
            input=item.data["text"],
            dimensions=dimensions,
        )
    else:
        response = await oai_client.embeddings.create(
            model=model_name,
            input=item.data["text"],
        )
    embedding = response.data[0].embedding

    return Annotation(
        data={
            "embedding": embedding,
            "dimensions": dimensions,
        },
        metadata={
            "model_name": model_name,
            "embedding_dimensions": len(embedding),
        },
    )


async def annotate(
    item: DataItem,
    model_name: str,
    oai_client: openai.AsyncOpenAI,
    max_retries: int,
    dimensions: int | None,
) -> Annotation:
    """Annotate- retry if inner generation loop does not work."""
    exceptions = []
    for _ in range(max_retries):
        try:
            return await _generate(
                item,
                model_name=model_name,
                oai_client=oai_client,
                dimensions=dimensions,
            )
        except Exception as e:
            exceptions.append(e)

    raise RuntimeError(exceptions)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_concurrency", type=int, default=128)
    parser.add_argument("--max_retries", type=int, default=6)
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

    coordinator = BatchCoordinator(
        oai_client=oai_client,
        model_name=args.model_name,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
        timeout=args.batch_timeout,
    )
    coordinator.start()

    async with S3DataTool().filter_for_annotation(
        name="posts",
        annotator_name=annotator_name,
        base_columns=["text"],
        annotator_columns={"feasibility_001": ["is_feasible"]},
        annotator_filters={
            "feasibility_001": RawDuckFilter(sql="is_feasible >= 1"),
        },
        base_filter=RawDuckFilter(sql="length(text) > 0"),
    ) as annotator_view:
        await annotator_view.annotate(
            lambda item: coordinator.annotate(item),
            max_concurrency=args.max_concurrency,
            streaming_configs=S3DataTool.StreamingConfigs(chunk_size=10),
        )

    await coordinator.close()


if __name__ == "__main__":
    asyncio.run(main())
