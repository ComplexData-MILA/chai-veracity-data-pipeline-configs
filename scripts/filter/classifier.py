import argparse
import asyncio

import backoff
import httpx

from s3_data_tool import Annotation, DataItem, S3DataTool

from scripts.utils.batch_coordinator import BatchCoordinator

# httpx errors worth retrying
TRANSIENT_HTTPX_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
)


class ClassifierBatchCoordinator(BatchCoordinator):
    """Batched classification coordinator using vLLM classify API."""

    def __init__(
        self,
        model_name: str,
        client: httpx.AsyncClient,
        base_url: str,
        batch_size: int = 256,
        timeout: float = 1.0,
    ):
        super().__init__(batch_size=batch_size, timeout=timeout)
        self.model_name = model_name
        self.client = client
        self.base_url = base_url

    @backoff.on_exception(backoff.expo, TRANSIENT_HTTPX_ERRORS)
    async def _send_batch(
        self,
        batch: list[tuple[DataItem, asyncio.Future[Annotation]]],
    ):
        items = [item for item, _ in batch]
        futures = [future for _, future in batch]

        response = await self.client.post(
            f"{self.base_url}/classify",
            json={
                "model": self.model_name,
                "input": [item.data["text"] for item in items],
            },
        )
        response.raise_for_status()
        body = response.json()

        annotations = [
            Annotation(
                data={
                    "classifier_label": result["label"],
                    "classifier_probs": result["probs"],
                },
                metadata={"model_name": self.model_name},
            )
            for result in body["data"]
        ]

        for future, annotation in zip(futures, annotations):
            future.set_result(annotation)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8000")
    parser.add_argument("--max_concurrency", type=int, default=128)
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
    parser.add_argument(
        "--batch",
        default=None,
        help="Process only this specific batch name (sets fraction=1.0).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Override the fraction of rows to annotate (default: 0.01 normally, 1.0 when --batch is set).",
    )
    args = parser.parse_args()

    fraction = args.fraction if args.fraction is not None else (1.0 if args.batch else 0.01)

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        coordinator = ClassifierBatchCoordinator(
            model_name=args.model_name,
            client=client,
            base_url=args.base_url,
            batch_size=args.batch_size,
            timeout=args.batch_timeout,
        )
        coordinator.start()

        async with S3DataTool().filter_for_annotation(
            name="posts",
            annotator_name="feasibility_classifier_001",
            base_columns=["text"],
            fraction=fraction,
        ) as annotator_view:
            await annotator_view.annotate(
                lambda item: coordinator.annotate(item),
                max_concurrency=args.max_concurrency,
                streaming_configs=S3DataTool.StreamingConfigs(chunk_size=10),
                batches=[args.batch] if args.batch else None,
            )

        await coordinator.close()


if __name__ == "__main__":
    asyncio.run(main())
