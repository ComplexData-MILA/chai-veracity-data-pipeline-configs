import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from s3_data_tool import Annotation, DataItem


@dataclass
class BatchCoordinator(ABC):
    """Batches inference requests up to batch_size or timeout.

    Subclasses implement _send_batch to call their specific API.
    """

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
                    item_and_future = await asyncio.wait_for(
                        self._queue.get(), timeout=self.timeout
                    )
                    batch.append(item_and_future)
                except asyncio.TimeoutError:
                    break

            try:
                await self._send_batch(batch)
            except Exception as exc:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)

    @abstractmethod
    async def _send_batch(
        self,
        batch: list[tuple[DataItem, asyncio.Future[Annotation]]],
    ):
        """Send a batch of items to the inference API.

        Must call future.set_result() or future.set_exception() for each item.
        """

    async def annotate(self, item: DataItem) -> Annotation:
        future: asyncio.Future[Annotation] = asyncio.get_event_loop().create_future()
        await self._queue.put((item, future))
        return await future

    async def close(self):
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
