from __future__ import annotations

import argparse
import datetime
import asyncio
import json
import random
from typing import Any
from collections import deque
from typing import AsyncIterator, AsyncIterable, Iterable, TypeVar
from urllib.parse import urlencode

import websockets
import pydantic
from websockets.exceptions import ConnectionClosed

from s3_data_tool import S3DataTool


class BlueskyPost(pydantic.BaseModel):
    did: str
    at_uri: str
    rkey: str
    cid: str | None
    text: str | None
    created_at: str | None
    raw: dict


class BlueskyJetstreamPosts(AsyncIterator[dict[str, Any]]):
    """
    Async iterator over real-time public Bluesky posts via Jetstream.

    Yields only post creation events by default.
    Set include_updates=True if you also want edited posts.
    """

    def __init__(
        self,
        endpoints: Iterable[str] | None = None,
        *,
        cursor_us: int | None = None,
        include_updates: bool = False,
        rewind_seconds: int = 3,
        max_seen: int = 10_000,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
        open_timeout: float = 20.0,
        min_backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        self._endpoints = tuple(
            endpoints
            or (
                "wss://jetstream1.us-east.bsky.network/subscribe",
                "wss://jetstream2.us-east.bsky.network/subscribe",
                "wss://jetstream1.us-west.bsky.network/subscribe",
                "wss://jetstream2.us-west.bsky.network/subscribe",
            )
        )
        if not self._endpoints:
            raise ValueError("At least one Jetstream endpoint is required.")

        self._cursor_us = cursor_us
        self._include_updates = include_updates
        self._rewind_us = max(rewind_seconds, 0) * 1_000_000
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._open_timeout = open_timeout
        self._min_backoff = min_backoff
        self._max_backoff = max_backoff

        self._seen = set()
        self._seen_order = deque(maxlen=max_seen)

        self._ws = None
        self._endpoint_index = 0
        self._closed = False
        self._backoff = min_backoff

    def __aiter__(self) -> "BlueskyJetstreamPosts":
        return self

    async def __aenter__(self) -> "BlueskyJetstreamPosts":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self._closed = True
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def __anext__(self) -> dict[str, Any]:
        while not self._closed:
            try:
                await self._ensure_connected()
                assert self._ws is not None

                message = await self._ws.recv()
                if isinstance(message, bytes):
                    message = message.decode("utf-8")

                event = json.loads(message)
                post = self._parse_event(event)
                if post is None:
                    continue
                return post.model_dump()

            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, OSError, TimeoutError, json.JSONDecodeError):
                await self._reset_connection()
                await self._sleep_with_backoff()

        raise StopAsyncIteration

    async def _ensure_connected(self) -> None:
        if self._ws is not None:
            return

        last_error: Exception | None = None
        for _ in range(len(self._endpoints)):
            endpoint = self._next_endpoint()
            url = self._build_url(endpoint)
            try:
                self._ws = await websockets.connect(
                    url,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    open_timeout=self._open_timeout,
                    max_size=None,
                )
                self._backoff = self._min_backoff
                return
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error

    async def _reset_connection(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None

    async def _sleep_with_backoff(self) -> None:
        delay = min(self._backoff, self._max_backoff)
        delay *= 1.0 + random.random() * 0.2
        await asyncio.sleep(delay)
        self._backoff = min(self._backoff * 2.0, self._max_backoff)

    def _next_endpoint(self) -> str:
        endpoint = self._endpoints[self._endpoint_index % len(self._endpoints)]
        self._endpoint_index += 1
        return endpoint

    def _build_url(self, endpoint: str) -> str:
        params: list[tuple[str, str]] = [("wantedCollections", "app.bsky.feed.post")]

        if self._cursor_us is not None:
            replay_from = max(self._cursor_us - self._rewind_us, 0)
            params.append(("cursor", str(replay_from)))

        return f"{endpoint}?{urlencode(params, doseq=True)}"

    def _parse_event(self, event: dict) -> BlueskyPost | None:
        time_us = event.get("time_us")
        if isinstance(time_us, int):
            if self._cursor_us is None or time_us > self._cursor_us:
                self._cursor_us = time_us

        if event.get("kind") != "commit":
            return None

        commit = event.get("commit")
        if not isinstance(commit, dict):
            return None

        if commit.get("collection") != "app.bsky.feed.post":
            return None

        operation = commit.get("operation")
        allowed_ops = {"create", "update"} if self._include_updates else {"create"}
        if operation not in allowed_ops:
            return None

        record = commit.get("record")
        if not isinstance(record, dict):
            return None

        if record.get("$type") != "app.bsky.feed.post":
            return None

        did = event.get("did")
        rkey = commit.get("rkey")
        if not isinstance(did, str) or not isinstance(rkey, str):
            return None

        at_uri = f"at://{did}/app.bsky.feed.post/{rkey}"

        text = record.get("text")
        created_at = record.get("createdAt")
        cid = commit.get("cid")

        return BlueskyPost(
            did=did,
            at_uri=at_uri,
            rkey=rkey,
            cid=cid if isinstance(cid, str) else None,
            text=str(text) if text else None,
            created_at=created_at if isinstance(created_at, str) else None,
            raw=event,
        )

T = TypeVar("T")


async def iter_until_timeout(
    source: AsyncIterable[T], timeout: float
) -> AsyncIterator[T]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    it = aiter(source)

    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            return

        try:
            yield await asyncio.wait_for(anext(it), timeout=remaining)
        except (StopAsyncIteration, asyncio.TimeoutError):
            return


async def main() -> None:
    """Generate from bluesky top/trending posts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dataset-name", default="posts")
    args = parser.parse_args()

    # Load secrets from env.
    while True:
        async with S3DataTool().dataset_generator() as dataset_generator:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H")
            batch_name = f"bsky-jetstream-{timestamp}"
            print(batch_name)

            async with BlueskyJetstreamPosts() as stream:
                await dataset_generator.from_async_iterator(
                    iter_until_timeout(stream, timeout=3600),
                    name=args.dataset_name,
                    batch=batch_name,
                    streaming_configs=S3DataTool.StreamingConfigs(chunk_size=100),
                    deduplicate_on=["text"],  # list of columns
                )


if __name__ == "__main__":
    asyncio.run(main())
