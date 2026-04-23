"""
Fetch tweets for usernames in a CSV.

Modes:
- default: upload to S3 dataset
- local test: write rows to a local CSV with --local-csv

Expected input CSV:
- a column named "username" by default
"""

import argparse
import asyncio
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from s3_data_tool import S3DataTool

# Fail fast instead of waiting forever when no account is available.
os.environ["TWS_RAISE_WHEN_NO_ACCOUNT"] = "1"


from twscrape import API, gather

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = str(BASE_DIR / "accounts.db")


def _load_usernames(csv_path: str, username_column: str) -> list[str]:
    usernames: list[str] = []
    seen: set[str] = set()

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if username_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{username_column}' not found. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            raw = (row.get(username_column) or "").strip()
            if not raw:
                continue

            username = raw.lstrip("@").strip()
            if not username:
                continue

            key = username.lower()
            if key in seen:
                continue

            seen.add(key)
            usernames.append(username)

    return usernames


def _tweet_to_row(source_username: str, tweet: Any) -> dict[str, Any]:
    text = (getattr(tweet, "rawContent", None) or "").replace("\r", " ").replace("\n", " ").strip()
    tweet_id = str(getattr(tweet, "id", ""))
    date = str(getattr(tweet, "date", "")) if getattr(tweet, "date", None) else ""

    return {
        "source": "x",
        "source_username": source_username,
        "source_id": f"x:{source_username}:{tweet_id}",
        "tweet_id": tweet_id,
        "date": date,
        "text": text,
        "raw": tweet.dict() if hasattr(tweet, "dict") else None,
    }


async def _fetch_one_user(
    api: API,
    username: str,
    tweets_per_user: int,
    skip_retweets: bool,
    skip_replies: bool,
) -> list[dict[str, Any]]:
    user = await api.user_by_login(username)
    tweets = await gather(api.user_tweets(user.id, limit=tweets_per_user))

    rows: list[dict[str, Any]] = []
    for tweet in tweets:
        if skip_retweets and getattr(tweet, "retweetedTweet", None):
            continue

        if skip_replies and getattr(tweet, "inReplyToTweetId", None):
            continue

        rows.append(_tweet_to_row(username, tweet))

    return rows


async def _get_iterator(
    api: API,
    input_csv: str,
    username_column: str,
    tweets_per_user: int,
    skip_retweets: bool,
    skip_replies: bool,
) -> AsyncGenerator[dict[str, Any], None]:
    usernames = _load_usernames(input_csv, username_column)

    for i, username in enumerate(usernames, start=1):
        print(f"[{i}/{len(usernames)}] Fetching tweets for @{username} ...")

        try:
            rows = await _fetch_one_user(
                api=api,
                username=username,
                tweets_per_user=tweets_per_user,
                skip_retweets=skip_retweets,
                skip_replies=skip_replies,
            )

            print(f"  Saved {len(rows)} tweets")
            for row in rows:
                yield row

        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            yield {
                "source": "x",
                "source_username": username,
                "source_id": f"x:{username}:error",
                "tweet_id": None,
                "date": None,
                "text": None,
                "raw": None,
                "error": f"{type(e).__name__}: {e}",
            }


async def _write_local_csv(
    output_csv: str,
    iterator: AsyncGenerator[dict[str, Any], None],
) -> int:
    count = 0

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "source_username",
                "source_id",
                "tweet_id",
                "date",
                "text",
                "error",
            ],
        )
        writer.writeheader()

        async for row in iterator:
            writer.writerow(
                {
                    "source": row.get("source"),
                    "source_username": row.get("source_username"),
                    "source_id": row.get("source_id"),
                    "tweet_id": row.get("tweet_id"),
                    "date": row.get("date"),
                    "text": row.get("text"),
                    "error": row.get("error"),
                }
            )
            count += 1

    return count


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="unique_usernames.csv")
    parser.add_argument("--username-column", default="username")
    parser.add_argument("--tweets-per-user", type=int, default=20)
    parser.add_argument("--dataset-name", default="posts")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--batch-prefix", default="x-user-tweets")
    parser.add_argument("--skip-retweets", action="store_true")
    parser.add_argument("--skip-replies", action="store_true")

    # Local test mode
    parser.add_argument("--local-csv", default=None,
                        help="Write results to a local CSV instead of uploading to S3")

    args = parser.parse_args()

    api = API(args.db_path)
    timestamp = datetime.now().strftime("%Y%m%d-%H")
    batch_name = f"{args.batch_prefix}-{timestamp}"

    iterator = _get_iterator(
        api=api,
        input_csv=args.input_csv,
        username_column=args.username_column,
        tweets_per_user=args.tweets_per_user,
        skip_retweets=args.skip_retweets,
        skip_replies=args.skip_replies,
    )

    # Local test mode
    if args.local_csv:
        total_rows = await _write_local_csv(args.local_csv, iterator)
        print(f"\nDone. Wrote {total_rows} rows to local CSV: {args.local_csv}")
        return

    # Default: S3 upload mode
    async with S3DataTool().dataset_generator() as dataset_generator:
        await dataset_generator.from_async_iterator(
            iterator,
            name=args.dataset_name,
            batch=batch_name,
            streaming_configs=S3DataTool.StreamingConfigs(chunk_size=100),
            deduplicate_on=["text", "source_id"],
        )

    print(f"\nDone. Uploaded batch: {batch_name}")


if __name__ == "__main__":
    asyncio.run(main())