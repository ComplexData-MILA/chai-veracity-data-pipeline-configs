#!/usr/bin/env python3
"""Burst detection on Bluesky reaction events (likes, reposts).

Three operating modes:

1. Live collection + analysis:
       python scripts/analysis/detect_bursts.py --duration-minutes 30

2. Local JSONL analysis:
       python scripts/analysis/detect_bursts.py --reactions-jsonl output/reactions.jsonl

3. S3 dataset analysis (offline / daily):
       python scripts/analysis/detect_bursts.py \\
           --reactions-dataset reactions \\
           --posts-dataset posts \\
           --batch-pattern "bsky-jetstream-reactions-20260513*"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

# Add repo root to path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from s3_data_tool import S3DataTool, DataItem
from scripts.analysis.kleinberg import detect_bursts, BurstResult

# ---------------------------------------------------------------------------
# Jetstream reaction collector (live mode)
# ---------------------------------------------------------------------------

JETSTREAM_ENDPOINTS = (
    "wss://jetstream1.us-east.bsky.network/subscribe",
    "wss://jetstream2.us-east.bsky.network/subscribe",
    "wss://jetstream1.us-west.bsky.network/subscribe",
    "wss://jetstream2.us-west.bsky.network/subscribe",
)

WANTED_COLLECTIONS = ["app.bsky.feed.like", "app.bsky.feed.repost"]


@dataclass
class ReactionEvent:
    event_type: str          # "like" or "repost"
    subject_uri: str         # at-uri of the post being reacted to
    actor_did: str           # who performed the reaction
    time_us: int             # Jetstream timestamp (microseconds)
    created_at: str | None   # ISO timestamp from the record


async def collect_reactions(
    duration_seconds: float,
    output_path: str,
    max_events: int = 500_000,
) -> int:
    """Collect reaction events from Jetstream for a fixed duration.

    Returns the number of events collected.
    """
    import random

    try:
        import websockets
    except ImportError:
        print("Error: 'websockets' package required. Install with: pip install websockets")
        sys.exit(1)

    from urllib.parse import urlencode

    params: list[tuple[str, str]] = []
    for col in WANTED_COLLECTIONS:
        params.append(("wantedCollections", col))

    query_string = urlencode(params, doseq=True)

    events: list[dict[str, Any]] = []
    endpoint_idx = 0
    backoff = 1.0

    deadline = asyncio.get_running_loop().time() + duration_seconds

    print(f"Collecting reactions for {duration_seconds / 60:.0f} minutes...")
    print(f"  Output: {output_path}")

    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        if len(events) >= max_events:
            print(f"Reached max_events limit ({max_events})")
            break

        endpoint = JETSTREAM_ENDPOINTS[endpoint_idx % len(JETSTREAM_ENDPOINTS)]
        endpoint_idx += 1
        url = f"{endpoint}?{query_string}"

        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                open_timeout=20,
                max_size=None,
            ) as ws:
                backoff = 1.0
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    if len(events) >= max_events:
                        break

                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 30.0))
                    except asyncio.TimeoutError:
                        break

                    if isinstance(message, bytes):
                        message = message.decode("utf-8")

                    reaction = _parse_reaction(json.loads(message))
                    if reaction is None:
                        continue

                    events.append({
                        "event_type": reaction.event_type,
                        "subject_uri": reaction.subject_uri,
                        "actor_did": reaction.actor_did,
                        "time_us": reaction.time_us,
                        "created_at": reaction.created_at,
                    })

                    if len(events) % 10000 == 0:
                        elapsed = duration_seconds - remaining
                        rate = len(events) / elapsed if elapsed > 0 else 0
                        print(f"  Collected {len(events):,} reactions ({rate:.0f}/s)")

        except Exception as e:
            print(f"  Connection error: {e}, reconnecting in {backoff:.1f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    print(f"Saved {len(events):,} reactions to {output_path}")
    return len(events)


def _parse_reaction(event: dict) -> ReactionEvent | None:
    """Parse a Jetstream event into a ReactionEvent, or None if not a reaction."""
    if event.get("kind") != "commit":
        return None

    commit = event.get("commit")
    if not isinstance(commit, dict):
        return None

    collection = commit.get("collection")
    if collection not in ("app.bsky.feed.like", "app.bsky.feed.repost"):
        return None

    if commit.get("operation") != "create":
        return None

    record = commit.get("record")
    if not isinstance(record, dict):
        return None

    if record.get("$type") not in ("app.bsky.feed.like", "app.bsky.feed.repost"):
        return None

    subject = record.get("subject")
    if not isinstance(subject, dict):
        return None

    subject_uri = subject.get("uri")
    if not isinstance(subject_uri, str):
        return None

    did = event.get("did")
    time_us = event.get("time_us")

    event_type = "like" if collection == "app.bsky.feed.like" else "repost"

    return ReactionEvent(
        event_type=event_type,
        subject_uri=subject_uri,
        actor_did=str(did) if did else "",
        time_us=int(time_us) if isinstance(time_us, int) else 0,
        created_at=record.get("createdAt"),
    )


# ---------------------------------------------------------------------------
# S3 dataset reader
# ---------------------------------------------------------------------------

async def load_reactions_from_s3(
    dataset_name: str,
    batch_pattern: str | None = None,
    max_reactions: int = 2_000_000,
) -> list[dict]:
    """Load reactions from an S3 dataset via FilterForExport.

    Args:
        dataset_name: Name of the S3 dataset (e.g. "reactions").
        batch_pattern: Optional glob pattern to filter batches
            (e.g. "bsky-jetstream-reactions-20260513*").
        max_reactions: Maximum number of reaction events to load.

    Returns:
        List of reaction dicts with keys: subject_uri, event_type, time_us, actor_did.
    """
    reactions: list[dict] = []
    print(f"Loading reactions from S3 dataset '{dataset_name}'...")

    base_columns = ["subject_uri", "event_type", "time_us", "actor_did", "_batch"]

    async with S3DataTool().filter_for_export(
        name=dataset_name,
        base_columns=base_columns,
    ) as view:
        async for row in view:
            data = row.data
            # Filter by batch pattern if specified
            if batch_pattern:
                import fnmatch
                batch = data.get("_batch", "")
                if not fnmatch.fnmatch(batch, batch_pattern):
                    continue

            reactions.append({
                "subject_uri": data["subject_uri"],
                "event_type": data["event_type"],
                "time_us": data["time_us"],
                "actor_did": data.get("actor_did", ""),
            })

            if len(reactions) % 50000 == 0:
                print(f"  Loaded {len(reactions):,} reactions...")

            if len(reactions) >= max_reactions:
                print(f"  Reached max_reactions limit ({max_reactions:,})")
                break

    print(f"  Total: {len(reactions):,} reactions loaded")
    return reactions


async def load_post_texts_from_s3(
    dataset_name: str,
    target_uris: set[str] | None = None,
    max_posts: int = 500_000,
) -> dict[str, str]:
    """Load post texts from an S3 posts dataset, filtered to specific URIs.

    Uses FilterForExport to stream through the posts dataset, matching
    against *target_uris* in memory. For production use on large datasets,
    consider indexing the posts dataset by at_uri.

    Args:
        dataset_name: Name of the S3 dataset (e.g. "posts").
        target_uris: If provided, only load posts whose at_uri is in this set.
        max_posts: Maximum number of posts to load.

    Returns:
        Dict mapping at_uri → text.
    """
    uri_to_text: dict[str, str] = {}
    if target_uris is not None and not target_uris:
        return uri_to_text

    needed = target_uris.copy()
    print(f"Loading post texts from S3 dataset '{dataset_name}'"
          + f" (looking up {len(target_uris):,} URIs)")

    async with S3DataTool().filter_for_export(
        name=dataset_name,
        base_columns=["at_uri", "text"],
    ) as view:
        async for row in view:
            data = row.data
            at_uri = data.get("at_uri")
            text = data.get("text")
            if not at_uri or not text:
                continue
            if at_uri not in needed:
                continue

            uri_to_text[at_uri] = text
            needed.discard(at_uri)

            if len(uri_to_text) % 1000 == 0:
                print(f"  Loaded {len(uri_to_text):,} post texts ({len(needed):,} remaining)...")

            if not needed:
                print(f"  All {len(target_uris):,} URIs matched")
                break
            if len(uri_to_text) >= max_posts:
                print(f"  Reached max_posts limit ({max_posts:,})")
                break

    print(f"  Total: {len(uri_to_text):,} posts loaded"
          f" (matched {len(uri_to_text)} / {len(target_uris):,})")
    return uri_to_text


# ---------------------------------------------------------------------------
# Burst analysis
# ---------------------------------------------------------------------------

@dataclass
class PostBurstMetrics:
    """Per-post burst metrics."""
    subject_uri: str
    num_reactions: int
    num_likes: int
    num_reposts: int
    time_span_s: float
    base_rate: float
    max_burst_weight: int
    total_burst_energy: float
    weighted_burst_count: float
    num_bursts: int
    burst_result: BurstResult | None = field(default=None, repr=False)
    post_text: str | None = None


def analyze_reactions(
    reactions: list[dict],
    post_texts: dict[str, str] | None = None,
    s: float = 2.0,
    gamma: float = 1.0,
    min_reactions: int = 3,
) -> list[PostBurstMetrics]:
    """Group reactions by post and run Kleinberg burst detection.

    Args:
        reactions: List of reaction dicts (subject_uri, event_type, time_us).
        post_texts: Optional lookup from at_uri → text for attaching post content.
        s: Kleinberg scaling parameter.
        gamma: Kleinberg transition cost.
        min_reactions: Minimum reactions a post must have for burst detection.

    Returns:
        List of PostBurstMetrics sorted by total_burst_energy (descending).
    """
    # Group reactions by post
    posts: dict[str, list[dict]] = defaultdict(list)
    like_counts: dict[str, int] = defaultdict(int)
    repost_counts: dict[str, int] = defaultdict(int)

    print(f"\nGrouping {len(reactions):,} reactions by post...")
    for ev in reactions:
        uri = ev["subject_uri"]
        posts[uri].append(ev)
        if ev["event_type"] == "like":
            like_counts[uri] += 1
        elif ev["event_type"] == "repost":
            repost_counts[uri] += 1

    total_posts = len(posts)
    eligible = sum(1 for evts in posts.values() if len(evts) >= min_reactions)
    print(f"  {total_posts:,} unique posts, {eligible:,} with ≥{min_reactions} reactions")

    results: list[PostBurstMetrics] = []

    for uri, events in posts.items():
        n = len(events)
        if n < min_reactions:
            continue

        events.sort(key=lambda e: e["time_us"])

        gaps: list[float] = []
        for i in range(1, n):
            dt_us = events[i]["time_us"] - events[i - 1]["time_us"]
            gaps.append(max(dt_us, 0) / 1_000_000.0)

        time_span_s = (events[-1]["time_us"] - events[0]["time_us"]) / 1_000_000.0

        burst_result = detect_bursts(gaps, s=s, gamma=gamma)
        if burst_result is None:
            continue

        post_text = None
        if post_texts:
            post_text = post_texts.get(uri)

        results.append(PostBurstMetrics(
            subject_uri=uri,
            num_reactions=n,
            num_likes=like_counts.get(uri, 0),
            num_reposts=repost_counts.get(uri, 0),
            time_span_s=time_span_s,
            base_rate=burst_result.base_rate,
            max_burst_weight=burst_result.max_burst_weight,
            total_burst_energy=burst_result.total_burst_energy,
            weighted_burst_count=burst_result.weighted_burst_count,
            num_bursts=len(burst_result.bursts),
            post_text=post_text,
            burst_result=burst_result,
        ))

    results.sort(key=lambda r: r.total_burst_energy, reverse=True)
    return results


def analyze_reactions_from_jsonl(
    reactions_path: str,
    s: float = 2.0,
    gamma: float = 1.0,
    min_reactions: int = 3,
) -> list[PostBurstMetrics]:
    """Load reactions from a local JSONL file and run burst detection."""
    print(f"\nLoading reactions from {reactions_path}...")
    reactions: list[dict] = []
    with open(reactions_path) as f:
        for line in f:
            reactions.append(json.loads(line))
    return analyze_reactions(reactions, s=s, gamma=gamma, min_reactions=min_reactions)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_report(results: list[PostBurstMetrics], top_n: int = 20) -> None:
    """Print a formatted report of burst detection results."""
    if not results:
        print("\nNo posts with sufficient reactions for burst detection.")
        return

    n_bursty = sum(1 for r in results if r.max_burst_weight > 0)
    n_burstless = len(results) - n_bursty

    has_text = any(r.post_text for r in results)

    print(f"\n{'=' * 80}")
    print(f"BURST DETECTION REPORT")
    print(f"{'=' * 80}")
    print(f"Posts analyzed:        {len(results):,}")
    print(f"  With bursts:         {n_bursty:,}")
    print(f"  Without bursts:      {n_burstless:,}")
    print(f"Parameters:            s=2.0 (scaling), gamma=1.0 (transition cost)")
    if has_text:
        print(f"Post texts:            included from S3 posts dataset")
    print()

    # Summary statistics
    max_weights = [r.max_burst_weight for r in results]
    energies = [r.total_burst_energy for r in results]
    w_counts = [r.weighted_burst_count for r in results]
    n_bursts_list = [r.num_bursts for r in results]

    print(f"{'─' * 80}")
    print(f"{'Metric':<30} {'Min':>10} {'Median':>10} {'Max':>10} {'Mean':>10}")
    print(f"{'─' * 80}")
    _print_stat_row("Max burst weight", max_weights)
    _print_stat_row("Total burst energy (s)", energies, fmt=".1f")
    _print_stat_row("Weighted burst count", w_counts, fmt=".1f")
    _print_stat_row("Num bursts (raw)", n_bursts_list)
    print(f"{'─' * 80}")

    # Top posts
    text_col = "Text" if has_text else ""
    print(f"\nTop {min(top_n, len(results))} posts by burst energy:\n")
    hdr = f"{'Rank':<5} {'URI':<42} {'Reacts':>7} {'MaxWt':>6} {'Energy':>10} {'WtCnt':>7} {'#Bursts':>7}"
    if has_text:
        hdr += f"  {'Text (first 80 chars)':<80}"
    print(hdr)
    print(f"{'─' * (95 + (85 if has_text else 0))}")

    for rank, r in enumerate(results[:top_n], 1):
        uri_short = r.subject_uri if len(r.subject_uri) <= 40 else "..." + r.subject_uri[-37:]
        line = (f"{rank:<5} {uri_short:<42} {r.num_reactions:>7} {r.max_burst_weight:>6} "
                f"{r.total_burst_energy:>10.1f} {r.weighted_burst_count:>7.1f} {r.num_bursts:>7}")
        if has_text:
            text_snippet = (r.post_text or "")[:80].replace("\n", " ")
            line += f"  {text_snippet:<80}"
        print(line)

    print(f"{'─' * (95 + (85 if has_text else 0))}")

    # Burst level distribution
    print(f"\nBurst level distribution:")
    level_counts: dict[int, int] = defaultdict(int)
    for r in results:
        level_counts[r.max_burst_weight] += 1
    for level in sorted(level_counts):
        label = "basal" if level == 0 else f"level {level}"
        bar = "█" * min(level_counts[level] // max(1, len(results) // 50), 50)
        print(f"  {label:>8}: {level_counts[level]:>6} {bar}")


def _print_stat_row(label: str, values: list, fmt: str = ".0f") -> None:
    if not values:
        return
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mn = sorted_vals[0]
    mx = sorted_vals[-1]
    med = sorted_vals[n // 2]
    mean = sum(values) / n
    print(f"  {label:<28} {mn:{fmt}}".ljust(42) + f"{med:{fmt}}".ljust(52) +
          f"{mx:{fmt}}".ljust(62) + f"{mean:{fmt}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Burst detection on Bluesky reaction events"
    )
    # Data source (mutually exclusive group in effect)
    parser.add_argument(
        "--duration-minutes", type=float, default=0,
        help="Live mode: collect reactions from Jetstream for N minutes"
    )
    parser.add_argument(
        "--reactions-jsonl", type=str, default=None,
        help="Local JSONL file of reactions (skip S3)"
    )
    parser.add_argument(
        "--reactions-dataset", type=str, default=None,
        help="S3 dataset name for reactions (e.g. 'reactions')"
    )
    parser.add_argument(
        "--posts-dataset", type=str, default=None,
        help="S3 dataset name for posts (e.g. 'posts'). Loads post texts for display."
    )
    parser.add_argument(
        "--batch-pattern", type=str, default=None,
        help="Glob pattern to filter batches (e.g. 'bsky-jetstream-reactions-20260513*')"
    )
    parser.add_argument(
        "--max-reactions", type=int, default=2_000_000,
        help="Maximum reactions to load from S3 (default: 2,000,000)"
    )
    parser.add_argument(
        "--max-posts", type=int, default=1_000_000,
        help="Maximum posts to load from S3 for text lookup (default: 1,000,000)"
    )
    # Kleinberg parameters
    parser.add_argument(
        "--s", type=float, default=2.0,
        help="Kleinberg rate scaling factor (default: 2.0)"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0,
        help="Kleinberg transition cost (default: 1.0)"
    )
    parser.add_argument(
        "--min-reactions", type=int, default=3,
        help="Minimum reactions for a post to be analyzed (default: 3)"
    )
    # Output
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for reactions JSONL in live mode (default: output/reactions_<ts>.jsonl)"
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top posts to show (default: 20)"
    )
    parser.add_argument(
        "--save-metrics", type=str, default=None,
        help="Save per-post metrics as JSONL"
    )
    args = parser.parse_args()

    # ---- Resolve data source ----
    reactions: list[dict] = []
    post_texts: dict[str, str] | None = None

    if args.reactions_jsonl:
        # Local JSONL mode
        if not os.path.exists(args.reactions_jsonl):
            print(f"Error: reactions file not found: {args.reactions_jsonl}")
            sys.exit(1)
        reactions_path = args.reactions_jsonl

    elif args.duration_minutes > 0:
        # Live collection mode
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        reactions_path = args.output or f"output/reactions_{timestamp}.jsonl"
        await collect_reactions(
            duration_seconds=args.duration_minutes * 60,
            output_path=reactions_path,
        )

    elif args.reactions_dataset:
        # S3 mode
        reactions = await load_reactions_from_s3(
            dataset_name=args.reactions_dataset,
            batch_pattern=args.batch_pattern,
            max_reactions=args.max_reactions,
        )
        reactions_path = None  # no local file; data is in memory

        if args.posts_dataset:
            # Only load posts for URIs that appear in the reactions
            target_uris = {r["subject_uri"] for r in reactions}
            post_texts = await load_post_texts_from_s3(
                dataset_name=args.posts_dataset,
                target_uris=target_uris,
                max_posts=args.max_posts,
            )
    else:
        print("Error: specify --duration-minutes, --reactions-jsonl, or --reactions-dataset")
        sys.exit(1)

    # ---- Run analysis ----
    if args.reactions_jsonl or args.duration_minutes > 0:
        results = analyze_reactions_from_jsonl(
            reactions_path,
            s=args.s,
            gamma=args.gamma,
            min_reactions=args.min_reactions,
        )
    else:
        results = analyze_reactions(
            reactions,
            post_texts=post_texts,
            s=args.s,
            gamma=args.gamma,
            min_reactions=args.min_reactions,
        )

    print_report(results, top_n=args.top_n)

    # ---- Save metrics ----
    if args.save_metrics:
        os.makedirs(os.path.dirname(args.save_metrics) or ".", exist_ok=True)
        with open(args.save_metrics, "w") as f:
            for r in results:
                record = {
                    "subject_uri": r.subject_uri,
                    "num_reactions": r.num_reactions,
                    "num_likes": r.num_likes,
                    "num_reposts": r.num_reposts,
                    "time_span_s": r.time_span_s,
                    "base_rate": r.base_rate,
                    "max_burst_weight": r.max_burst_weight,
                    "total_burst_energy": r.total_burst_energy,
                    "weighted_burst_count": r.weighted_burst_count,
                    "num_bursts": r.num_bursts,
                }
                if r.post_text:
                    record["post_text"] = r.post_text
                f.write(json.dumps(record) + "\n")
        print(f"\nSaved per-post metrics to {args.save_metrics}")


if __name__ == "__main__":
    asyncio.run(main())
