"""Evaluate the LLM feasibility judge on test JSONL files.

Reports accuracy and Fleiss' Kappa per file. When multiple files are provided,
also reports combined Fleiss' Kappa across model + all annotators on the
intersection of texts present in all files.

Usage:
    python scripts/analysis/evaluation_llm_judge.py \\
        --test_files $SCRATCH/external_eval_data/annotator_a.jsonl \\
                      $SCRATCH/external_eval_data/annotator_b.jsonl \\
        --model_name gpt-4.1 \\
        --base_url https://api.openai.com/v1 \\
        --max_concurrency 32
"""

import argparse
import asyncio
import json
import os
import sys

import numpy as np
import openai
from tqdm.auto import tqdm

# Add project root to path so scripts.filter.feasibility can be imported
# regardless of whether this is run as a file or a module.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.filter.feasibility import TEMPLATE, _parse_verdict


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' Kappa for multiple raters with multi-class categorical labels.

    Args:
        ratings: (n_subjects, n_raters) array of integer category labels.
                 Missing ratings may be represented as NaN.

    Returns:
        Fleiss' Kappa value. NaN if fewer than 2 subjects have >= 2 raters.
    """
    n_subjects, n_raters = ratings.shape
    n_categories = int(np.nanmax(ratings)) + 1

    counts = np.zeros((n_subjects, n_categories), dtype=np.int64)
    for i in range(n_subjects):
        for j in range(n_raters):
            val = ratings[i, j]
            if not np.isnan(val):
                counts[i, int(val)] += 1

    n_rated = counts.sum(axis=1)
    mask = n_rated >= 2
    counts = counts[mask]
    n_rated = n_rated[mask]
    n_subjects = counts.shape[0]

    if n_subjects < 2:
        return float("nan")

    total_assignments = counts.sum()
    p_j = counts.sum(axis=0) / total_assignments

    denom = n_rated * (n_rated - 1)
    P_i = np.where(denom > 0, (np.sum(counts**2, axis=1) - n_rated) / denom, 0.0)
    P_bar = np.mean(P_i)

    P_e = np.sum(p_j**2)
    if P_e >= 1.0:
        return 1.0

    return (P_bar - P_e) / (1.0 - P_e)


def _load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


async def _judge_one(
    text: str,
    model_name: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> int | None:
    """Get LLM feasibility rating for a single text. Returns None on failure."""
    if not text.strip():
        return None

    prompt = TEMPLATE.format(text=text)

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=16384,
                )
            output = response.choices[0].message.content
            if output is None:
                continue
            _, verdict = _parse_verdict(output)
            return verdict
        except Exception:
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(2**attempt)

    return None


async def evaluate(args):
    file_samples: dict[str, list[dict]] = {}
    for path in args.test_files:
        file_samples[path] = _load_jsonl(path)
        print(f"Loaded {len(file_samples[path])} samples from {path}")

    # Collect unique texts
    all_texts: list[str] = []
    text_to_idx: dict[str, int] = {}
    for samples in file_samples.values():
        for s in samples:
            text = s["text"]
            if text not in text_to_idx:
                text_to_idx[text] = len(all_texts)
                all_texts.append(text)

    print(f"Total unique texts: {len(all_texts)}")

    # Run LLM judge on all unique texts
    client = openai.AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def _judge_with_progress(text: str) -> int | None:
        result = await _judge_one(text, args.model_name, client, semaphore)
        pbar.update(1)
        return result

    pbar = tqdm(total=len(all_texts), desc="LLM judging")
    tasks = [_judge_with_progress(t) for t in all_texts]
    pred_labels = await asyncio.gather(*tasks)
    pbar.close()

    failed = sum(1 for p in pred_labels if p is None)
    if failed:
        print(f"Warning: {failed} texts failed to get a prediction")

    # Build text -> prediction map (default to 0 for failures)
    text_to_pred: dict[str, int] = {}
    for text, idx in text_to_idx.items():
        text_to_pred[text] = pred_labels[idx] if pred_labels[idx] is not None else 0

    # Per-file evaluation
    print("\n--- Per-file results ---")
    all_labels: dict[str, dict[str, int]] = {}
    for path, samples in file_samples.items():
        valid_samples = [(s, text_to_pred[s["text"]]) for s in samples if text_to_pred.get(s["text"]) is not None]
        labels = np.array([s["is_feasible"] for s, _ in valid_samples])
        preds = np.array([p for _, p in valid_samples])

        accuracy = (preds == labels).mean()

        ratings = np.column_stack([labels, preds]).astype(np.float64)
        kappa = fleiss_kappa(ratings)

        print(f"\n{path}")
        print(f"  Samples: {len(valid_samples)} (total {len(samples)})")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Fleiss' Kappa: {kappa:.4f}")

        for s in samples:
            text = s["text"]
            if text not in all_labels:
                all_labels[text] = {}
            all_labels[text][path] = s["is_feasible"]

    # Combined evaluation
    if len(file_samples) > 1:
        print("\n--- Combined (LLM judge + all annotators) ---")
        rater_paths = sorted(file_samples.keys())
        rows = []
        for text in all_texts:
            if text not in text_to_pred:
                continue
            if not all(text in all_labels and p in all_labels[text] for p in rater_paths):
                continue
            row = [text_to_pred[text]]
            for p in rater_paths:
                row.append(all_labels[text][p])
            rows.append(row)

        if rows:
            ratings = np.array(rows, dtype=np.float64)
            kappa = fleiss_kappa(ratings)
            print(f"  Texts in intersection: {len(rows)}")
            print(f"  Raters: LLM + {len(rater_paths)} annotators")
            print(f"  Fleiss' Kappa: {kappa:.4f}")
        else:
            print("  No texts in intersection.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the LLM feasibility judge on test JSONL files"
    )
    parser.add_argument(
        "--test_files", nargs="+", required=True, help="One or more test JSONL files"
    )
    parser.add_argument(
        "--model_name", default="gpt-4.1", help="LLM model name"
    )
    parser.add_argument(
        "--base_url", default="https://api.openai.com/v1", help="OpenAI-compatible API base URL"
    )
    parser.add_argument(
        "--api_key", default="sk-local", help="API key for the LLM service"
    )
    parser.add_argument(
        "--max_concurrency", type=int, default=32, help="Max concurrent API calls"
    )
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
