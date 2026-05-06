"""Evaluate a trained classifier on test JSONL files via vLLM classify API.

Reports accuracy and Fleiss' Kappa per file. When multiple files are provided,
also reports combined Fleiss' Kappa across model + all annotators on the
intersection of texts present in all files.

Usage:
    python scripts/analysis/evaluation_classifier.py \\
        --test_files $SCRATCH/external_eval_data/annotator_a.jsonl \\
                      $SCRATCH/external_eval_data/annotator_b.jsonl \\
        --base_url http://127.0.0.1:8000 \\
        --model_name custom-classifier
"""

import argparse
import asyncio
import json

import httpx
import numpy as np


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


async def _classify_batch(
    client: httpx.AsyncClient,
    base_url: str,
    model_name: str,
    texts: list[str],
) -> list[int]:
    response = await client.post(
        f"{base_url}/classify",
        json={"model": model_name, "input": texts},
    )
    response.raise_for_status()
    # vLLM classify returns string labels like "LABEL_0"; parse to int
    labels = []
    for r in response.json()["data"]:
        label = r["label"]
        if isinstance(label, str) and label.startswith("LABEL_"):
            label = int(label.rsplit("_", 1)[-1])
        labels.append(label)
    return labels


async def _classify_all(
    client: httpx.AsyncClient,
    base_url: str,
    model_name: str,
    texts: list[str],
    batch_size: int,
) -> list[int]:
    tasks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tasks.append(_classify_batch(client, base_url, model_name, batch))
    results = await asyncio.gather(*tasks)
    return [label for batch_result in results for label in batch_result]


async def evaluate(args):
    # Load all test files
    file_samples: dict[str, list[dict]] = {}
    for path in args.test_files:
        file_samples[path] = _load_jsonl(path)
        print(f"Loaded {len(file_samples[path])} samples from {path}")

    # Collect all unique texts (deduplicate for efficiency)
    all_texts: list[str] = []
    text_to_idx: dict[str, int] = {}
    for samples in file_samples.values():
        for s in samples:
            text = s["text"]
            if text not in text_to_idx:
                text_to_idx[text] = len(all_texts)
                all_texts.append(text)

    print(f"Total unique texts: {len(all_texts)}")

    # Get model predictions for all unique texts
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        pred_labels = await _classify_all(
            client, args.base_url, args.model_name, all_texts, args.batch_size
        )

    # Build text -> model_prediction map
    text_to_pred: dict[str, int] = {
        text: pred_labels[idx] for text, idx in text_to_idx.items()
    }

    # Per-file evaluation
    print("\n--- Per-file results ---")
    all_labels: dict[str, dict[str, int]] = {}  # text -> {annotator_path: label}
    for path, samples in file_samples.items():
        labels = np.array([s["is_feasible"] for s in samples])
        preds = np.array([text_to_pred[s["text"]] for s in samples])

        accuracy = (preds == labels).mean()

        ratings = np.column_stack([labels, preds]).astype(np.float64)
        kappa = fleiss_kappa(ratings)

        print(f"\n{path}")
        print(f"  Samples: {len(samples)}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Fleiss' Kappa: {kappa:.4f}")

        for s in samples:
            text = s["text"]
            if text not in all_labels:
                all_labels[text] = {}
            all_labels[text][path] = s["is_feasible"]

    # Combined evaluation: model + all annotators on text intersection
    if len(file_samples) > 1:
        print("\n--- Combined (model + all annotators) ---")
        rater_paths = sorted(file_samples.keys())
        n_raters = 1 + len(rater_paths)  # model + annotators

        rows = []
        for text in all_texts:
            if not all(text in all_labels and p in all_labels[text] for p in rater_paths):
                continue
            row = [text_to_pred[text]]  # model prediction first
            for p in rater_paths:
                row.append(all_labels[text][p])
            rows.append(row)

        if rows:
            ratings = np.array(rows, dtype=np.float64)
            kappa = fleiss_kappa(ratings)
            print(f"  Texts in intersection: {len(rows)}")
            print(f"  Raters: model + {len(rater_paths)} annotators")
            print(f"  Fleiss' Kappa: {kappa:.4f}")
        else:
            print("  No texts in intersection.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained classifier on test JSONL files via vLLM"
    )
    parser.add_argument(
        "--test_files", nargs="+", required=True, help="One or more test JSONL files"
    )
    parser.add_argument(
        "--base_url", default="http://127.0.0.1:8000", help="vLLM server base URL"
    )
    parser.add_argument(
        "--model_name", default="custom-classifier", help="Model name served by vLLM"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Texts per batch request"
    )
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
