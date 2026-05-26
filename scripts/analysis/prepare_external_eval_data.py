"""One-time script: convert external_data.jsonl to classifier-compatible JSONL files split by annotator.

Reports Fleiss' Kappa multi-class inter-annotator agreement on claims where both
annotators ("a", "b") have labeled feasibility.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

VALUE_MAP = {
    "Not feasible": 0,
    "Feasible (requires web search)": 1,
    "Feasible (no search needed)": 2,
}


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

    # Proportion of all assignments to each category
    total_assignments = counts.sum()
    p_j = counts.sum(axis=0) / total_assignments

    # Per-subject agreement
    denom = n_rated * (n_rated - 1)
    P_i = np.where(denom > 0, (np.sum(counts**2, axis=1) - n_rated) / denom, 0.0)
    P_bar = np.mean(P_i)

    P_e = np.sum(p_j**2)
    if P_e >= 1.0:
        return 1.0

    return (P_bar - P_e) / (1.0 - P_e)


def main():
    parser = argparse.ArgumentParser(
        description="Convert external feasibility annotations to classifier format"
    )
    parser.add_argument(
        "--input",
        default=os.path.expandvars("$SCRATCH/external_data/data.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.expandvars("$SCRATCH/external_eval_data"),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # claim -> {"text": str, "annotations": {annotator: label}}
    claim_data: dict[str, dict] = defaultdict(
        lambda: {"text": None, "annotations": {}}
    )

    kept, skipped_no_match = 0, 0
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            if row.get("attribute_name") != "feasibility":
                continue

            annotator = row.get("annotator", "").strip()
            if annotator not in ("a", "b"):
                continue

            value = row.get("attribute_value", "")
            label = VALUE_MAP.get(value)
            if label is None:
                skipped_no_match += 1
                continue

            claim = row["claim"]
            if claim_data[claim]["text"] is None:
                claim_data[claim]["text"] = claim
            claim_data[claim]["annotations"][annotator] = label
            kept += 1

    print(f"Kept {kept} rows (skipped {skipped_no_match} with unmatched attribute_value)")

    # Write per-annotator JSONL files
    for annotator in ("a", "b"):
        output_path = os.path.join(args.output_dir, f"annotator_{annotator}.jsonl")
        count = 0
        with open(output_path, "w") as out:
            for data in claim_data.values():
                if annotator in data["annotations"]:
                    obj = {
                        "text": data["text"],
                        "is_feasible": data["annotations"][annotator],
                    }
                    out.write(json.dumps(obj) + "\n")
                    count += 1
        print(f"Annotator {annotator}: {count} rows -> {output_path}")

    # Inter-annotator Fleiss' Kappa on claims where both annotated
    common = [
        (d["annotations"]["a"], d["annotations"]["b"])
        for d in claim_data.values()
        if "a" in d["annotations"] and "b" in d["annotations"]
    ]

    if common:
        ratings = np.array(common, dtype=np.float64)
        kappa = fleiss_kappa(ratings)
        print(f"\nClaims annotated by both: {len(common)}")
        print(f"Fleiss' Kappa (inter-annotator): {kappa:.4f}")
    else:
        print("\nNo claims have annotations from both annotators.")


if __name__ == "__main__":
    main()
