"""Recover original fact_check_eval.py plots from Brave-eval results.json.

The Brave eval script (fact_check_brave_eval.py) overwrote results.json and the
fact_check_accuracy.png/pdf plots. This script reads the current results.json,
maps the OAI pseudo-label metrics back to the original format expected by
fact_check_eval._plot(), and regenerates the original bar chart — no API calls.

Usage:
    uv run --env-file .oai.env python scripts/analysis/recover_fact_check_plots.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.analysis.fact_check_eval import _plot

INPUT_JSON = Path("outputs/fact_check_eval/results.json")
OUTPUT_DIR = Path("outputs/fact_check_eval")


def _nullify_nan(v):
    """Replace NaN with None to keep JSON serializable."""
    import math
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def convert_results(brave_results: list[dict]) -> list[dict]:
    """Convert Brave-eval result rows to original fact_check_eval._evaluate format."""
    converted = []
    for r in brave_results:
        original = {
            "dataset": r["dataset"],
            "n_examples": r["n_examples"],
            "n_agent_runs": r["n_agent_runs"],
            "n_judge_runs": r["n_judge_runs"],
            "has_ground_truth": r["has_ground_truth"],
            "n_ground_truth_available": r.get("n_ground_truth_available", 0),
            "pseudo_label_true_count": r.get("oai_pseudo_true", 0),
            "pseudo_label_false_count": r.get("oai_pseudo_false", 0),
            # No-search vs OAI pseudo-label
            "acc_vs_pseudo_mean": _nullify_nan(r.get("nosearch_vs_oai_acc_label_mean")),
            "acc_vs_pseudo_std": _nullify_nan(r.get("nosearch_vs_oai_acc_label_std")),
            "acc_vs_pseudo_sem": None,
            "acc_vs_pseudo_ci_low": _nullify_nan(r.get("nosearch_vs_oai_acc_label_ci_low")),
            "acc_vs_pseudo_ci_high": _nullify_nan(r.get("nosearch_vs_oai_acc_label_ci_high")),
            "run_acc_vs_pseudo": r.get("nosearch_vs_oai_run_acc_label", []),
            # No-search vs Ground Truth
            "acc_vs_ground_mean": _nullify_nan(r.get("nosearch_vs_oai_acc_gt_mean")),
            "acc_vs_ground_std": _nullify_nan(r.get("nosearch_vs_oai_acc_gt_std")),
            "acc_vs_ground_sem": None,
            "acc_vs_ground_ci_low": _nullify_nan(r.get("nosearch_vs_oai_acc_gt_ci_low")),
            "acc_vs_ground_ci_high": _nullify_nan(r.get("nosearch_vs_oai_acc_gt_ci_high")),
            "run_acc_vs_ground": r.get("nosearch_vs_oai_run_acc_gt", []),
            # OAI pseudo vs ground truth agreement
            "agreement_pseudo_vs_ground": _nullify_nan(r.get("agreement_oai_gt")),
        }
        converted.append(original)
    return converted


def main():
    if not INPUT_JSON.exists():
        print(f"Error: {INPUT_JSON} not found")
        sys.exit(1)

    with open(INPUT_JSON) as f:
        brave_results = json.load(f)

    original_results = convert_results(brave_results)

    # Regenerate the original plot
    n_judge_runs = original_results[0]["n_judge_runs"] if original_results else 5
    _plot(original_results, n_judge_runs, OUTPUT_DIR)

    # Also save a recovery CSV/JSON alongside the plots
    csv_path = OUTPUT_DIR / "results_original.csv"
    import pandas as pd
    rows = []
    for r in original_results:
        row = {
            "dataset": r["dataset"],
            "n": r["n_examples"],
            "n_gt": r["n_ground_truth_available"],
            "pseudo_true": r["pseudo_label_true_count"],
            "pseudo_false": r["pseudo_label_false_count"],
            "acc_pseudo_%": r["acc_vs_pseudo_mean"],
            "acc_pseudo_CI": f"[{r['acc_vs_pseudo_ci_low']}, {r['acc_vs_pseudo_ci_high']}]",
        }
        if r["has_ground_truth"]:
            row["acc_ground_%"] = r["acc_vs_ground_mean"]
            row["acc_ground_CI"] = f"[{r['acc_vs_ground_ci_low']}, {r['acc_vs_ground_ci_high']}]"
            row["agreement(pseudo,gt)_%"] = r["agreement_pseudo_vs_ground"]
        else:
            row["acc_ground_%"] = "-"
            row["acc_ground_CI"] = "-"
            row["agreement(pseudo,gt)_%"] = "-"
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    json_path = OUTPUT_DIR / "results_original.json"
    with open(json_path, "w") as f:
        json.dump(original_results, f, indent=2, default=str)
    print(f"Saved {json_path}")

    print(f"\nOriginal plots regenerated:")
    print(f"  {OUTPUT_DIR / 'fact_check_accuracy.png'}")
    print(f"  {OUTPUT_DIR / 'fact_check_accuracy.pdf'}")

    # Print comparison table
    print()
    print("=" * 110)
    print("Original Fact-Check Evaluation (recovered)")
    print("=" * 110)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
