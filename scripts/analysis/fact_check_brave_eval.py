"""Evaluate Brave Search as an additional fact-check baseline.

Adds two Brave Search variants to the existing fact-check evaluation:

- **Brave-full** (no staleness): pseudo-label generator, equivalent to OpenAI
  web-search agent. Majority vote of N runs → pseudo-label.

- **Brave-stale** (time-limited freshness=2020-01-01 to two weeks ago): judge
  evaluated against Brave-full pseudo-labels. Like no-search LLM but with
  access to stale web results.

Loads existing no-search LLM raw outputs from a prior run (no re-execution).
Only the Brave search phases issue new LLM calls.

Usage:
    source .brave.env && source .oai.env && uv run python scripts/analysis/fact_check_brave_eval.py \\
        --n-samples 5 --n-agent-runs 3 --n-judge-runs 3 --max-concurrency 4 \\
        --agent-model-name gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

# Add project root to path (same pattern as evaluation_llm_judge.py)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import backoff
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd

import agents
from scripts.analysis.brave_search_tool import (
    create_brave_search_tool,
    get_stale_freshness_range,
)
from scripts.analysis.fact_check_eval import (
    FactCheckResult,
    _bool_from_veracity,
    _get_text_column,
    _load_datasets,
    _mean_ci,
    _parse_verdict,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (reuse existing)
# ---------------------------------------------------------------------------

with open("templates/fact_check_search.txt") as f:
    SEARCH_TEMPLATE = f.read()

with open("templates/fact_check_no_search.txt") as f:
    NO_SEARCH_TEMPLATE = f.read()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_existing_nosearch_raw(
    raw_dir: Path, n_examples: int, n_runs: int
) -> list[list[dict]]:
    """Load existing no-search raw outputs from disk.

    Returns per-example details compatible with the format from
    ``_run_no_search_phase``: list[list[dict]] where each inner list
    contains per-run dicts with keys run, validity, explanation.
    """
    per_example_details: list[list[dict]] = []
    for ex_idx in range(n_examples):
        path = raw_dir / f"nosearch_ex{ex_idx:04d}.json"
        if path.exists():
            with open(path) as fh:
                entry = json.load(fh)
            per_example_details.append(entry["runs"])
        else:
            # Missing file → treat all runs as None
            per_example_details.append([
                {"run": r, "validity": None, "explanation": None}
                for r in range(n_runs)
            ])
    return per_example_details


def _load_existing_search_pseudo_labels(
    raw_dir: Path, n_examples: int
) -> list[bool]:
    """Load pseudo-labels saved from a prior search phase."""
    labels: list[bool] = []
    for ex_idx in range(n_examples):
        path = raw_dir / f"search_ex{ex_idx:04d}.json"
        if path.exists():
            with open(path) as fh:
                entry = json.load(fh)
            labels.append(entry["pseudo_label"])
        else:
            labels.append(False)
    return labels


def _compute_per_run_preds(
    per_example_details: list[list[dict]],
    n_runs: int | None = None,
) -> list[list[tuple[int, bool]]]:
    """Group per-example run details into per-run prediction lists.

    Returns list of lists: run_idx -> [(ex_idx, bool_prediction), ...]
    If n_runs is not provided, it is inferred from the maximum run index in the data.
    """
    # Infer n_runs from the data if not given
    if n_runs is None:
        max_run = 0
        for details in per_example_details:
            for d in details:
                if d["validity"] is not None:
                    max_run = max(max_run, d["run"])
        n_runs = max_run + 1

    per_run_preds: list[list[tuple[int, bool]]] = [[] for _ in range(n_runs)]

    for ex_idx, details in enumerate(per_example_details):
        for d in details:
            if d["validity"] is not None:
                run_idx = d["run"]
                if run_idx >= n_runs:
                    # Extend list if needed (shouldn't happen with correct inference)
                    per_run_preds.extend(
                        [[] for _ in range(run_idx - n_runs + 1)]
                    )
                    n_runs = run_idx + 1
                per_run_preds[run_idx].append((ex_idx, bool(d["validity"])))

    return per_run_preds


def _compute_accuracy(
    per_run_preds: list[list[tuple[int, bool]]],
    labels: list[bool],
    ground_truth: list[bool | None] | None = None,
) -> tuple[list[float], list[float]]:
    """Compute per-run accuracy vs reference labels and optionally vs ground truth.

    Returns (acc_vs_labels, acc_vs_gt) where each is a list of per-run percentages.
    """
    n_runs = len(per_run_preds)
    acc_vs_labels: list[float] = []
    acc_vs_gt: list[float] = []

    for run_idx in range(n_runs):
        preds = dict(per_run_preds[run_idx])
        if not preds:
            continue

        # vs reference labels
        correct = sum(
            1 for ex_idx, pred in preds.items()
            if pred == labels[ex_idx]
        )
        acc_vs_labels.append(correct / len(preds) * 100)

        # vs ground truth
        if ground_truth is not None:
            gt_indices = [
                ex_idx for ex_idx, pred in preds.items()
                if ground_truth[ex_idx] is not None
            ]
            if gt_indices:
                correct_gt = sum(
                    1 for ex_idx in gt_indices
                    if preds[ex_idx] == ground_truth[ex_idx]
                )
                acc_vs_gt.append(correct_gt / len(gt_indices) * 100)

    return acc_vs_labels, acc_vs_gt


def _compute_agreement(
    labels_a: list[bool], labels_b: list[bool],
    mask: list[bool] | None = None,
) -> float | None:
    """Compute percentage agreement between two label lists.

    If mask is provided, only consider indices where mask[i] is True.
    """
    if mask is None:
        mask = [True] * len(labels_a)
    indices = [i for i, m in enumerate(mask) if m]
    if not indices:
        return None
    agree = sum(1 for i in indices if labels_a[i] == labels_b[i])
    return agree / len(indices) * 100


def _round(v):
    return round(v, 2) if not math.isnan(v) else None


# ---------------------------------------------------------------------------
# Brave search agent run
# ---------------------------------------------------------------------------

def _build_brave_agent(tool):
    """Create an Agent with the given Brave search tool."""
    return agents.Agent(
        name="fact_check_brave",
        instructions=SEARCH_TEMPLATE,
        tools=[tool],
        output_type=FactCheckResult,
    )


@backoff.on_exception(backoff.expo, [openai.APIConnectionError])
async def _run_brave_agent(
    text: str,
    agent: agents.Agent,
    run_config: agents.RunConfig,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int] | None:
    """Run a Brave-equipped agent once. Returns (explanation, validity) or None."""
    if not text or not text.strip():
        return None
    async with semaphore:
        try:
            output = await agents.Runner.run(
                agent, input=text, run_config=run_config,
            )
            result = output.final_output_as(FactCheckResult)
            return (result.explanation, result.validity)
        except Exception:
            return None


async def _run_brave_phase(
    texts: list[str],
    n_runs: int,
    agent: agents.Agent,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    raw_dir: Path | None = None,
    label: str = "brave",
) -> tuple[list[bool], list[list[dict]]]:
    """Run Brave-equipped agent n_runs times per example.

    Returns (majority_vote_labels, per_example_details).
    """
    agents.set_default_openai_client(client)
    run_config = agents.RunConfig(model=agent.model)

    n_examples = len(texts)
    total_calls = n_examples * n_runs
    logger.info(
        "  %s: %d examples x %d runs = %d calls",
        label, n_examples, n_runs, total_calls,
    )

    all_tasks = []
    for run_idx in range(n_runs):
        for ex_idx, text in enumerate(texts):
            all_tasks.append((
                run_idx,
                ex_idx,
                _run_brave_agent(text, agent, run_config, semaphore),
            ))

    results = await asyncio.gather(*[t[2] for t in all_tasks])

    per_example: list[list[int | None]] = [[] for _ in range(n_examples)]
    per_example_details: list[list[dict]] = [[] for _ in range(n_examples)]
    for (run_idx, ex_idx, _), rv in zip(all_tasks, results):
        if rv is not None:
            explanation, validity = rv
            per_example[ex_idx].append(validity)
            per_example_details[ex_idx].append({
                "run": run_idx,
                "validity": validity,
                "explanation": explanation,
            })
        else:
            per_example[ex_idx].append(None)
            per_example_details[ex_idx].append({
                "run": run_idx,
                "validity": None,
                "explanation": None,
            })

    # Majority vote → pseudo-label
    labels: list[bool] = []
    for votes in per_example:
        valid = [v for v in votes if v is not None]
        if not valid:
            labels.append(False)
        else:
            n_false = sum(1 for v in valid if v == 0)
            n_true = sum(1 for v in valid if v == 1)
            labels.append(n_true >= n_false)

    n_failed = sum(1 for vlist in per_example for v in vlist if v is None)
    logger.info(
        "  %s complete: %d/%d failed, labels: %d true, %d false",
        label, n_failed, total_calls,
        sum(labels), len(labels) - sum(labels),
    )

    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for ex_idx, details in enumerate(per_example_details):
            entry = {
                "example_idx": ex_idx,
                "text": texts[ex_idx],
                "pseudo_label": labels[ex_idx],
                "runs": details,
            }
            raw_path = raw_dir / f"search_ex{ex_idx:04d}.json"
            with open(raw_path, "w") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)

    return labels, per_example_details


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def _build_result_row(
    dataset: str,
    n_examples: int,
    n_agent_runs: int,
    n_judge_runs: int,
    has_ground_truth: bool,
    n_gt_available: int,
    # Pseudo-labels
    oai_pseudo_labels: list[bool] | None,
    brave_pseudo_labels: list[bool] | None,
    # Judge predictions (per-example details)
    nosearch_details: list[list[dict]] | None,
    bravestale_details: list[list[dict]] | None,
    # Ground truth
    ground_truth: list[bool | None],
) -> dict:
    """Compute all comparison metrics for one dataset and return a result dict."""
    gt_mask = [g is not None for g in ground_truth]

    def _acc_ci(per_run_preds, labels, gt):
        acc_l, acc_g = _compute_accuracy(per_run_preds, labels, gt)
        m_l, s_l, se_l, cl_l, ch_l = _mean_ci(acc_l)
        m_g, s_g, se_g, cl_g, ch_g = _mean_ci(acc_g) if acc_g else (
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
        )
        return {
            "acc_label_mean": _round(m_l), "acc_label_std": _round(s_l),
            "acc_label_ci_low": _round(cl_l), "acc_label_ci_high": _round(ch_l),
            "run_acc_label": [_round(v) for v in acc_l],
            "acc_gt_mean": _round(m_g), "acc_gt_std": _round(s_g),
            "acc_gt_ci_low": _round(cl_g), "acc_gt_ci_high": _round(ch_g),
            "run_acc_gt": [_round(v) for v in acc_g],
        }

    row: dict = {
        "dataset": dataset,
        "n_examples": n_examples,
        "n_agent_runs": n_agent_runs,
        "n_judge_runs": n_judge_runs,
        "has_ground_truth": has_ground_truth,
        "n_ground_truth_available": n_gt_available,
    }

    # --- Pseudo-label counts ---
    if oai_pseudo_labels is not None:
        row["oai_pseudo_true"] = int(sum(oai_pseudo_labels))
        row["oai_pseudo_false"] = int(len(oai_pseudo_labels) - sum(oai_pseudo_labels))
    if brave_pseudo_labels is not None:
        row["brave_pseudo_true"] = int(sum(brave_pseudo_labels))
        row["brave_pseudo_false"] = int(len(brave_pseudo_labels) - sum(brave_pseudo_labels))

    # --- No-search vs OAI pseudo ---
    if nosearch_details is not None and oai_pseudo_labels is not None:
        preds = _compute_per_run_preds(nosearch_details)
        m = _acc_ci(preds, oai_pseudo_labels, ground_truth)
        for k, v in m.items():
            row[f"nosearch_vs_oai_{k}"] = v

    # --- No-search vs Brave pseudo ---
    if nosearch_details is not None and brave_pseudo_labels is not None:
        preds = _compute_per_run_preds(nosearch_details)
        m = _acc_ci(preds, brave_pseudo_labels, ground_truth)
        for k, v in m.items():
            row[f"nosearch_vs_brave_{k}"] = v

    # --- Brave-stale vs Brave pseudo ---
    if bravestale_details is not None and brave_pseudo_labels is not None:
        preds = _compute_per_run_preds(bravestale_details)
        m = _acc_ci(preds, brave_pseudo_labels, ground_truth)
        for k, v in m.items():
            row[f"bravestale_vs_brave_{k}"] = v

    # --- Brave-stale vs OAI pseudo (secondary) ---
    if bravestale_details is not None and oai_pseudo_labels is not None:
        preds = _compute_per_run_preds(bravestale_details)
        m = _acc_ci(preds, oai_pseudo_labels, None)  # no GT for this comparison
        for k, v in m.items():
            row[f"bravestale_vs_oai_{k}"] = v

    # --- Agreements ---
    if oai_pseudo_labels is not None and brave_pseudo_labels is not None:
        row["agreement_oai_brave"] = _round(
            _compute_agreement(oai_pseudo_labels, brave_pseudo_labels)
        )
    if brave_pseudo_labels is not None and has_ground_truth:
        row["agreement_brave_gt"] = _round(
            _compute_agreement(brave_pseudo_labels, ground_truth, mask=gt_mask)
        )
    if oai_pseudo_labels is not None and has_ground_truth:
        row["agreement_oai_gt"] = _round(
            _compute_agreement(oai_pseudo_labels, ground_truth, mask=gt_mask)
        )

    return row


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def _print_table(results: list[dict]) -> pd.DataFrame:
    """Build and print results DataFrame."""
    rows = []
    for r in results:
        row = {
            "dataset": r["dataset"],
            "n": r["n_examples"],
            "n_gt": r.get("n_ground_truth_available", 0),
        }

        def _acc_str(prefix):
            m = r.get(f"{prefix}_acc_label_mean")
            cl = r.get(f"{prefix}_acc_label_ci_low")
            ch = r.get(f"{prefix}_acc_label_ci_high")
            if m is None:
                return "-", "-"
            ci_str = f"[{cl:.1f}, {ch:.1f}]" if cl is not None else "-"
            return f"{m:.1f}%", ci_str

        # Judge columns
        for col_prefix, col_label in [
            ("nosearch_vs_oai", "NS-OAI"),
            ("nosearch_vs_brave", "NS-Brv"),
            ("bravestale_vs_brave", "BS-Brv"),
        ]:
            val, ci = _acc_str(col_prefix)
            row[f"{col_label}_%"] = val
            row[f"{col_label}_CI"] = ci

        # Ground truth accuracies
        for col_prefix, col_label in [
            ("nosearch_vs_oai", "NS-GT"),
            ("bravestale_vs_brave", "BS-GT"),
        ]:
            m = r.get(f"{col_prefix}_acc_gt_mean")
            cl = r.get(f"{col_prefix}_acc_gt_ci_low")
            ch = r.get(f"{col_prefix}_acc_gt_ci_high")
            if m is not None and not math.isnan(m):
                ci_str = f"[{cl:.1f}, {ch:.1f}]" if cl is not None else "-"
                row[f"{col_label}_%"] = f"{m:.1f}%"
                row[f"{col_label}_CI"] = ci_str
            else:
                row[f"{col_label}_%"] = "-"
                row[f"{col_label}_CI"] = "-"

        # Agreements
        row["Agr(OAI,Brv)"] = (
            f"{r['agreement_oai_brave']:.1f}%"
            if r.get("agreement_oai_brave") is not None else "-"
        )
        row["Agr(Brv,GT)"] = (
            f"{r['agreement_brave_gt']:.1f}%"
            if r.get("agreement_brave_gt") is not None else "-"
        )
        row["Agr(OAI,GT)"] = (
            f"{r['agreement_oai_gt']:.1f}%"
            if r.get("agreement_oai_gt") is not None else "-"
        )

        rows.append(row)

    df = pd.DataFrame(rows)
    print()
    print("=" * 130)
    print("Fact-Check Evaluation — Brave Search Baseline")
    print("=" * 130)
    print(df.to_string(index=False))
    print()
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_main(results: list[dict], output_dir: Path, n_runs: int) -> None:
    """Main bar chart with all judge vs pseudo-label comparisons."""
    datasets = [r["dataset"] for r in results]
    x = np.arange(len(datasets))
    n_groups_per_dataset = 3  # NS-OAI, NS-Brv, BS-Brv
    width = 0.22
    group_width = width * n_groups_per_dataset + width * 0.5

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ["#2E75B6", "#D4A017", "#8B4513"]  # blue, gold, brown
    labels = [
        "No-Search vs OAI pseudo",
        "No-Search vs Brave pseudo",
        "Brave-Stale vs Brave pseudo",
    ]
    prefixes = ["nosearch_vs_oai", "nosearch_vs_brave", "bravestale_vs_brave"]

    for gi, (prefix, color, label) in enumerate(zip(prefixes, colors, labels)):
        offset = (gi - (n_groups_per_dataset - 1) / 2) * width
        means = []
        yerr_low = []
        yerr_high = []
        for r in results:
            m = r.get(f"{prefix}_acc_label_mean")
            cl = r.get(f"{prefix}_acc_label_ci_low")
            ch = r.get(f"{prefix}_acc_label_ci_high")
            if m is not None and not math.isnan(m):
                means.append(m)
                yerr_low.append(m - cl if cl is not None else 0)
                yerr_high.append(ch - m if ch is not None else 0)
            else:
                means.append(0)
                yerr_low.append(0)
                yerr_high.append(0)

        bars = ax.bar(x + offset, means, width, label=label, color=color, alpha=0.85)
        ax.errorbar(
            x + offset, means,
            yerr=[yerr_low, yerr_high],
            fmt="none", ecolor="black", capsize=4, capthick=1.2, linewidth=1.2,
        )

        # Annotate
        for i, (m, cl, ch) in enumerate(zip(means, yerr_low, yerr_high)):
            if m > 0:
                ax.text(
                    x[i] + offset, m + 2,
                    f"{m:.1f}%", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold", rotation=90,
                )

    # GT accuracy markers (if available)
    gt_indices_all = []
    gt_offsets_all = []
    gt_values_all = []
    for r_idx, r in enumerate(results):
        for gi, prefix in enumerate(prefixes):
            gt_mean = r.get(f"{prefix}_acc_gt_mean")
            if gt_mean is not None and not math.isnan(gt_mean):
                offset = (gi - (n_groups_per_dataset - 1) / 2) * width
                gt_indices_all.append(r_idx)
                gt_offsets_all.append(x[r_idx] + offset)
                gt_values_all.append(gt_mean)

    if gt_values_all:
        ax.scatter(
            gt_offsets_all, gt_values_all,
            marker="*", color="red", s=80, zorder=5,
            label="vs Ground Truth (marker only, no CI)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Fact-Check Accuracy: Judge Methods vs. Pseudo-Labels\n"
        f"(95% CI, t-distribution, n={n_runs} runs)",
        fontsize=13,
    )
    ax.set_ylim(0, 120)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        path = output_dir / f"fact_check_accuracy.{fmt}"
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    plt.close(fig)


def _plot_secondary(results: list[dict], output_dir: Path, n_runs: int) -> None:
    """Secondary chart: Brave-stale vs OpenAI pseudo-labels (reviewer reference)."""
    datasets = [r["dataset"] for r in results]
    x = np.arange(len(datasets))
    width = 0.45

    fig, ax = plt.subplots(figsize=(10, 6))

    means = []
    yerr_low = []
    yerr_high = []
    for r in results:
        m = r.get("bravestale_vs_oai_acc_label_mean")
        cl = r.get("bravestale_vs_oai_acc_label_ci_low")
        ch = r.get("bravestale_vs_oai_acc_label_ci_high")
        if m is not None and not math.isnan(m):
            means.append(m)
            yerr_low.append(m - cl if cl is not None else 0)
            yerr_high.append(ch - m if ch is not None else 0)
        else:
            means.append(0)
            yerr_low.append(0)
            yerr_high.append(0)

    ax.bar(x, means, width, color="#8B4513", alpha=0.85)
    ax.errorbar(
        x, means,
        yerr=[yerr_low, yerr_high],
        fmt="none", ecolor="black", capsize=6, capthick=1.5, linewidth=1.5,
    )

    for i, (m, cl, ch) in enumerate(zip(means, yerr_low, yerr_high)):
        if m > 0:
            ci_str = f"\n[{m - cl:.1f}, {m + ch:.1f}]"
            ax.text(i, m + 3, f"{m:.1f}%{ci_str}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "Brave-Stale vs. OpenAI Pseudo-Labels (Reviewer Reference)\n"
        f"(95% CI, t-distribution, n={n_runs} runs)",
        fontsize=13,
    )
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        path = output_dir / f"fact_check_accuracy_brave_vs_oai.{fmt}"
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Brave Search baseline for fact-check evaluation",
    )
    parser.add_argument(
        "--agent-model-name", default=None,
        help="Model for Brave search agent calls",
    )
    parser.add_argument(
        "--n-samples", type=int, default=25,
    )
    parser.add_argument(
        "--n-agent-runs", type=int, default=5,
        help="Brave agent runs per example (for majority-vote pseudo-label and brave-stale)",
    )
    parser.add_argument(
        "--n-judge-runs", type=int, default=5,
        help="Runs per example expected in existing no-search raw data",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=16,
    )
    parser.add_argument(
        "--output-dir", default="outputs/fact_check_eval",
    )
    parser.add_argument(
        "--existing-results",
        default="outputs/fact_check_eval/results.json",
        help="Path to existing results.json for merging",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--datasets", default=None,
        help="Path to JSON file with custom dataset configs",
    )
    args = parser.parse_args()

    agent_model_name = args.agent_model_name or os.environ.get("MODEL_NAME", "gpt-5-mini")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"

    # Dataset configs
    if args.datasets:
        with open(args.datasets) as f:
            datasets_config = json.load(f)
    else:
        from scripts.analysis.fact_check_eval import DEFAULT_DATASETS
        datasets_config = DEFAULT_DATASETS

    # Load existing results for OAI pseudo-labels and GT data
    existing_path = Path(args.existing_results)
    if existing_path.exists():
        with open(existing_path) as f:
            existing_results = json.load(f)
        existing_by_name = {r["dataset"]: r for r in existing_results}
        logger.info("Loaded existing results from %s (%d datasets)",
                     existing_path, len(existing_results))
    else:
        existing_by_name = {}
        logger.info("No existing results found at %s", existing_path)

    # Load datasets
    dataset_data = _load_datasets(datasets_config, args.n_samples, args.seed)

    # LLM client
    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.max_concurrency)

    logger.info(
        "Agent model: %s | concurrency: %d | n_samples: %d | n_agent_runs: %d | n_judge_runs: %d",
        agent_model_name, args.max_concurrency, args.n_samples,
        args.n_agent_runs, args.n_judge_runs,
    )

    # Build Brave agents
    brave_full_agent = _build_brave_agent(create_brave_search_tool())
    brave_full_agent.model = agent_model_name

    stale_start, stale_end = get_stale_freshness_range()
    brave_stale_agent = _build_brave_agent(
        create_brave_search_tool(freshness_start=stale_start, freshness_end=stale_end)
    )
    brave_stale_agent.model = agent_model_name
    logger.info("Stale freshness range: %s to %s", stale_start, stale_end)

    all_results: list[dict] = []

    for name, data in dataset_data.items():
        logger.info("=== Dataset: %s ===", name)
        texts = data["texts"]
        ground_truth = data["ground_truth"]
        n_examples = len(texts)

        has_gt = any(g is not None for g in ground_truth)
        n_gt = sum(1 for g in ground_truth if g is not None)

        # --- Load existing OAI pseudo-labels ---
        oai_search_raw = raw_dir / f"{name}_search"
        oai_pseudo_labels: list[bool] | None = None
        if oai_search_raw.is_dir():
            oai_pseudo_labels = _load_existing_search_pseudo_labels(
                oai_search_raw, n_examples
            )
            logger.info("  Loaded %d OAI pseudo-labels from %s", len(oai_pseudo_labels), oai_search_raw)
        else:
            logger.warning("  No OAI search raw data at %s — skipping OAI comparisons", oai_search_raw)

        # --- Load existing no-search raw outputs ---
        nosearch_raw = raw_dir / f"{name}_nosearch"
        nosearch_details: list[list[dict]] | None = None
        if nosearch_raw.is_dir():
            nosearch_details = _load_existing_nosearch_raw(
                nosearch_raw, n_examples, args.n_judge_runs
            )
            n_failed = sum(
                1 for dlist in nosearch_details for d in dlist
                if d["validity"] is None
            )
            logger.info("  Loaded no-search data (%d/%d failed)",
                         n_failed, n_examples * args.n_judge_runs)
        else:
            logger.warning("  No no-search raw data at %s", nosearch_raw)

        # --- Phase A: Brave-full search → pseudo-labels ---
        brave_full_raw = raw_dir / f"{name}_brave_full"
        brave_pseudo_labels, _ = await _run_brave_phase(
            texts=texts,
            n_runs=args.n_agent_runs,
            agent=brave_full_agent,
            client=client,
            semaphore=semaphore,
            raw_dir=brave_full_raw,
            label="Brave-full",
        )

        # --- Phase B: Brave-stale search → per-run predictions ---
        brave_stale_raw = raw_dir / f"{name}_brave_stale"
        _, bravestale_details = await _run_brave_phase(
            texts=texts,
            n_runs=args.n_agent_runs,
            agent=brave_stale_agent,
            client=client,
            semaphore=semaphore,
            raw_dir=brave_stale_raw,
            label="Brave-stale",
        )

        # --- Build result row ---
        result = _build_result_row(
            dataset=name,
            n_examples=n_examples,
            n_agent_runs=args.n_agent_runs,
            n_judge_runs=args.n_judge_runs,
            has_ground_truth=has_gt,
            n_gt_available=n_gt,
            oai_pseudo_labels=oai_pseudo_labels,
            brave_pseudo_labels=brave_pseudo_labels,
            nosearch_details=nosearch_details,
            bravestale_details=bravestale_details,
            ground_truth=ground_truth,
        )
        all_results.append(result)

        # Log summary
        for prefix, judge_name in [
            ("nosearch_vs_oai", "NS-OAI"),
            ("nosearch_vs_brave", "NS-Brv"),
            ("bravestale_vs_brave", "BS-Brv"),
        ]:
            m = result.get(f"{prefix}_acc_label_mean")
            cl = result.get(f"{prefix}_acc_label_ci_low")
            ch = result.get(f"{prefix}_acc_label_ci_high")
            if m is not None:
                ci_str = f"[{cl:.1f}, {ch:.1f}]" if cl is not None else "N/A"
                logger.info("  %s acc: %.1f%% (95%% CI: %s)", judge_name, m, ci_str)

        agree_oai_brave = result.get("agreement_oai_brave")
        if agree_oai_brave is not None:
            logger.info("  OAI-Brave agreement: %.1f%%", agree_oai_brave)
        agree_brave_gt = result.get("agreement_brave_gt")
        if agree_brave_gt is not None:
            logger.info("  Brave-GT agreement: %.1f%%", agree_brave_gt)

    # --- Print table ---
    df = _print_table(all_results)

    # --- Save merged results ---
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV to %s", csv_path)

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved JSON to %s", json_path)

    # --- Agreements ---
    agreement_data = {}
    for r in all_results:
        if r.get("agreement_brave_gt") is not None or r.get("agreement_oai_brave") is not None:
            entry = {}
            if r.get("agreement_oai_gt") is not None:
                entry["agreement_oai_vs_gt_%"] = r["agreement_oai_gt"]
            if r.get("agreement_brave_gt") is not None:
                entry["agreement_brave_vs_gt_%"] = r["agreement_brave_gt"]
            if r.get("agreement_oai_brave") is not None:
                entry["agreement_oai_vs_brave_%"] = r["agreement_oai_brave"]
            entry["n_ground_truth_available"] = r.get("n_ground_truth_available", 0)
            agreement_data[r["dataset"]] = entry
    if agreement_data:
        agree_path = output_dir / "agreement.json"
        with open(agree_path, "w") as f:
            json.dump(agreement_data, f, indent=2)
        logger.info("Saved agreement to %s", agree_path)

    # --- Plots ---
    n_runs_val = max(args.n_agent_runs, args.n_judge_runs)
    _plot_main(all_results, output_dir, n_runs_val)
    _plot_secondary(all_results, output_dir, n_runs_val)

    print(f"\nOutput saved to {output_dir}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(main())
