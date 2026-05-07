"""Evaluate LLM fact-checking accuracy without web search vs. web-search pseudo-labels.

Phase A: Web-search agent runs N times per example → majority vote → pseudo-label.
Phase B: No-search LLM runs N times per example → per-run accuracy vs pseudo-label
         and vs ground truth (baseline datasets only).

Usage:
    # End-to-end test with small sample
    uv run --env-file .oai.env python scripts/analysis/fact_check_eval.py \\
        --n-samples 5 --n-agent-runs 3 --n-judge-runs 3 --max-concurrency 4

    # Full run with defaults
    uv run --env-file .oai.env python scripts/analysis/fact_check_eval.py \\
        --model-name gpt-5.1-nano
"""

import argparse
import asyncio
import json
import logging
import math
import os
import re
from pathlib import Path

import backoff
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import pydantic
from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetGenerationError
from huggingface_hub import hf_hub_download, list_repo_tree
from scipy import stats as scipy_stats

import agents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

with open("templates/fact_check_search.txt") as f:
    SEARCH_TEMPLATE = f.read()

with open("templates/fact_check_no_search.txt") as f:
    NO_SEARCH_TEMPLATE = f.read()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEXT_COLUMN_CANDIDATES = ["claim", "statement", "text", "title"]

DEFAULT_DATASETS = {
    "chai-veracity": {
        "path": "ComplexDataLab/chai-veracity-dry-run-20260506-clustered",
        "split": None,
        "subset": None,
    },
    "fever": {
        "path": "ComplexDataLab/Misinfo_Datasets",
        "split": "test",
        "subset": "fever",
    },
    "liar": {
        "path": "ComplexDataLab/Misinfo_Datasets",
        "split": "test",
        "subset": "liar",
    },
    "liar_new": {
        "path": "ComplexDataLab/Misinfo_Datasets",
        "split": "test",
        "subset": "liar_new",
    },
}


# ---------------------------------------------------------------------------
# Pydantic model for web-search agent structured output
# ---------------------------------------------------------------------------

class FactCheckResult(pydantic.BaseModel):
    """Binary fact-check result. 0 = false, 1 = true."""
    explanation: str
    validity: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_text_column(columns: list[str]) -> str | None:
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def _parse_verdict(output: str) -> tuple[str, int]:
    """Extract explanation and binary rating (0 or 1) from model output.

    Supports:
      - "explanation | 0"
      - "explanation\\n\\nRating: 1"
      - Any text ending with a 0 or 1 after the last newline/pipe.
    """
    # Try pipe-delimited format first
    parts = output.split("|")
    if len(parts) >= 2:
        explanation = "|".join(parts[:-1])
        verdict_str = parts[-1]
        match = re.search(r"\b([01])\b", verdict_str)
        if match is not None:
            return explanation.strip(), int(match.group(1))

    # Try "Rating: X" or "Validity: X" pattern
    match = re.search(r"\b(?:Rating|Validity|Verdict)\s*:\s*([01])\b", output, re.IGNORECASE)
    if match is not None:
        return output.strip(), int(match.group(1))

    # Last resort: find the last 0 or 1 surrounded by word boundaries
    matches = list(re.finditer(r"\b([01])\b", output))
    if matches:
        last = matches[-1]
        explanation = output[:last.start()].strip()
        return explanation, int(last.group(1))

    raise ValueError(
        f"Could not parse fact-check rating from: {output[:200]}"
    )


def _bool_from_veracity(raw: str) -> bool | None:
    """Convert dataset veracity string to boolean. Returns None for 'unknown'."""
    v = raw.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    return None


def _mean_ci(values: list[float]) -> tuple[float, float, float, float, float]:
    """Return (mean, std, sem, ci_low, ci_high) via t-distribution 95% CI."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(values))
    if n >= 2:
        std = float(np.std(values, ddof=1))
        sem = std / math.sqrt(n)
        t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
        ci_low = max(0.0, mean - t_crit * sem)
        ci_high = min(100.0, mean + t_crit * sem)
    else:
        std = float("nan")
        sem = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    return mean, std, sem, ci_low, ci_high


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _iter_parquet_urls(repo_id: str, subset: str | None, split: str) -> list[str]:
    """Find parquet file paths in a HF dataset repo. Returns (filename, local_path) pairs."""
    tree = list_repo_tree(repo_id, recursive=True, repo_type="dataset")
    candidates: list[str] = []
    if subset:
        prefix = f"data/{subset}/"
    else:
        prefix = "data/"

    for item in tree:
        if not hasattr(item, "path"):
            continue
        p = item.path
        if not p.startswith(prefix) or not p.endswith(".parquet"):
            continue
        # Prefer split-specific files, accept any in the subset dir
        candidates.append(p)

    # Sort: files matching the split name first, then everything else
    candidates.sort(key=lambda p: (0 if split in p else 1, p))
    return candidates


def _load_single_dataset(
    name: str,
    cfg: dict,
    n_samples: int,
    seed: int,
) -> tuple[list[str], list[bool | None]]:
    """Load one dataset and return (texts, ground_truth_list)."""
    repo_id = cfg["path"]
    subset = cfg.get("subset")
    split = cfg.get("split")

    # Attempt normal load_dataset; fall back to raw parquet on schema mismatch
    ds = None
    try:
        load_kwargs = {"path": repo_id}
        if subset:
            load_kwargs["name"] = subset
        if split:
            load_kwargs["split"] = split
        ds = load_dataset(**load_kwargs)
        if hasattr(ds, "keys"):
            key = split or list(ds.keys())[0]
            ds = ds[key]
    except DatasetGenerationError:
        logger.info("  Falling back to raw parquet for %s", name)
        filenames = _iter_parquet_urls(repo_id, subset, split or "test")
        if not filenames:
            filenames = _iter_parquet_urls(repo_id, subset, "train")
        if not filenames:
            raise ValueError(f"No parquet files found for {name} in {repo_id}")
        # Download only the top (best-matching) file
        local = hf_hub_download(
            repo_id=repo_id, filename=filenames[0], repo_type="dataset",
        )
        ds = Dataset.from_parquet(local)
        logger.info("  Loaded %d rows from %s", len(ds), filenames[0])

    columns = ds.column_names
    text_col = _get_text_column(columns)
    if text_col is None:
        raise ValueError(
            f"Could not find text column in {name}. Available: {columns}"
        )
    has_veracity = "veracity" in columns
    logger.info("  Text column: '%s' (has_veracity=%s, %d rows)", text_col, has_veracity, len(ds))

    # Subsample
    n_total = len(ds)
    if n_total > n_samples:
        rng = np.random.RandomState(seed)
        indices = set(int(i) for i in rng.choice(n_total, size=n_samples, replace=False))
    else:
        indices = set(range(n_total))

    texts = []
    ground_truth: list[bool | None] = []
    for i in range(n_total):
        if i not in indices:
            continue
        texts.append(str(ds[int(i)][text_col]))
        if has_veracity:
            raw = str(ds[int(i)]["veracity"])
            gt = _bool_from_veracity(raw)
        else:
            gt = None
        ground_truth.append(gt)

    n_gt = sum(1 for g in ground_truth if g is not None)
    n_unknown = sum(1 for g in ground_truth if g is None and has_veracity)
    logger.info(
        "  Subsampled %d (gt: %d, unknown: %d, no-col: %d)",
        len(texts), n_gt, n_unknown, len(texts) - n_gt - n_unknown,
    )
    return texts, ground_truth


def _load_datasets(
    configs: dict, n_samples: int, seed: int
) -> dict[str, dict]:
    """Load and subsample all datasets. Thin async wrapper for progress logging."""
    result: dict[str, dict] = {}
    for name, cfg in configs.items():
        logger.info("Loading dataset: %s", name)
        texts, ground_truth = _load_single_dataset(name, cfg, n_samples, seed)
        result[name] = {"texts": texts, "ground_truth": ground_truth}
    return result


# ---------------------------------------------------------------------------
# Web-search agent (Phase A)
# ---------------------------------------------------------------------------

_search_agent = agents.Agent(
    name="fact_check_search",
    instructions=SEARCH_TEMPLATE,
    tools=[agents.WebSearchTool()],
    output_type=FactCheckResult,
)


@backoff.on_exception(backoff.expo, [openai.APIConnectionError])
async def _run_search_agent(
    text: str,
    run_config: agents.RunConfig,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int] | None:
    """Run web-search agent once. Returns (explanation, validity) or None on failure."""
    if not text or not text.strip():
        return None
    async with semaphore:
        try:
            output = await agents.Runner.run(
                _search_agent, input=text, run_config=run_config,
            )
            result = output.final_output_as(FactCheckResult)
            return (result.explanation, result.validity)
        except Exception:
            return None


async def _run_search_phase(
    texts: list[str],
    n_runs: int,
    model_name: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    raw_dir: Path | None = None,
) -> tuple[list[bool], list[list[dict]]]:
    """Run web-search agent n_runs times per example.

    Returns:
        pseudo_labels: majority-vote boolean label per example
        per_example_runs: list of per-example run details (for raw output)
    """
    agents.set_default_openai_client(client)
    run_config = agents.RunConfig(model=model_name)

    n_examples = len(texts)
    total_calls = n_examples * n_runs
    logger.info(
        "  Phase A (web-search): %d examples x %d runs = %d calls",
        n_examples, n_runs, total_calls,
    )

    all_tasks = []
    for run_idx in range(n_runs):
        for ex_idx, text in enumerate(texts):
            all_tasks.append((
                run_idx,
                ex_idx,
                _run_search_agent(text, run_config, semaphore),
            ))

    results = await asyncio.gather(*[t[2] for t in all_tasks])

    # Group by example
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

    # Majority vote
    pseudo_labels: list[bool] = []
    for votes in per_example:
        valid = [v for v in votes if v is not None]
        if not valid:
            pseudo_labels.append(False)
        else:
            n_false = sum(1 for v in valid if v == 0)
            n_true = sum(1 for v in valid if v == 1)
            pseudo_labels.append(n_true >= n_false)

    n_failed = sum(1 for vlist in per_example for v in vlist if v is None)
    logger.info(
        "  Phase A complete: %d/%d calls failed, pseudo-labels: %d true, %d false",
        n_failed, total_calls,
        sum(pseudo_labels), len(pseudo_labels) - sum(pseudo_labels),
    )

    # Save raw outputs
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for ex_idx, details in enumerate(per_example_details):
            entry = {
                "example_idx": ex_idx,
                "text": texts[ex_idx],
                "pseudo_label": pseudo_labels[ex_idx],
                "runs": details,
            }
            raw_path = raw_dir / f"search_ex{ex_idx:04d}.json"
            with open(raw_path, "w") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)

    return pseudo_labels, per_example_details


# ---------------------------------------------------------------------------
# No-search LLM (Phase B)
# ---------------------------------------------------------------------------

async def _run_no_search_llm(
    text: str,
    model_name: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int] | None:
    """Run no-search LLM once. Returns (explanation, validity) or None on failure."""
    if not text or not text.strip():
        return None

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": NO_SEARCH_TEMPLATE},
                    {"role": "user", "content": f"Statement to fact-check:\n\n{text}"},
                ],
                max_completion_tokens=4096,
            )
            output = response.choices[0].message.content
            if output is None:
                return None
            explanation, validity = _parse_verdict(output)
            return (explanation, validity)
        except Exception:
            return None


async def _run_no_search_phase(
    texts: list[str],
    n_runs: int,
    model_name: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    raw_dir: Path | None = None,
) -> list[list[dict]]:
    """Run no-search LLM n_runs times per example.

    Returns per_example_details for raw output.
    The per-run judgments are grouped by run for downstream accuracy computation.
    """
    n_examples = len(texts)
    total_calls = n_examples * n_runs
    logger.info(
        "  Phase B (no-search): %d examples x %d runs = %d calls",
        n_examples, n_runs, total_calls,
    )

    all_tasks = []
    for run_idx in range(n_runs):
        for ex_idx, text in enumerate(texts):
            all_tasks.append((
                run_idx,
                ex_idx,
                _run_no_search_llm(text, model_name, client, semaphore),
            ))

    results = await asyncio.gather(*[t[2] for t in all_tasks])

    # Group by example
    per_example_details: list[list[dict]] = [[] for _ in range(n_examples)]
    for (run_idx, ex_idx, _), rv in zip(all_tasks, results):
        if rv is not None:
            explanation, validity = rv
            per_example_details[ex_idx].append({
                "run": run_idx,
                "validity": validity,
                "explanation": explanation,
            })
        else:
            per_example_details[ex_idx].append({
                "run": run_idx,
                "validity": None,
                "explanation": None,
            })

    n_failed = sum(1 for dlist in per_example_details for d in dlist if d["validity"] is None)
    logger.info("  Phase B complete: %d/%d calls failed", n_failed, total_calls)

    # Save raw outputs
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for ex_idx, details in enumerate(per_example_details):
            entry = {
                "example_idx": ex_idx,
                "text": texts[ex_idx],
                "runs": details,
            }
            raw_path = raw_dir / f"nosearch_ex{ex_idx:04d}.json"
            with open(raw_path, "w") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)

    return per_example_details


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _evaluate(
    name: str,
    texts: list[str],
    ground_truth: list[bool | None],
    pseudo_labels: list[bool],
    no_search_details: list[list[dict]],
    n_judge_runs: int,
    n_agent_runs: int,
) -> dict:
    """Compute per-run accuracies and aggregate metrics for one dataset.

    Returns a dict with all metrics for downstream reporting.
    """
    n_examples = len(texts)
    has_ground_truth = any(g is not None for g in ground_truth)

    # Build per-run predictions matrix: run_idx -> [(ex_idx, bool_prediction)]
    per_run_preds: list[list[tuple[int, bool]]] = [[] for _ in range(n_judge_runs)]
    for ex_idx, details in enumerate(no_search_details):
        for d in details:
            if d["validity"] is not None:
                run_idx = d["run"]
                per_run_preds[run_idx].append((ex_idx, bool(d["validity"])))

    # Per-run accuracy vs pseudo-label
    run_acc_pseudo: list[float] = []
    # Per-run accuracy vs ground truth
    run_acc_ground: list[float] = []

    for run_idx in range(n_judge_runs):
        preds = dict(per_run_preds[run_idx])
        if not preds:
            continue

        # vs pseudo-label
        correct = sum(
            1 for ex_idx, pred in preds.items()
            if pred == pseudo_labels[ex_idx]
        )
        run_acc_pseudo.append(correct / len(preds) * 100)

        # vs ground truth
        if has_ground_truth:
            gt_indices = [
                ex_idx for ex_idx, pred in preds.items()
                if ground_truth[ex_idx] is not None
            ]
            if gt_indices:
                correct_gt = sum(
                    1 for ex_idx in gt_indices
                    if preds[ex_idx] == ground_truth[ex_idx]
                )
                run_acc_ground.append(correct_gt / len(gt_indices) * 100)

    # t-distribution CI for no-search accuracies
    pseudo_mean, pseudo_std, pseudo_sem, pseudo_ci_low, pseudo_ci_high = _mean_ci(run_acc_pseudo)
    gt_mean, gt_std, gt_sem, gt_ci_low, gt_ci_high = _mean_ci(run_acc_ground) if run_acc_ground else (
        float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
    )

    # Pseudo-label vs ground truth agreement (single value, no CI)
    agreement_pseudo_gt = None
    if has_ground_truth:
        gt_available = [
            (i, g) for i, g in enumerate(ground_truth) if g is not None
        ]
        if gt_available:
            agree_count = sum(
                1 for i, gt in gt_available if pseudo_labels[i] == gt
            )
            agreement_pseudo_gt = agree_count / len(gt_available) * 100

    def _r(v):
        return round(v, 2) if not math.isnan(v) else None

    return {
        "dataset": name,
        "n_examples": n_examples,
        "n_agent_runs": n_agent_runs,
        "n_judge_runs": n_judge_runs,
        "has_ground_truth": has_ground_truth,
        "n_ground_truth_available": sum(1 for g in ground_truth if g is not None),
        "pseudo_label_true_count": int(sum(pseudo_labels)),
        "pseudo_label_false_count": int(len(pseudo_labels) - sum(pseudo_labels)),
        # No-search vs pseudo-label
        "acc_vs_pseudo_mean": _r(pseudo_mean),
        "acc_vs_pseudo_std": _r(pseudo_std),
        "acc_vs_pseudo_sem": _r(pseudo_sem),
        "acc_vs_pseudo_ci_low": _r(pseudo_ci_low),
        "acc_vs_pseudo_ci_high": _r(pseudo_ci_high),
        "run_acc_vs_pseudo": [_r(v) for v in run_acc_pseudo],
        # No-search vs ground truth
        "acc_vs_ground_mean": _r(gt_mean),
        "acc_vs_ground_std": _r(gt_std),
        "acc_vs_ground_sem": _r(gt_sem),
        "acc_vs_ground_ci_low": _r(gt_ci_low),
        "acc_vs_ground_ci_high": _r(gt_ci_high),
        "run_acc_vs_ground": [_r(v) for v in run_acc_ground],
        # Pseudo-label vs ground truth agreement
        "agreement_pseudo_vs_ground": _r(agreement_pseudo_gt) if agreement_pseudo_gt is not None else None,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(results: list[dict], n_judge_runs: int, output_dir: Path) -> None:
    """Bar chart: side-by-side bars for baseline (pseudo + ground accuracy),
    single bar for chai-veracity (pseudo accuracy only)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    dataset_names = [r["dataset"] for r in results]
    n_datasets = len(dataset_names)
    x = np.arange(n_datasets)
    width = 0.3

    # Pseudo-label accuracy bars (all datasets)
    pseudo_means = [r["acc_vs_pseudo_mean"] for r in results]
    pseudo_yerr_low = []
    pseudo_yerr_high = []
    for r in results:
        ci_low = r["acc_vs_pseudo_ci_low"]
        ci_high = r["acc_vs_pseudo_ci_high"]
        mean = r["acc_vs_pseudo_mean"]
        if ci_low is not None and ci_high is not None and mean is not None:
            pseudo_yerr_low.append(mean - ci_low)
            pseudo_yerr_high.append(ci_high - mean)
        else:
            pseudo_yerr_low.append(0)
            pseudo_yerr_high.append(0)

    ax.bar(
        x - width / 2, pseudo_means, width,
        label="vs. Pseudo-Label (web-search agent)",
        color="#2E75B6",
    )
    ax.errorbar(
        x - width / 2, pseudo_means,
        yerr=[pseudo_yerr_low, pseudo_yerr_high],
        fmt="none", ecolor="black", capsize=5, capthick=1.5, linewidth=1.5,
    )

    # Ground truth accuracy bars (baseline datasets only)
    gt_indices = []
    gt_means = []
    gt_yerr_low = []
    gt_yerr_high = []
    for i, r in enumerate(results):
        if r["has_ground_truth"] and r["acc_vs_ground_mean"] is not None:
            gt_indices.append(i)
            gt_means.append(r["acc_vs_ground_mean"])
            ci_low = r["acc_vs_ground_ci_low"]
            ci_high = r["acc_vs_ground_ci_high"]
            mean = r["acc_vs_ground_mean"]
            if ci_low is not None and ci_high is not None and mean is not None:
                gt_yerr_low.append(mean - ci_low)
                gt_yerr_high.append(ci_high - mean)
            else:
                gt_yerr_low.append(0)
                gt_yerr_high.append(0)

    gt_x_positions = [x[i] + width / 2 for i in gt_indices]
    if gt_x_positions:
        ax.bar(
            gt_x_positions, gt_means, width,
            label="vs. Ground Truth (dataset label)",
            color="#A0C8E8",
        )
        ax.errorbar(
            gt_x_positions, gt_means,
            yerr=[gt_yerr_low, gt_yerr_high],
            fmt="none", ecolor="black", capsize=5, capthick=1.5, linewidth=1.5,
        )

    # Annotate bars with values
    for i, r in enumerate(results):
        # Pseudo-label bar annotation
        pseudo_mean = r["acc_vs_pseudo_mean"]
        if pseudo_mean is not None and not math.isnan(pseudo_mean):
            ci_str = ""
            if r["acc_vs_pseudo_ci_low"] is not None and r["acc_vs_pseudo_ci_high"] is not None:
                ci_str = f"\n[{r['acc_vs_pseudo_ci_low']:.1f}, {r['acc_vs_pseudo_ci_high']:.1f}]"
            ax.text(x[i] - width / 2, pseudo_mean + 1.5,
                    f"{pseudo_mean:.1f}%{ci_str}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        # Ground truth bar annotation
        if r["has_ground_truth"] and r["acc_vs_ground_mean"] is not None:
            gt_mean = r["acc_vs_ground_mean"]
            if not math.isnan(gt_mean):
                ci_str = ""
                if r["acc_vs_ground_ci_low"] is not None and r["acc_vs_ground_ci_high"] is not None:
                    ci_str = f"\n[{r['acc_vs_ground_ci_low']:.1f}, {r['acc_vs_ground_ci_high']:.1f}]"
                ax.text(x[i] + width / 2, gt_mean + 1.5,
                        f"{gt_mean:.1f}%{ci_str}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "Fact-Check Accuracy: LLM (no web search) vs. Pseudo-Label & Ground Truth\n"
        f"(95% CI, t-distribution, n={n_judge_runs} runs)",
        fontsize=13,
    )
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    png_path = output_dir / "fact_check_accuracy.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved plot to %s", png_path)

    pdf_path = output_dir / "fact_check_accuracy.pdf"
    fig.savefig(pdf_path)
    logger.info("Saved plot to %s", pdf_path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

def _print_table(results: list[dict]) -> pd.DataFrame:
    """Build and print results DataFrame."""
    rows = []
    for r in results:
        row = {
            "dataset": r["dataset"],
            "n": r["n_examples"],
            "n_gt": r.get("n_ground_truth_available", 0),
            "pseudo_true": r["pseudo_label_true_count"],
            "pseudo_false": r["pseudo_label_false_count"],
            "acc_pseudo_%": r["acc_vs_pseudo_mean"],
            "acc_pseudo_CI": (
                f"[{r['acc_vs_pseudo_ci_low']}, {r['acc_vs_pseudo_ci_high']}]"
                if r["acc_vs_pseudo_ci_low"] is not None else "-"
            ),
        }
        if r["has_ground_truth"]:
            row["acc_ground_%"] = r["acc_vs_ground_mean"]
            row["acc_ground_CI"] = (
                f"[{r['acc_vs_ground_ci_low']}, {r['acc_vs_ground_ci_high']}]"
                if r["acc_vs_ground_ci_low"] is not None else "-"
            )
            row["agreement(pseudo,gt)_%"] = r["agreement_pseudo_vs_ground"]
        else:
            row["acc_ground_%"] = "-"
            row["acc_ground_CI"] = "-"
            row["agreement(pseudo,gt)_%"] = "-"
        rows.append(row)

    df = pd.DataFrame(rows)

    print()
    print("=" * 110)
    print("Fact-Check Evaluation — LLM (no web search) vs. Pseudo-Label & Ground Truth")
    print("=" * 110)
    print(df.to_string(index=False))
    print()

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM fact-checking without web search vs. web-search pseudo-labels",
    )
    parser.add_argument(
        "--model-name", default=None,
        help="OpenAI model name for both agent and judge (default: env MODEL_NAME)",
    )
    parser.add_argument(
        "--agent-model-name", default=None,
        help="Model for web-search agent (must support web_search_preview). Overrides --model-name for Phase A.",
    )
    parser.add_argument(
        "--judge-model-name", default=None,
        help="Model for no-search judge. Overrides --model-name for Phase B.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=25,
        help="Examples to subsample per dataset",
    )
    parser.add_argument(
        "--n-agent-runs", type=int, default=5,
        help="Web-search agent runs per example (for majority-vote pseudo-label)",
    )
    parser.add_argument(
        "--n-judge-runs", type=int, default=5,
        help="No-search LLM runs per example (for t-distribution CI)",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=16,
        help="Max concurrent LLM calls",
    )
    parser.add_argument(
        "--output-dir", default="outputs/fact_check_eval",
        help="Output directory for CSV, JSON, and plots",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for subsampling",
    )
    parser.add_argument(
        "--datasets", default=None,
        help="Path to JSON file with custom dataset configs",
    )
    args = parser.parse_args()

    model_name = args.model_name or os.environ["MODEL_NAME"]
    agent_model_name = args.agent_model_name or model_name
    judge_model_name = args.judge_model_name or model_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Dataset configs
    if args.datasets:
        with open(args.datasets) as f:
            datasets_config = json.load(f)
    else:
        datasets_config = DEFAULT_DATASETS

    # Load datasets
    dataset_data = _load_datasets(datasets_config, args.n_samples, args.seed)

    # LLM client
    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.max_concurrency)

    logger.info(
        "Agent model: %s | Judge model: %s | concurrency: %d | n_samples: %d | n_agent_runs: %d | n_judge_runs: %d",
        agent_model_name, judge_model_name, args.max_concurrency, args.n_samples,
        args.n_agent_runs, args.n_judge_runs,
    )

    all_results: list[dict] = []

    for name, data in dataset_data.items():
        logger.info("=== Dataset: %s ===", name)
        texts = data["texts"]
        ground_truth = data["ground_truth"]

        # Phase A: Web-search agent → pseudo-labels
        search_raw_dir = raw_dir / f"{name}_search"
        pseudo_labels, _ = await _run_search_phase(
            texts=texts,
            n_runs=args.n_agent_runs,
            model_name=agent_model_name,
            client=client,
            semaphore=semaphore,
            raw_dir=search_raw_dir,
        )

        # Phase B: No-search LLM → per-run predictions
        nosearch_raw_dir = raw_dir / f"{name}_nosearch"
        no_search_details = await _run_no_search_phase(
            texts=texts,
            n_runs=args.n_judge_runs,
            model_name=judge_model_name,
            client=client,
            semaphore=semaphore,
            raw_dir=nosearch_raw_dir,
        )

        # Evaluate
        result = _evaluate(
            name=name,
            texts=texts,
            ground_truth=ground_truth,
            pseudo_labels=pseudo_labels,
            no_search_details=no_search_details,
            n_judge_runs=args.n_judge_runs,
            n_agent_runs=args.n_agent_runs,
        )
        all_results.append(result)

        # Log summary
        p_mean = result["acc_vs_pseudo_mean"]
        p_ci = f"[{result['acc_vs_pseudo_ci_low']:.1f}, {result['acc_vs_pseudo_ci_high']:.1f}]" if result["acc_vs_pseudo_ci_low"] is not None else "N/A"
        p_str = f"{p_mean:.1f}%" if p_mean is not None else "N/A"
        logger.info("  Acc vs pseudo: %s (95%% CI: %s)", p_str, p_ci)
        if result["has_ground_truth"]:
            g_mean = result["acc_vs_ground_mean"]
            g_ci = f"[{result['acc_vs_ground_ci_low']:.1f}, {result['acc_vs_ground_ci_high']:.1f}]" if result["acc_vs_ground_ci_low"] is not None else "N/A"
            g_str = f"{g_mean:.1f}%" if g_mean is not None else "N/A"
            agree = result["agreement_pseudo_vs_ground"]
            agree_str = f"{agree:.1f}%" if agree is not None else "N/A"
            logger.info("  Acc vs ground: %s (95%% CI: %s)", g_str, g_ci)
            logger.info("  Pseudo ↔ Ground agreement: %s", agree_str)

    # Print table
    df = _print_table(all_results)

    # Save CSV
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV to %s", csv_path)

    # Save full JSON results
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved JSON to %s", json_path)

    # Save agreement JSON (pseudo-label vs ground truth)
    agreement_data = {}
    for r in all_results:
        if r["has_ground_truth"]:
            agreement_data[r["dataset"]] = {
                "agreement_pseudo_vs_ground_%": r["agreement_pseudo_vs_ground"],
                "n_ground_truth_available": r["n_ground_truth_available"],
            }
    if agreement_data:
        agree_path = output_dir / "agreement.json"
        with open(agree_path, "w") as f:
            json.dump(agreement_data, f, indent=2)
        logger.info("Saved agreement to %s", agree_path)

    # Plot
    _plot(all_results, args.n_judge_runs, output_dir)

    print(f"\nOutput saved to {output_dir}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(main())
