"""Evaluate how web search changes LLM fact-checking verdicts.

Phase A: Web-search agent runs N times per example → per-run verdicts.
Phase B: No-search LLM runs N times per example → per-run verdicts.
Analysis: For each example, compute the fraction of run-pairs where the verdict
          "flipped" (search vs no-search disagree). Report per-dataset flip rate
          with t-distribution 95% CI.

Usage:
    # End-to-end test with small sample
    uv run --env-file .openai.env python scripts/analysis/fact_check_eval.py \\
        --n-samples 5 --n-runs 3 --max-concurrency 4

    # Full run with defaults
    uv run --env-file .openai.env python scripts/analysis/fact_check_eval.py \\
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
    "Chai-Veracity\n(ours)": {
        "path": "ComplexDataLab/chai-veracity-dry-run-20260521-dbscan-diverse",
        "split": None,
        "subset": None,
    },
    "FEVER": {
        "path": "ComplexDataLab/Misinfo_Datasets",
        "split": "test",
        "subset": "fever",
    },
    "LIAR": {
        "path": "ComplexDataLab/Misinfo_Datasets",
        "split": "test",
        "subset": "liar",
    },
    "LIAR-New": {
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
    except (DatasetGenerationError, FileNotFoundError):
        logger.info("  Falling back to raw parquet for %s", name)
        filenames = _iter_parquet_urls(repo_id, subset, split or "test")
        if not filenames:
            filenames = _iter_parquet_urls(repo_id, subset, "train")
        if not filenames:
            raise ValueError(f"No parquet files found for {name} in {repo_id}")
        # Download only the top (best-matching) file
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filenames[0],
            repo_type="dataset",
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
) -> tuple[str, int, int] | None:
    """Run web-search agent once. Returns (explanation, validity, n_search_calls) or None on failure."""
    if not text or not text.strip():
        return None
    async with semaphore:
        try:
            output = await agents.Runner.run(
                _search_agent, input=text, run_config=run_config,
            )
            result = output.final_output_as(FactCheckResult)
            # Count web_search calls from the tool-use tracker snapshot
            tool_snapshot: dict[str, list[str]] = getattr(
                output, "_tool_use_tracker_snapshot", {}
            )
            n_search_calls = sum(
                name == "web_search"
                for names in tool_snapshot.values()
                for name in names
            )
            return (result.explanation, result.validity, n_search_calls)
        except Exception:
            return None


async def _run_search_phase(
    texts: list[str],
    n_runs: int,
    model_name: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    raw_dir: Path | None = None,
) -> tuple[list[bool], list[list[dict]], float, list[float]]:
    """Run web-search agent n_runs times per example.

    Returns:
        pseudo_labels: majority-vote boolean label per example
        per_example_runs: list of per-example run details (for raw output)
        avg_search_calls: average number of web_search tool calls per run
        per_example_avg_search_calls: per-example mean search call counts (for CI)
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
    all_search_counts: list[int] = []
    for (run_idx, ex_idx, _), rv in zip(all_tasks, results):
        if rv is not None:
            explanation, validity, n_search = rv
            per_example[ex_idx].append(validity)
            per_example_details[ex_idx].append(
                {
                    "run": run_idx,
                    "validity": validity,
                    "explanation": explanation,
                    "n_search_calls": n_search,
                }
            )
            all_search_counts.append(n_search)
        else:
            per_example[ex_idx].append(None)
            per_example_details[ex_idx].append(
                {
                    "run": run_idx,
                    "validity": None,
                    "explanation": None,
                    "n_search_calls": None,
                }
            )

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
    avg_search = (
        float(np.mean(all_search_counts)) if all_search_counts else float("nan")
    )

    # Per-example average search calls (for t-distribution CI across examples)
    per_example_avg_search: list[float] = []
    for details in per_example_details:
        counts = [
            d["n_search_calls"] for d in details if d["n_search_calls"] is not None
        ]
        if counts:
            per_example_avg_search.append(float(np.mean(counts)))

    logger.info(
        "  Phase A complete: %d/%d calls failed, pseudo-labels: %d true, %d false, "
        "avg %.1f web_search calls/run",
        n_failed,
        total_calls,
        sum(pseudo_labels),
        len(pseudo_labels) - sum(pseudo_labels),
        avg_search,
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

    return pseudo_labels, per_example_details, avg_search, per_example_avg_search


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
    search_details: list[list[dict]],
    no_search_details: list[list[dict]],
    n_runs: int,
) -> dict:
    """Compute per-example flip rates and aggregate with t-distribution CI.

    For each example, pairs search and no-search runs by run_idx. A "flip"
    occurs when the two verdicts disagree (0 vs 1). The per-example flip rate
    is the fraction of valid run-pairs that flipped. Dataset-level flip rate
    is the mean of per-example flip rates, with 95% t-distribution CI.
    """
    n_examples = len(texts)

    # Build per-example prediction dicts: ex_idx -> {run_idx: bool_prediction}
    def _build_preds(details: list[list[dict]]) -> list[dict[int, bool]]:
        preds: list[dict[int, bool]] = []
        for ex_details in details:
            ex_dict: dict[int, bool] = {}
            for d in ex_details:
                if d["validity"] is not None:
                    ex_dict[d["run"]] = bool(d["validity"])
            preds.append(ex_dict)
        return preds

    search_preds = _build_preds(search_details)
    no_search_preds = _build_preds(no_search_details)

    # Per-example flip rates
    per_example_flip_rates: list[float] = []
    total_pairs = 0
    total_flips = 0
    for ex_idx in range(n_examples):
        s_dict = search_preds[ex_idx]
        n_dict = no_search_preds[ex_idx]
        # Find run indices present in both
        common_runs = set(s_dict.keys()) & set(n_dict.keys())
        if not common_runs:
            continue
        flips = sum(1 for r in common_runs if s_dict[r] != n_dict[r])
        per_example_flip_rates.append(flips / len(common_runs))
        total_pairs += len(common_runs)
        total_flips += flips

    # t-distribution CI across per-example flip rates (in percentage)
    flip_rates_pct = [r * 100 for r in per_example_flip_rates]
    flip_mean, flip_std, flip_sem, flip_ci_low, flip_ci_high = _mean_ci(flip_rates_pct)

    def _r(v):
        return round(v, 2) if not math.isnan(v) else None

    return {
        "dataset": name,
        "n_examples": n_examples,
        "n_runs": n_runs,
        "n_valid_pairs": total_pairs,
        "n_total_flips": total_flips,
        "n_examples_with_pairs": len(per_example_flip_rates),
        "flip_rate_mean": _r(flip_mean),
        "flip_rate_std": _r(flip_std),
        "flip_rate_sem": _r(flip_sem),
        "flip_rate_ci_low": _r(flip_ci_low),
        "flip_rate_ci_high": _r(flip_ci_high),
        "per_example_flip_rates_pct": [_r(v) for v in flip_rates_pct],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(results: list[dict], n_runs: int, output_dir: Path) -> None:
    """Horizontal bar chart: one bar per dataset showing % of cases where search vs no-search verdicts flipped."""
    plt.rcParams.update({"font.size": 18})

    fig, ax = plt.subplots(figsize=(9, 6))

    dataset_names = [r["dataset"] for r in results]
    n_datasets = len(dataset_names)
    y = np.arange(n_datasets)
    height = 0.5

    flip_means = [r["flip_rate_mean"] for r in results]
    xerr_low = []
    xerr_high = []
    for r in results:
        ci_low = r["flip_rate_ci_low"]
        ci_high = r["flip_rate_ci_high"]
        mean = r["flip_rate_mean"]
        if ci_low is not None and ci_high is not None and mean is not None:
            xerr_low.append(mean - ci_low)
            xerr_high.append(ci_high - mean)
        else:
            xerr_low.append(0)
            xerr_high.append(0)

    colors = ["#2E75B6" if i == 0 else "#7CB5EC" for i in range(n_datasets)]
    ax.barh(
        y, flip_means, height, color=colors, label="Verdict flips (search vs no-search)"
    )
    ax.invert_yaxis()

    ax.errorbar(
        flip_means,
        y,
        xerr=[xerr_low, xerr_high],
        fmt="none",
        ecolor="black",
        capsize=8,
        capthick=1.5,
        linewidth=1.5,
    )

    # Annotate bars
    for i, r in enumerate(results):
        mean = r["flip_rate_mean"]
        if mean is not None and not math.isnan(mean):
            ci_str = ""
            if r["flip_rate_ci_low"] is not None and r["flip_rate_ci_high"] is not None:
                ci_str = (
                    f"\n[{r['flip_rate_ci_low']:.1f}, {r['flip_rate_ci_high']:.1f}]"
                )
            ax.text(
                mean + 1.5,
                y[i],
                f"{mean:.1f}%{ci_str}",
                ha="left",
                va="center",
                fontsize=18,
                fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(dataset_names)
    ax.set_xlabel("Flip Rate (%)")
    ax.set_title(
        f"Verdict Flips: Web-Search Agent vs No-Search\n"
        f"(95% CI, t-distribution, {n_runs} rollouts per claim.)"
    )
    ax.set_xlim(0, max(115, max(flip_means) + 20 if flip_means else 100))
    # ax.legend(loc="lower right", fontsize=14, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    png_path = output_dir / "fact_check_flip_rate.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved plot to %s", png_path)

    pdf_path = output_dir / "fact_check_flip_rate.pdf"
    fig.savefig(pdf_path)
    logger.info("Saved plot to %s", pdf_path)

    plt.close(fig)


def _plot_search_calls(results: list[dict], n_runs: int, output_dir: Path) -> None:
    """Horizontal bar chart: one bar per dataset showing avg web-search tool calls per run with 95% t-distribution CI."""
    plt.rcParams.update({"font.size": 18})

    fig, ax = plt.subplots(figsize=(9, 6))

    dataset_names = [r["dataset"] for r in results]
    n_datasets = len(dataset_names)
    y = np.arange(n_datasets)
    height = 0.5

    search_means = [r.get("avg_search_calls", 0) or 0 for r in results]
    xerr_low = []
    xerr_high = []
    for r in results:
        ci_low = r.get("search_calls_ci_low")
        ci_high = r.get("search_calls_ci_high")
        mean = r.get("avg_search_calls")
        if ci_low is not None and ci_high is not None and mean is not None:
            xerr_low.append(mean - ci_low)
            xerr_high.append(ci_high - mean)
        else:
            xerr_low.append(0)
            xerr_high.append(0)

    colors = ["#2E75B6" if i == 0 else "#7CB5EC" for i in range(n_datasets)]
    ax.barh(
        y, search_means, height, color=colors, label="Avg web-search tool calls per run"
    )
    ax.invert_yaxis()

    ax.errorbar(
        search_means,
        y,
        xerr=[xerr_low, xerr_high],
        fmt="none",
        ecolor="black",
        capsize=8,
        capthick=1.5,
        linewidth=1.5,
    )

    # Annotate bars
    for i, r in enumerate(results):
        mean = r.get("avg_search_calls")
        if mean is not None and not math.isnan(mean):
            ci_str = ""
            ci_low = r.get("search_calls_ci_low")
            ci_high = r.get("search_calls_ci_high")
            if ci_low is not None and ci_high is not None:
                ci_str = f"\n[{ci_low:.1f}, {ci_high:.1f}]"
            ax.text(
                mean + 0.05,
                y[i],
                f"{mean:.1f}{ci_str}",
                ha="left",
                va="center",
                fontsize=18,
                fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(dataset_names)
    ax.set_xlabel("Number of Web-Search Tool Calls per claim")
    ax.set_title(
        "Average Web-Search Tool Calls per Claim\n"
        f"(95% CI, t-distribution, {n_runs} rollouts per claim.)"
    )
    xmax = (max(search_means) or 0) + 1.5 if search_means else 5
    ax.set_xlim(0, xmax)
    # ax.legend(loc="lower right", fontsize=14, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    png_path = output_dir / "search_calls_per_dataset.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved plot to %s", png_path)

    pdf_path = output_dir / "search_calls_per_dataset.pdf"
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
            "n_examples": r["n_examples"],
            "n_valid_pairs": r["n_valid_pairs"],
            "n_flips": r["n_total_flips"],
            "flip_rate_%": r["flip_rate_mean"],
            "flip_CI_95%": (
                f"[{r['flip_rate_ci_low']}, {r['flip_rate_ci_high']}]"
                if r["flip_rate_ci_low"] is not None
                else "-"
            ),
            "avg_search_calls": r.get("avg_search_calls", "-"),
            "search_calls_CI_95%": (
                f"[{r.get('search_calls_ci_low')}, {r.get('search_calls_ci_high')}]"
                if r.get("search_calls_ci_low") is not None
                else "-"
            ),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    print()
    print("=" * 90)
    print("Fact-Check Flip Analysis — Verdict Disagreement (Search vs No-Search)")
    print("=" * 90)
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
        "--model-name",
        default=None,
        help="OpenAI model name for both agent and judge (default: env MODEL_NAME)",
    )
    parser.add_argument(
        "--search-agent-model-name",
        default=None,
        help="Model for web-search agent (must support web_search_preview). Overrides --model-name for Phase A.",
    )
    parser.add_argument(
        "--no-search-model-name",
        default=None,
        help="Model for no-search agent. Overrides --model-name for Phase B.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Examples to subsample per dataset",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Runs per example for both search agent and no-search LLM (for flip-rate CI estimation)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help="Max concurrent LLM calls",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/fact_check_eval",
        help="Output directory for CSV, JSON, and plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Path to JSON file with custom dataset configs",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate plots from existing results.json (no LLM calls)",
    )
    args = parser.parse_args()

    if args.regenerate:
        json_path = Path(args.output_dir) / "results.json"
        if not json_path.exists():
            raise FileNotFoundError(f"results.json not found at {json_path}")
        with open(json_path) as f:
            results = json.load(f)
        n_runs = results[0]["n_runs"] if results else args.n_runs
        output_dir = Path(args.output_dir)
        _plot(results, n_runs, output_dir)
        _plot_search_calls(results, n_runs, output_dir)
        print(f"Regenerated plots in {output_dir}/")
        return

    model_name = args.model_name or os.environ["MODEL_NAME"]
    search_agent_model_name = args.search_agent_model_name or model_name
    no_search_model_name = args.no_search_model_name or model_name
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
        "Agent model: %s | Judge model: %s | concurrency: %d | n_samples: %d | n_runs: %d",
        search_agent_model_name,
        no_search_model_name,
        args.max_concurrency,
        args.n_samples,
        args.n_runs,
    )

    all_results: list[dict] = []

    for name, data in dataset_data.items():
        logger.info("=== Dataset: %s ===", name)
        texts = data["texts"]
        ground_truth = data["ground_truth"]

        # Phase A: Web-search agent → per-run verdicts
        search_raw_dir = raw_dir / f"{name}_search"
        (
            _,
            search_details,
            avg_search_calls,
            per_ex_avg_search,
        ) = await _run_search_phase(
            texts=texts,
            n_runs=args.n_runs,
            model_name=search_agent_model_name,
            client=client,
            semaphore=semaphore,
            raw_dir=search_raw_dir,
        )

        # Phase B: No-search LLM → per-run verdicts
        nosearch_raw_dir = raw_dir / f"{name}_nosearch"
        no_search_details = await _run_no_search_phase(
            texts=texts,
            n_runs=args.n_runs,
            model_name=no_search_model_name,
            client=client,
            semaphore=semaphore,
            raw_dir=nosearch_raw_dir,
        )

        # Evaluate
        result = _evaluate(
            name=name,
            texts=texts,
            ground_truth=ground_truth,
            search_details=search_details,
            no_search_details=no_search_details,
            n_runs=args.n_runs,
        )
        result["avg_search_calls"] = (
            round(avg_search_calls, 2) if not math.isnan(avg_search_calls) else None
        )
        search_mean, search_std, search_sem, search_ci_low, search_ci_high = _mean_ci(
            per_ex_avg_search
        )
        result["search_calls_ci_low"] = (
            round(search_ci_low, 2) if not math.isnan(search_ci_low) else None
        )
        result["search_calls_ci_high"] = (
            round(search_ci_high, 2) if not math.isnan(search_ci_high) else None
        )
        all_results.append(result)

        # Log summary
        flip_mean = result["flip_rate_mean"]
        flip_ci = (
            f"[{result['flip_rate_ci_low']:.1f}, {result['flip_rate_ci_high']:.1f}]"
            if result["flip_rate_ci_low"] is not None
            else "N/A"
        )
        flip_str = f"{flip_mean:.1f}%" if flip_mean is not None else "N/A"
        logger.info("  Flip rate: %s (95%% CI: %s)", flip_str, flip_ci)

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

    # Plot
    _plot(all_results, args.n_runs, output_dir)
    _plot_search_calls(all_results, output_dir)

    print(f"\nOutput saved to {output_dir}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(main())
