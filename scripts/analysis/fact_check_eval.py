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
    uv run --env-file .openai.env python scripts/analysis/fact_check_eval.py
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
import tqdm
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


def _wrap_with_progress(coro, pbar):
    """Await *coro* and tick *pbar* on completion, preserving result/exception."""
    async def _inner():
        try:
            return await coro
        finally:
            pbar.update(1)
    return _inner()


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

    pbar = tqdm.tqdm(
        total=total_calls,
        desc="  Phase A (web-search)",
        unit="call",
        ncols=100,
    )
    results = await asyncio.gather(
        *[_wrap_with_progress(t[2], pbar) for t in all_tasks]
    )
    pbar.close()

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

    pbar = tqdm.tqdm(
        total=total_calls,
        desc="  Phase B (no-search)",
        unit="call",
        ncols=100,
    )
    results = await asyncio.gather(
        *[_wrap_with_progress(t[2], pbar) for t in all_tasks]
    )
    pbar.close()

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

    # Per-agent class distribution: true% for search and no-search agents
    def _per_example_true_rates(details: list[list[dict]]) -> list[float]:
        rates: list[float] = []
        for ex_details in details:
            valid = [d for d in ex_details if d["validity"] is not None]
            if valid:
                n_true = sum(1 for d in valid if d["validity"] == 1)
                rates.append(n_true / len(valid) * 100)
        return rates

    search_true_rates = _per_example_true_rates(search_details)
    no_search_true_rates = _per_example_true_rates(no_search_details)

    s_true_mean, _, _, s_true_ci_low, s_true_ci_high = _mean_ci(search_true_rates)
    ns_true_mean, _, _, ns_true_ci_low, ns_true_ci_high = _mean_ci(no_search_true_rates)

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
        "search_true_rate_mean": _r(s_true_mean),
        "search_true_rate_ci_low": _r(s_true_ci_low),
        "search_true_rate_ci_high": _r(s_true_ci_high),
        "no_search_true_rate_mean": _r(ns_true_mean),
        "no_search_true_rate_ci_low": _r(ns_true_ci_low),
        "no_search_true_rate_ci_high": _r(ns_true_ci_high),
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
    ax.set_xlim(0, 100)
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
    """Vertical bar chart: one bar per dataset showing avg web-search tool calls per run with 95% t-distribution CI."""
    plt.rcParams.update({"font.size": 18})

    fig, ax = plt.subplots(figsize=(9, 6))

    dataset_names = [r["dataset"] for r in results]
    n_datasets = len(dataset_names)
    x = np.arange(n_datasets)
    width = 0.5

    search_means = [r.get("avg_search_calls", 0) or 0 for r in results]
    yerr_low = []
    yerr_high = []
    for r in results:
        ci_low = r.get("search_calls_ci_low")
        ci_high = r.get("search_calls_ci_high")
        mean = r.get("avg_search_calls")
        if ci_low is not None and ci_high is not None and mean is not None:
            yerr_low.append(mean - ci_low)
            yerr_high.append(ci_high - mean)
        else:
            yerr_low.append(0)
            yerr_high.append(0)

    colors = ["#2E75B6" if i == 0 else "#7CB5EC" for i in range(n_datasets)]
    ax.bar(
        x, search_means, width, color=colors, label="Avg web-search tool calls per run"
    )

    ax.errorbar(
        x,
        search_means,
        yerr=[yerr_low, yerr_high],
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
                x[i],
                mean + 0.05,
                f"{mean:.1f}{ci_str}",
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylabel("Number of Web-Search Tool Calls per claim")
    ymax = (max(search_means) or 0) + 1.5 if search_means else 5
    ax.set_ylim(0, ymax)
    # ax.legend(loc="lower right", fontsize=14, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    png_path = output_dir / "search_calls_per_dataset.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved plot to %s", png_path)

    pdf_path = output_dir / "search_calls_per_dataset.pdf"
    fig.savefig(pdf_path)
    logger.info("Saved plot to %s", pdf_path)

    plt.close(fig)


def _plot_class_distribution(results: list[dict], n_runs: int, output_dir: Path) -> None:
    """Grouped bar chart: per dataset, two bars showing true% for no-search and web-search agents."""
    plt.rcParams.update({"font.size": 18})

    fig, ax = plt.subplots(figsize=(9, 6))

    dataset_names = [r["dataset"] for r in results]
    n_datasets = len(dataset_names)
    x = np.arange(n_datasets)
    bar_width = 0.3

    dark_grey = "#666666"
    light_grey = "#BBBBBB"
    dark_blue = "#2E75B6"
    light_blue = "#7CB5EC"

    search_true_means = [r.get("search_true_rate_mean") or 0 for r in results]
    no_search_true_means = [r.get("no_search_true_rate_mean") or 0 for r in results]

    # No-search bars (left in each group) — grey
    ax.bar(
        x[0] - bar_width / 2, no_search_true_means[0], bar_width,
        color=dark_grey, label="No-Search",
    )
    if n_datasets > 1:
        ax.bar(
            x[1:] - bar_width / 2, no_search_true_means[1:], bar_width,
            color=light_grey,
        )

    # Web-search bars (right in each group) — blue
    ax.bar(
        x[0] + bar_width / 2, search_true_means[0], bar_width,
        color=dark_blue, label="Web-Search",
    )
    if n_datasets > 1:
        ax.bar(
            x[1:] + bar_width / 2, search_true_means[1:], bar_width,
            color=light_blue,
        )

    # CI error bars
    for i, r in enumerate(results):
        for offset, mean_key, ci_low_key, ci_high_key in [
            (-bar_width / 2, "no_search_true_rate_mean", "no_search_true_rate_ci_low", "no_search_true_rate_ci_high"),
            (+bar_width / 2, "search_true_rate_mean", "search_true_rate_ci_low", "search_true_rate_ci_high"),
        ]:
            mean = r.get(mean_key)
            ci_low = r.get(ci_low_key)
            ci_high = r.get(ci_high_key)
            if mean is not None and ci_low is not None and ci_high is not None:
                ax.errorbar(
                    x[i] + offset, mean,
                    yerr=[[mean - ci_low], [ci_high - mean]],
                    fmt="none", ecolor="black", capsize=6, capthick=1.5, linewidth=1.5,
                )

    # Annotate true% above each bar
    for i, r in enumerate(results):
        ns_mean = r.get("no_search_true_rate_mean")
        s_mean = r.get("search_true_rate_mean")
        ns_off = 1.5
        s_off = 1.5
        if ns_mean is not None and s_mean is not None and abs(ns_mean - s_mean) < 10:
            # Stagger labels so they don't overlap
            if ns_mean >= s_mean:
                ns_off += 8
            else:
                s_off += 8
        if ns_mean is not None:
            ax.text(
                x[i] - bar_width / 2, ns_mean + ns_off, f"{ns_mean:.1f}%",
                ha="center", va="bottom", fontsize=18, fontweight="bold",
            )
        if s_mean is not None:
            ax.text(
                x[i] + bar_width / 2, s_mean + s_off, f"{s_mean:.1f}%",
                ha="center", va="bottom", fontsize=18, fontweight="bold",
            )

    ax.legend(loc="upper right", fontsize=18, framealpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylabel("True Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    png_path = output_dir / "class_distribution.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved plot to %s", png_path)

    pdf_path = output_dir / "class_distribution.pdf"
    fig.savefig(pdf_path)
    logger.info("Saved plot to %s", pdf_path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Flip-category LaTeX table
# ---------------------------------------------------------------------------


def _escape_latex(text: str) -> str:
    """Escape special characters for LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "^": r"\^{}",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "%": r"\%",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text


def _escape_latex_with_links(text: str) -> str:
    """Escape LaTeX special chars, converting markdown [text](url) to \\href."""
    links: list[tuple[str, str]] = []

    def _collect(match: re.Match) -> str:
        links.append((match.group(1), match.group(2)))
        return f"\x00MARKDOWN_LINK_{len(links) - 1}\x00"

    text = re.sub(r"\[([^\]]*)\]\(([^)]*)\)", _collect, text)
    text = _escape_latex(text)
    for i, (link_text, url) in enumerate(links):
        escaped_text = _escape_latex(link_text)
        placeholder = _escape_latex(f"\x00MARKDOWN_LINK_{i}\x00")
        text = text.replace(placeholder, f"\\href{{{url}}}{{{escaped_text}}}")
    return text


def _write_latex_flip_table(results: list[dict], output_dir: Path) -> None:
    """Write a LaTeX table with flip-category percentages and 95% CIs.

    Categories: no-search verdict -> web-search verdict
      F->F, F->T, T->F, T->T
    Each cell shows: mean [CI_low, CI_high].
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Dataset & F $\to$ F & F $\to$ T & T $\to$ F & T $\to$ T \\",
        r"\midrule",
    ]

    for r in results:
        ds = r["dataset"].replace("_", r"\_")
        cells = []
        for cat in ["ff", "ft", "tf", "tt"]:
            mean = r.get(f"{cat}_mean")
            ci_low = r.get(f"{cat}_ci_low")
            ci_high = r.get(f"{cat}_ci_high")
            if mean is not None and ci_low is not None and ci_high is not None:
                cells.append(f"{mean:.1f} [{ci_low:.1f}, {ci_high:.1f}]")
            else:
                cells.append("--")
        lines.append(
            f"  {ds} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\"
        )

    n_runs = results[0]["n_runs"] if results else "?"
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-example mean flip-category percentages with 95\% confidence"
        r" intervals (t-distribution, $n=" + str(n_runs) + r"$ runs per example)."
        r" Categories show the no-search $\to$ web-search verdict transition.}",
        r"\label{tab:flip-categories}",
        r"\end{table}",
    ])

    tex_path = output_dir / "flip_categories_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved LaTeX table to %s", tex_path)


def _write_appendix_flip_table(
    flip_examples: dict[str, dict[str, list[dict]]],
    results: list[dict],
    output_dir: Path,
) -> None:
    """Write a LaTeX longtable with example claims and raw outputs per flip category.

    For each verdict-transition category (F->T, T->F, F->F, T->T) and each
    dataset, up to 5 claims are shown with their no-search and web-search
    explanations from a representative run.
    """
    # Build dataset -> result lookup keyed by dataset display name
    stats_by_dataset: dict[str, dict] = {r["dataset"]: r for r in results}

    flip_names = {
        "ft": r"False $\to$ True (flip to true with search)",
        "tf": r"True $\to$ False (flip to false with search)",
        "ff": r"False $\to$ False (stable false)",
        "tt": r"True $\to$ True (stable true)",
    }

    lines = [
        r"\begin{longtable}{p{\textwidth}}",
        r"\caption{Example claims with no-search and web-search fact-checking outputs for each verdict-transition category and dataset. Claims are shown in regular typeface; raw model outputs in typewriter font. Both the No-search and Web-search experiment are implemented using gpt-5.4-mini and the OpenAI Python Agent SDK. Web search uses the first-party web search tool from OpenAI. Refer to Appendix~\ref{app:fact-check-prompts} for instructions for the agents. The web search agent cites its sources in the Markdown hyperlink format. These URLs have been replaced with hyper-ref links in this table. Transition probabilities are calculated based on a sample of $50$ claims from each dataset. Student's-t confidence intervals for the transition probabilities are based on $5$ independent rollouts for each claim.}",
        r"\label{tab:appendix-flip-examples} \\",
        r"\toprule",
        r"\endfirsthead",
        r"\multicolumn{1}{c}{\textit{continued from previous page}} \\",
        r"\toprule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]

    flip_short = {
        "ft": r"F$\to$T",
        "tf": r"T$\to$F",
        "ff": r"F$\to$F",
        "tt": r"T$\to$T",
    }

    for cat in ["ft", "tf", "ff", "tt"]:
        lines.append(r"\midrule")
        lines.append(
            r"\multicolumn{1}{c}{\textbf{" + flip_names[cat] + r"}} \\"
        )
        lines.append(r"\midrule")

        for ds_name, categories in flip_examples.items():
            examples = categories.get(cat, [])
            if not examples:
                continue

            stats = stats_by_dataset.get(ds_name, {})
            cat_mean = stats.get(f"{cat}_mean")
            cat_ci_low = stats.get(f"{cat}_ci_low")
            cat_ci_high = stats.get(f"{cat}_ci_high")
            if cat_mean is not None and cat_ci_low is not None and cat_ci_high is not None:
                stat_str = (
                    f"{flip_short[cat]}  [{cat_mean:.1f}\\%, 95\\% CI: "
                    f"{cat_ci_low:.1f}--{cat_ci_high:.1f}]"
                )
            else:
                stat_str = flip_short[cat]

            lines.append(
                r"\textbf{" + _escape_latex(ds_name) + r"} "
                + r"{\normalfont " + stat_str + r"} \\"
            )
            lines.append(r"\midrule")

            for ex in examples:
                claim = _escape_latex(ex["text"].strip())
                ns_expl = " \\newline ".join(
                    _escape_latex_with_links(seg)
                    for seg in ex["no_search_explanation"].strip().split("\n")
                    if seg.strip()
                )
                s_expl = " \\newline ".join(
                    _escape_latex_with_links(seg)
                    for seg in ex["search_explanation"].strip().split("\n")
                    if seg.strip()
                )

                ns_label = "True" if ex["no_search_validity"] == 1 else "False"
                s_label = "True" if ex["search_validity"] == 1 else "False"

                lines.append(
                    r"\textit{Claim:} " + claim + r" \\"
                )
                lines.append(
                    r"\quad \textbf{No-search} (verdict: " + ns_label
                    + r"): {\ttfamily\small " + ns_expl + r"} \\"
                )
                lines.append(
                    r"\quad \textbf{Web-search} (verdict: " + s_label
                    + r"): {\ttfamily\small " + s_expl + r"} \\"
                )
                lines.append(r"\hline\addlinespace")

    lines.extend([
        r"\bottomrule",
        r"\end{longtable}",
    ])

    tex_path = output_dir / "appendix_flip_examples.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved appendix LaTeX table to %s", tex_path)


# ---------------------------------------------------------------------------
# Flip-category plot
# ---------------------------------------------------------------------------


def _plot_flip_categories(results: list[dict], output_dir: Path) -> None:
    """Stacked horizontal bar chart: per dataset, bars show F->T and T->F
    as the 'flip' components, with F->F and T->T as stable components.
    """
    plt.rcParams.update({"font.size": 18})

    fig, ax = plt.subplots(figsize=(9, 6))

    dataset_names = [r["dataset"] for r in results]
    n_datasets = len(dataset_names)
    x = np.arange(n_datasets)
    width = 0.55

    ff_vals = [r["ff_mean"] or 0 for r in results]
    ft_vals = [r["ft_mean"] or 0 for r in results]
    tf_vals = [r["tf_mean"] or 0 for r in results]
    tt_vals = [r["tt_mean"] or 0 for r in results]

    # Stack order (bottom to top): F->F, F->T, T->F, T->T
    bottom_ft = ff_vals
    bottom_tf = [a + b for a, b in zip(ff_vals, ft_vals)]
    bottom_tt = [a + b + c for a, b, c in zip(ff_vals, ft_vals, tf_vals)]

    colors = ["#999999", "#D55E00", "#009E73", "#2E75B6"]
    labels = [r"F $\to$ F", r"F $\to$ T", r"T $\to$ F", r"T $\to$ T"]

    ax.bar(x, ff_vals, width, label=labels[0], color=colors[0])
    ax.bar(x, ft_vals, width, bottom=bottom_ft, label=labels[1], color=colors[1])
    ax.bar(x, tf_vals, width, bottom=bottom_tf, label=labels[2], color=colors[2])
    ax.bar(x, tt_vals, width, bottom=bottom_tt, label=labels[3], color=colors[3])

    # Annotate segments
    for i in range(n_datasets):
        for vals, bottom, color in [
            (ff_vals, [0] * n_datasets, "white"),
            (ft_vals, bottom_ft, "white"),
            (tf_vals, bottom_tf, "white"),
            (tt_vals, bottom_tt, "white"),
        ]:
            if vals[i] > 6:
                ax.text(
                    x[i], bottom[i] + vals[i] / 2, f"{vals[i]:.1f}%",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color=color,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylabel("Percentage of run-pairs (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=14, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    png_path = output_dir / "flip_categories.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved plot to %s", png_path)

    pdf_path = output_dir / "flip_categories.pdf"
    fig.savefig(pdf_path)
    logger.info("Saved plot to %s", pdf_path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------


def _print_flip_table(results: list[dict]) -> pd.DataFrame:
    """Build and print flip-category results DataFrame."""
    rows = []
    for r in results:
        row = {
            "dataset": r["dataset"],
            "n": r["n_examples"],
            "F->F_%": r["ff_mean"],
            "F->F_CI": f"[{r['ff_ci_low']}, {r['ff_ci_high']}]" if r["ff_ci_low"] is not None else "-",
            "F->T_%": r["ft_mean"],
            "F->T_CI": f"[{r['ft_ci_low']}, {r['ft_ci_high']}]" if r["ft_ci_low"] is not None else "-",
            "T->F_%": r["tf_mean"],
            "T->F_CI": f"[{r['tf_ci_low']}, {r['tf_ci_high']}]" if r["tf_ci_low"] is not None else "-",
            "T->T_%": r["tt_mean"],
            "T->T_CI": f"[{r['tt_ci_low']}, {r['tt_ci_high']}]" if r["tt_ci_low"] is not None else "-",
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    print()
    print("=" * 110)
    print("Fact-Check Flip Categories — Verdict Transitions (No-Search -> Web-Search)")
    print("=" * 110)
    print(df.to_string(index=False))
    print()

    return df


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
# Regenerate helper: compute class distribution from raw output files
# ---------------------------------------------------------------------------


def _dataset_to_raw_prefix(dataset_name: str, existing_prefixes: set[str]) -> str | None:
    """Map a dataset display name to a raw-directory prefix."""
    # Exact match first (raw dirs are named with the dataset key verbatim)
    if dataset_name in existing_prefixes:
        return dataset_name

    # Fall back to sanitized matching for older or differently-named dirs
    def _sanitize(name: str) -> str:
        s = name.lower()
        s = s.replace("\n", " ").replace("(", "").replace(")", "")
        s = re.sub(r"\s+", "-", s).strip("-")
        return s

    sanitized = _sanitize(dataset_name)
    sanitized_map = {_sanitize(p): p for p in existing_prefixes}

    if sanitized in sanitized_map:
        return sanitized_map[sanitized]

    sanitized_u = sanitized.replace("-", "_")
    if sanitized_u in sanitized_map:
        return sanitized_map[sanitized_u]

    # Partial match (against sanitized forms)
    for s_prefix, original in sanitized_map.items():
        if s_prefix in sanitized or sanitized in s_prefix:
            return original

    return None


def _compute_flip_categories_from_raw(raw_dir: Path) -> list[dict]:
    """Read raw per-run JSON files and compute flip-category rates.

    For each dataset (matched *_search / *_nosearch dir pair), loads all
    example files, pairs search and no-search runs by run_idx, and categorizes
    each pair by the no-search -> web-search verdict transition:
    F→F, F→T, T→F, T→T.

    Returns list of per-dataset result dicts with per-example mean and
    t-distribution 95% CI for each category.
    """
    datasets: dict[str, dict[str, Path]] = {}
    for d in sorted(raw_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name.endswith("_search"):
            prefix = name.removesuffix("_search")
            datasets.setdefault(prefix, {})["search"] = d
        elif name.endswith("_nosearch"):
            prefix = name.removesuffix("_nosearch")
            datasets.setdefault(prefix, {})["nosearch"] = d

    # Order: Chai-Veracity first, then alphabetical
    def _sort_key(prefix: str) -> tuple[int, str]:
        p = prefix.replace("\n", " ").lower()
        if "chai" in p:
            return (0, p)
        return (1, p)

    results: list[dict] = []
    n_runs = 0

    for prefix in sorted(datasets.keys(), key=_sort_key):
        dirs = datasets[prefix]
        if "search" not in dirs or "nosearch" not in dirs:
            logger.warning("  Skipping %s: missing search or nosearch dir", prefix)
            continue

        search_dir = dirs["search"]
        nosearch_dir = dirs["nosearch"]

        # Load search data keyed by example_idx
        search_data: dict[int, dict] = {}
        for fpath in sorted(search_dir.glob("search_ex*.json")):
            with open(fpath) as f:
                data = json.load(f)
            search_data[data["example_idx"]] = data

        # Load nosearch data keyed by example_idx
        nosearch_data: dict[int, dict] = {}
        for fpath in sorted(nosearch_dir.glob("nosearch_ex*.json")):
            with open(fpath) as f:
                data = json.load(f)
            nosearch_data[data["example_idx"]] = data

        # Per-example category rates
        per_ex_ff: list[float] = []
        per_ex_ft: list[float] = []
        per_ex_tf: list[float] = []
        per_ex_tt: list[float] = []

        common_indices = sorted(set(search_data.keys()) & set(nosearch_data.keys()))
        for ex_idx in common_indices:
            s_runs = {
                r["run"]: r["validity"]
                for r in search_data[ex_idx]["runs"]
                if r["validity"] is not None
            }
            n_runs_dict = {
                r["run"]: r["validity"]
                for r in nosearch_data[ex_idx]["runs"]
                if r["validity"] is not None
            }

            common_runs = set(s_runs.keys()) & set(n_runs_dict.keys())
            if not common_runs:
                continue

            ff = ft = tf = tt = 0
            for run in common_runs:
                ns = n_runs_dict[run]  # 0=False, 1=True
                s = s_runs[run]       # 0=False, 1=True
                if ns == 0 and s == 0:
                    ff += 1
                elif ns == 0 and s == 1:
                    ft += 1
                elif ns == 1 and s == 0:
                    tf += 1
                else:
                    tt += 1

            total = ff + ft + tf + tt
            per_ex_ff.append(ff / total * 100)
            per_ex_ft.append(ft / total * 100)
            per_ex_tf.append(tf / total * 100)
            per_ex_tt.append(tt / total * 100)

        n_runs = max(n_runs, max(len(s_runs), len(n_runs_dict)) if common_indices else 0)

        def _stats(vals):
            return _mean_ci(vals)

        ff_m, _, _, ff_l, ff_h = _stats(per_ex_ff)
        ft_m, _, _, ft_l, ft_h = _stats(per_ex_ft)
        tf_m, _, _, tf_l, tf_h = _stats(per_ex_tf)
        tt_m, _, _, tt_l, tt_h = _stats(per_ex_tt)

        def _r(v):
            return round(v, 2) if not math.isnan(v) else None

        # Clean dataset name for display
        display = prefix.replace("\n", " ").strip()

        if len(per_ex_ff) == 0:
            logger.warning("  Skipping %s: no examples with valid run-pairs", display)
            continue

        results.append({
            "dataset": display,
            "n_examples": len(per_ex_ff),
            "n_runs": n_runs,
            "ff_mean": _r(ff_m),  "ff_ci_low": _r(ff_l),  "ff_ci_high": _r(ff_h),
            "ft_mean": _r(ft_m),  "ft_ci_low": _r(ft_l),  "ft_ci_high": _r(ft_h),
            "tf_mean": _r(tf_m),  "tf_ci_low": _r(tf_l),  "tf_ci_high": _r(tf_h),
            "tt_mean": _r(tt_m),  "tt_ci_low": _r(tt_l),  "tt_ci_high": _r(tt_h),
        })

        logger.info(
            "  %s: F→F %.1f%%, F→T %.1f%%, T→F %.1f%%, T→T %.1f%%",
            display, ff_m, ft_m, tf_m, tt_m,
        )

    return results


def _collect_flip_examples(raw_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """Collect example claims and explanations for each flip category.

    Returns dict mapping dataset_name -> {category -> [examples]}
    where category is one of 'ff', 'ft', 'tf', 'tt' and each example is:
        {'text': str, 'run': int, 'no_search_validity': int,
         'no_search_explanation': str, 'search_validity': int,
         'search_explanation': str}
    """
    MAX_PER_CATEGORY = 5

    datasets: dict[str, dict[str, Path]] = {}
    for d in sorted(raw_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name.endswith("_search"):
            prefix = name.removesuffix("_search")
            datasets.setdefault(prefix, {})["search"] = d
        elif name.endswith("_nosearch"):
            prefix = name.removesuffix("_nosearch")
            datasets.setdefault(prefix, {})["nosearch"] = d

    def _sort_key(prefix: str) -> tuple[int, str]:
        p = prefix.replace("\n", " ").lower()
        if "chai" in p:
            return (0, p)
        return (1, p)

    all_examples: dict[str, dict[str, list[dict]]] = {}

    for prefix in sorted(datasets.keys(), key=_sort_key):
        dirs = datasets[prefix]
        if "search" not in dirs or "nosearch" not in dirs:
            continue

        search_dir = dirs["search"]
        nosearch_dir = dirs["nosearch"]

        search_data: dict[int, dict] = {}
        for fpath in sorted(search_dir.glob("search_ex*.json")):
            with open(fpath) as f:
                data = json.load(f)
            search_data[data["example_idx"]] = data

        nosearch_data: dict[int, dict] = {}
        for fpath in sorted(nosearch_dir.glob("nosearch_ex*.json")):
            with open(fpath) as f:
                data = json.load(f)
            nosearch_data[data["example_idx"]] = data

        display = prefix.replace("\n", " ").strip()
        all_examples[display] = {"ff": [], "ft": [], "tf": [], "tt": []}

        common_indices = sorted(set(search_data.keys()) & set(nosearch_data.keys()))
        seen: dict[str, set[int]] = {"ff": set(), "ft": set(), "tf": set(), "tt": set()}

        for ex_idx in common_indices:
            s_runs = {
                r["run"]: (r["validity"], r.get("explanation", ""))
                for r in search_data[ex_idx]["runs"]
                if r["validity"] is not None
            }
            n_runs = {
                r["run"]: (r["validity"], r.get("explanation", ""))
                for r in nosearch_data[ex_idx]["runs"]
                if r["validity"] is not None
            }

            common_runs = set(s_runs.keys()) & set(n_runs.keys())
            if not common_runs:
                continue

            text = search_data[ex_idx].get("text", "")

            for run in sorted(common_runs):
                ns_v, ns_expl = n_runs[run]
                s_v, s_expl = s_runs[run]

                if ns_v == 0 and s_v == 0:
                    cat = "ff"
                elif ns_v == 0 and s_v == 1:
                    cat = "ft"
                elif ns_v == 1 and s_v == 0:
                    cat = "tf"
                else:
                    cat = "tt"

                if ex_idx in seen[cat]:
                    continue

                cat_list = all_examples[display][cat]
                if len(cat_list) < MAX_PER_CATEGORY:
                    seen[cat].add(ex_idx)
                    cat_list.append({
                        "text": text,
                        "run": run,
                        "no_search_validity": ns_v,
                        "no_search_explanation": ns_expl,
                        "search_validity": s_v,
                        "search_explanation": s_expl,
                    })

        logger.info(
            "  %s: collected ff=%d ft=%d tf=%d tt=%d",
            display,
            len(all_examples[display]["ff"]),
            len(all_examples[display]["ft"]),
            len(all_examples[display]["tf"]),
            len(all_examples[display]["tt"]),
        )

    return all_examples


def _compute_class_dist_from_raw(results: list[dict], raw_dir: Path) -> None:
    """Populate class-distribution fields in *results* from raw per-run JSON files."""
    # Discover available raw-directory prefixes
    existing: dict[str, dict[str, Path]] = {}
    for d in raw_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.endswith("_search"):
            existing.setdefault(name.removesuffix("_search"), {})["search"] = d
        elif name.endswith("_nosearch"):
            existing.setdefault(name.removesuffix("_nosearch"), {})["nosearch"] = d

    existing_prefixes = set(existing.keys())

    for r in results:
        prefix = _dataset_to_raw_prefix(r["dataset"], existing_prefixes)
        if prefix is None or prefix not in existing:
            logger.warning("  No raw dir match for dataset '%s'", r["dataset"])
            continue

        dirs = existing[prefix]
        for agent_type, dir_key in [("search", "search"), ("no_search", "nosearch")]:
            agent_dir = dirs.get(dir_key)
            if agent_dir is None:
                continue
            true_rates: list[float] = []
            for fpath in sorted(agent_dir.glob("*ex*.json")):
                with open(fpath) as fh:
                    data = json.load(fh)
                valid = [d for d in data["runs"] if d["validity"] is not None]
                if valid:
                    n_true = sum(1 for d in valid if d["validity"] == 1)
                    true_rates.append(n_true / len(valid) * 100)

            mean, _, _, ci_low, ci_high = _mean_ci(true_rates)

            def _r2(v):
                return round(v, 2) if not math.isnan(v) else None

            if agent_type == "search":
                r["search_true_rate_mean"] = _r2(mean)
                r["search_true_rate_ci_low"] = _r2(ci_low)
                r["search_true_rate_ci_high"] = _r2(ci_high)
            else:
                r["no_search_true_rate_mean"] = _r2(mean)
                r["no_search_true_rate_ci_low"] = _r2(ci_low)
                r["no_search_true_rate_ci_high"] = _r2(ci_high)


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
        default=1,
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
        output_dir = Path(args.output_dir)
        raw_dir = output_dir / "raw"
        if not raw_dir.exists():
            raise FileNotFoundError(f"raw directory not found at {raw_dir}")
        results = _compute_flip_categories_from_raw(raw_dir)

        _print_flip_table(results)

        json_path = output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Saved JSON to %s", json_path)

        _write_latex_flip_table(results, output_dir)
        _plot_flip_categories(results, output_dir)

        flip_examples = _collect_flip_examples(raw_dir)
        _write_appendix_flip_table(flip_examples, results, output_dir)

        print(f"Regenerated flip-category analysis in {output_dir}/")
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
    _plot_search_calls(all_results, args.n_runs, output_dir)
    _plot_class_distribution(all_results, args.n_runs, output_dir)

    print(f"\nOutput saved to {output_dir}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(main())
