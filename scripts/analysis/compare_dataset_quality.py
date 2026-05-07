"""Compare dataset quality across HF datasets using LLM-as-a-Judge feasibility labels.

Each example is judged n times (default: 5) to obtain a t-distribution
confidence interval on the percentage of examples in each feasibility class
(0 = not feasible, 1 = ambiguous, 2 = clear).

Usage:
    uv run --env-file .env python scripts/analysis/compare_dataset_quality.py \
        --model-name $MODEL_NAME --n-samples 100 --n-judge-runs 5
"""

import argparse
import asyncio
import json
import logging
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
from datasets import load_dataset
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

with open("templates/feasibility_filter.txt") as f:
    TEMPLATE = f.read()

TEXT_COLUMN_CANDIDATES = ["claim", "statement", "text", "title"]

DEFAULT_DATASETS = {
    "chai-veracity (clustered)": {
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


def _get_text_column(columns: list[str]) -> str | None:
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def _parse_verdict(output: str) -> tuple[str, int]:
    """Extract explanation and feasibility rating (0-2) from model output."""
    parts = output.split("|")
    if len(parts) < 2:
        raise ValueError(f"Output missing '|' delimiter. Output: {output[:200]}")
    explanation = "|".join(parts[:-1])
    verdict_str = parts[-1]
    match = re.search(r"\b([0-2])\b", verdict_str)
    if match is None:
        raise ValueError(
            f"Could not parse feasibility rating from: {verdict_str[:200]}"
        )
    return explanation, int(match.group(1))


async def _judge_single(
    text: str,
    model_name: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> int | None:
    """Run one feasibility judgment. Returns 0/1/2 or None on failure."""
    if not text or not text.strip():
        return None

    prompt = TEMPLATE.format(text=text)
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=16384,
            )
            output = response.choices[0].message.content
            if output is None:
                return None
            _, verdict = _parse_verdict(output)
            return verdict
        except Exception:
            return None


def _mean_ci(values: list[float]) -> tuple[float, float, float, float, float]:
    """Return (mean, std, sem, ci_low, ci_high) for a list of per-run percentages."""
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


def _load_datasets(
    configs: dict, n_samples: int, seed: int
) -> dict[str, list[str]]:
    """Load and subsample datasets from HuggingFace Hub.

    Returns dict mapping dataset display name -> list of text strings.
    """
    dataset_texts: dict[str, list[str]] = {}

    for name, cfg in configs.items():
        logger.info("Loading dataset: %s", name)
        kwargs = {"path": cfg["path"]}
        if cfg.get("subset"):
            kwargs["name"] = cfg["subset"]
        if cfg.get("split"):
            kwargs["split"] = cfg["split"]

        ds = load_dataset(**kwargs)
        # DatasetDict -> pick the right split
        if hasattr(ds, "keys"):
            split_name = cfg.get("split") or list(ds.keys())[0]
            ds = ds[split_name]

        text_col = _get_text_column(ds.column_names)
        if text_col is None:
            raise ValueError(
                f"Could not find text column in {name}. "
                f"Available columns: {ds.column_names}"
            )
        logger.info("  Text column: '%s' (%d total examples)", text_col, len(ds))

        n_total = len(ds)
        if n_total > n_samples:
            rng = np.random.RandomState(seed)
            indices = rng.choice(n_total, size=n_samples, replace=False)
            texts = [ds[int(i)][text_col] for i in indices]
        else:
            texts = [row[text_col] for row in ds]

        dataset_texts[name] = texts
        logger.info("  Subsampled %d examples", len(texts))

    return dataset_texts


async def _judge_dataset(
    name: str,
    texts: list[str],
    n_judge_runs: int,
    model_name: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Judge all examples in a dataset across n independent runs.

    Returns a dict with per-class percentages, CIs, and per-run judgments.
    """
    n_examples = len(texts)
    total_calls = n_examples * n_judge_runs
    logger.info(
        "  Judging: %d examples x %d runs = %d LLM calls",
        n_examples, n_judge_runs, total_calls,
    )

    # Build all tasks: (run_idx, example_idx, coroutine)
    all_tasks = []
    for run_idx in range(n_judge_runs):
        for ex_idx, text in enumerate(texts):
            all_tasks.append((
                run_idx,
                ex_idx,
                _judge_single(text, model_name, client, semaphore),
            ))

    judgments = await asyncio.gather(*[t[2] for t in all_tasks])

    # Group judgments by run
    per_run: dict[int, list[int | None]] = {
        i: [] for i in range(n_judge_runs)
    }
    for (run_idx, ex_idx, _), judgment in zip(all_tasks, judgments):
        per_run[run_idx].append(judgment)

    # Compute per-run class percentages
    run_pct_0: list[float] = []
    run_pct_1: list[float] = []
    run_pct_2: list[float] = []
    run_pct_feasible: list[float] = []
    total_valid = 0
    total_0 = 0
    total_1 = 0
    total_2 = 0
    total_failed = 0

    for run_idx in range(n_judge_runs):
        jlist = per_run[run_idx]
        n_valid = sum(1 for j in jlist if j is not None)
        n_0 = sum(1 for j in jlist if j == 0)
        n_1 = sum(1 for j in jlist if j == 1)
        n_2 = sum(1 for j in jlist if j == 2)
        n_failed = sum(1 for j in jlist if j is None)

        total_valid += n_valid
        total_0 += n_0
        total_1 += n_1
        total_2 += n_2
        total_failed += n_failed

        if n_valid > 0:
            run_pct_0.append(n_0 / n_valid * 100)
            run_pct_1.append(n_1 / n_valid * 100)
            run_pct_2.append(n_2 / n_valid * 100)
            run_pct_feasible.append((n_1 + n_2) / n_valid * 100)

    mean_0, std_0, sem_0, ci0_low, ci0_high = _mean_ci(run_pct_0)
    mean_1, std_1, sem_1, ci1_low, ci1_high = _mean_ci(run_pct_1)
    mean_2, std_2, sem_2, ci2_low, ci2_high = _mean_ci(run_pct_2)
    mean_feas, std_feas, sem_feas, ci_feas_low, ci_feas_high = _mean_ci(run_pct_feasible)

    def _r(v):
        return round(v, 2) if not math.isnan(v) else None

    return {
        "dataset": name,
        "n_examples": n_examples,
        "n_judge_runs": n_judge_runs,
        "total_judged": total_valid,
        "total_failed": total_failed,
        "total_0": total_0,
        "total_1": total_1,
        "total_2": total_2,
        "pct_0_mean": _r(mean_0),
        "pct_0_std": _r(std_0),
        "pct_0_ci_low": _r(ci0_low),
        "pct_0_ci_high": _r(ci0_high),
        "pct_1_mean": _r(mean_1),
        "pct_1_std": _r(std_1),
        "pct_1_ci_low": _r(ci1_low),
        "pct_1_ci_high": _r(ci1_high),
        "pct_2_mean": _r(mean_2),
        "pct_2_std": _r(std_2),
        "pct_2_ci_low": _r(ci2_low),
        "pct_2_ci_high": _r(ci2_high),
        "pct_feasible_mean": _r(mean_feas),
        "pct_feasible_std": _r(std_feas),
        "pct_feasible_ci_low": _r(ci_feas_low),
        "pct_feasible_ci_high": _r(ci_feas_high),
        "run_pct_0": [_r(v) for v in run_pct_0],
        "run_pct_1": [_r(v) for v in run_pct_1],
        "run_pct_2": [_r(v) for v in run_pct_2],
        "run_pct_feasible": [_r(v) for v in run_pct_feasible],
        # Per-run judgments (for reproducibility)
        "per_run_judgments": {
            f"run_{i}": per_run[i] for i in range(n_judge_runs)
        },
    }


def _plot(results: list[dict], output_dir: Path, n_judge_runs: int) -> None:
    """Plot stacked bar chart of feasibility class distribution per dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))

    dataset_names = [r["dataset"] for r in results]
    x = np.arange(len(dataset_names))
    width = 0.55

    pct_1 = [r["pct_1_mean"] for r in results]
    pct_2 = [r["pct_2_mean"] for r in results]

    # Stacked bars: class 1 bottom, class 2 top
    ax.bar(x, pct_1, width, label="Class 1 (ambiguous)", color="#7CB5EC")
    ax.bar(x, pct_2, width, bottom=pct_1, label="Class 2 (clear)", color="#2E75B6")

    # Error bars on total feasible (class 1 + 2) showing 95% CI
    feasible_means = [r["pct_feasible_mean"] for r in results]
    yerr_low = []
    yerr_high = []
    for r in results:
        ci_low = r["pct_feasible_ci_low"]
        ci_high = r["pct_feasible_ci_high"]
        if ci_low is not None and ci_high is not None:
            yerr_low.append(r["pct_feasible_mean"] - ci_low)
            yerr_high.append(ci_high - r["pct_feasible_mean"])
        else:
            yerr_low.append(0)
            yerr_high.append(0)

    ax.errorbar(
        x, feasible_means,
        yerr=[yerr_low, yerr_high],
        fmt="none", ecolor="black", capsize=6, capthick=1.5, linewidth=1.5,
        label=f"95% CI (t-dist, n={n_judge_runs} runs)",
    )

    # Annotate percentages
    for i, (m1, m2, mf) in enumerate(zip(pct_1, pct_2, feasible_means)):
        if m1 is not None and m1 > 6:
            ax.text(i, m1 / 2, f"{m1:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        if m2 is not None and m2 > 6:
            ax.text(i, m1 + m2 / 2, f"{m2:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        # Feasible total above bar with CI range
        r = results[i]
        ci_low = r["pct_feasible_ci_low"]
        ci_high = r["pct_feasible_ci_high"]
        ci_str = ""
        if ci_low is not None and ci_high is not None:
            ci_str = f"  [{ci_low:.1f}, {ci_high:.1f}]"
        ax.text(i, mf + 2.5, f"{mf:.1f}%{ci_str}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, fontsize=10)
    ax.set_ylabel("Percentage of examples (%)", fontsize=12)
    ax.set_title(
        "Feasibility Distribution by Dataset\n(LLM-as-a-Judge feasibility filter)",
        fontsize=13,
    )
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = output_dir / "feasibility_comparison.png"
    fig.savefig(plot_path, dpi=150)
    logger.info("Saved plot to %s", plot_path)
    plt.close(fig)


def _print_table(results: list[dict]) -> pd.DataFrame:
    """Build DataFrame, print to stdout, and return it."""
    rows = []
    for r in results:
        row = {
            "dataset": r["dataset"],
            "n": r["n_examples"],
            "judged": r["total_judged"],
            "failed": r["total_failed"],
            "class_0_%": r["pct_0_mean"],
            "class_0_CI": (
                f"[{r['pct_0_ci_low']}, {r['pct_0_ci_high']}]"
                if r["pct_0_ci_low"] is not None else "-"
            ),
            "class_1_%": r["pct_1_mean"],
            "class_1_CI": (
                f"[{r['pct_1_ci_low']}, {r['pct_1_ci_high']}]"
                if r["pct_1_ci_low"] is not None else "-"
            ),
            "class_2_%": r["pct_2_mean"],
            "class_2_CI": (
                f"[{r['pct_2_ci_low']}, {r['pct_2_ci_high']}]"
                if r["pct_2_ci_low"] is not None else "-"
            ),
            "feasible_%": r["pct_feasible_mean"],
            "feasible_CI": (
                f"[{r['pct_feasible_ci_low']}, {r['pct_feasible_ci_high']}]"
                if r["pct_feasible_ci_low"] is not None else "-"
            ),
        }
        rows.append(row)
    df = pd.DataFrame(rows)

    print()
    print("=" * 100)
    print("Dataset Quality Comparison — LLM-as-a-Judge Feasibility Labels")
    print("=" * 100)
    print(df.to_string(index=False))
    print()
    return df


async def main():
    parser = argparse.ArgumentParser(
        description="Compare dataset quality via LLM-as-a-Judge feasibility labels",
    )
    parser.add_argument(
        "--model-name", default=None,
        help="OpenAI model name (default: env MODEL_NAME)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of examples to subsample per dataset",
    )
    parser.add_argument(
        "--n-judge-runs", type=int, default=5,
        help="Independent judgment runs per example for t-distribution CI",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=128,
        help="Max concurrent LLM calls",
    )
    parser.add_argument(
        "--output-dir", default="outputs/compare_dataset_quality",
        help="Output directory for CSV, JSON, and plot",
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset configs
    if args.datasets:
        with open(args.datasets) as f:
            datasets_config = json.load(f)
    else:
        datasets_config = DEFAULT_DATASETS

    # Load datasets
    dataset_texts = _load_datasets(datasets_config, args.n_samples, args.seed)

    # LLM client
    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.max_concurrency)

    logger.info(
        "Model: %s | concurrency: %d | n_runs: %d | n_samples: %d",
        model_name, args.max_concurrency, args.n_judge_runs, args.n_samples,
    )

    # Judge each dataset
    results = []
    for name, texts in dataset_texts.items():
        logger.info("=== Dataset: %s ===", name)
        result = await _judge_dataset(
            name=name,
            texts=texts,
            n_judge_runs=args.n_judge_runs,
            model_name=model_name,
            client=client,
            semaphore=semaphore,
        )
        results.append(result)
        feas = result["pct_feasible_mean"]
        ci_low = result["pct_feasible_ci_low"]
        ci_high = result["pct_feasible_ci_high"]
        ci_str = f" [{ci_low:.1f}, {ci_high:.1f}]" if ci_low is not None else ""
        logger.info(
            "  Feasible: %.1f%%%s | class 0: %.1f%% | class 1: %.1f%% | class 2: %.1f%%",
            feas, ci_str,
            result["pct_0_mean"], result["pct_1_mean"], result["pct_2_mean"],
        )

    # Print table
    df = _print_table(results)

    # Save CSV
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV to %s", csv_path)

    # Save JSON (full results with per-run details)
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved JSON to %s", json_path)

    # Plot
    _plot(results, output_dir, args.n_judge_runs)

    print(f"Output saved to {output_dir}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(main())
