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
import pyarrow.parquet as pq
from datasets import load_dataset
from datasets.exceptions import DatasetGenerationError
from huggingface_hub import hf_hub_download, list_repo_files
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

with open("templates/feasibility_filter.txt") as f:
    TEMPLATE = f.read()

TEXT_COLUMN_CANDIDATES = ["claim", "statement", "text", "title"]

DEFAULT_DATASETS = {
    "chai-veracity (clustered)": {
        "path": "ComplexDataLab/chai-veracity-dry-run-20260521-dbscan-diverse",
        "split": "train",
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


def _sanitize_dirname(name: str) -> str:
    """Convert a dataset name to a safe directory name."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


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
    seed: int = 0,
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
                seed=seed,
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


def _load_parquet_fallback(
    path: str, subset: str | None, split: str | None
) -> list[dict]:
    """Load a HF dataset by reading parquet files directly, bypassing schema checks."""
    all_files = list_repo_files(path, repo_type="dataset")
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    logger.info("  Found %d parquet files on HF Hub", len(parquet_files))

    # Filter by subset/config directory
    if subset:
        prefix = f"data/{subset}/"
        parquet_files = [f for f in parquet_files if f.startswith(prefix)]
        if not parquet_files:
            prefix = f"{subset}/"
            parquet_files = [f for f in parquet_files if f.startswith(prefix)]

    # Filter by split in filename
    if split:
        parquet_files = [
            f for f in parquet_files
            if f"{split}-" in os.path.basename(f)
        ]

    if not parquet_files:
        raise ValueError(
            f"No parquet files found for {path} subset={subset} split={split}"
        )

    rows: list[dict] = []
    for pf in parquet_files:
        local = hf_hub_download(path, pf, repo_type="dataset")
        table = pq.read_table(local)
        for i in range(len(table)):
            rows.append({col: table[col][i].as_py() for col in table.column_names})
    return rows


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

        try:
            ds = load_dataset(**kwargs)
        except DatasetGenerationError:
            logger.warning(
                "  load_dataset failed (schema mismatch), falling back to parquet..."
            )
            rows = _load_parquet_fallback(
                cfg["path"], cfg.get("subset"), cfg.get("split"),
            )
            text_col = _get_text_column(list(rows[0].keys()) if rows else [])
            if text_col is None:
                raise ValueError(
                    f"Could not find text column in fallback for {name}. "
                    f"Available columns: {list(rows[0].keys()) if rows else []}"
                )
            texts = [row[text_col] for row in rows]
            _subsample_and_store(name, texts, n_samples, seed, dataset_texts)
            continue

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

        texts = [row[text_col] for row in ds]
        _subsample_and_store(name, texts, n_samples, seed, dataset_texts)

    return dataset_texts


def _subsample_and_store(
    name: str,
    texts: list[str],
    n_samples: int,
    seed: int,
    dataset_texts: dict[str, list[str]],
) -> None:
    n_total = len(texts)
    if n_total > n_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(n_total, size=n_samples, replace=False)
        selected = [texts[int(i)] for i in indices]
    else:
        selected = texts
    dataset_texts[name] = selected
    logger.info("  Subsampled %d examples (from %d total)", len(selected), n_total)


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
            all_tasks.append(
                (
                    run_idx,
                    ex_idx,
                    _judge_single(text, model_name, client, semaphore, seed=ex_idx),
                )
            )

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


def _compute_class_pcts(results: list[dict]) -> list[dict]:
    """Compute per-class percentages from results, counting failed (None)
    judgments as class 0. Returns list of dicts with keys pct_0, pct_1, pct_2.
    """
    out = []
    for r in results:
        total_attempts = r["total_judged"] + r["total_failed"]
        out.append(
            {
                "dataset": r["dataset"],
                "pct_0": (r["total_0"] + r["total_failed"]) / total_attempts * 100,
                "pct_1": r["total_1"] / total_attempts * 100,
                "pct_2": r["total_2"] / total_attempts * 100,
            }
        )
    return out


def _plot(results: list[dict], output_dir: Path) -> None:
    """Plot stacked bar chart of feasibility class distribution per dataset.
    Failed (None) judgments are counted as class 0. No confidence intervals.
    """
    plt.rcParams.update({"font.size": 18})

    fig, ax = plt.subplots(figsize=(9, 6))

    pcts = _compute_class_pcts(results)
    dataset_names = [p["dataset"] for p in pcts]
    x = np.arange(len(dataset_names))
    width = 0.55

    pct_0 = [p["pct_0"] for p in pcts]
    pct_1 = [p["pct_1"] for p in pcts]
    pct_2 = [p["pct_2"] for p in pcts]
    bottom_1 = pct_0
    bottom_2 = [a + b for a, b in zip(pct_0, pct_1)]

    ax.bar(x, pct_0, width, label="Class 0 (not feasible)", color="#999999")
    ax.bar(
        x, pct_1, width, bottom=bottom_1, label="Class 1 (ambiguous)", color="#7CB5EC"
    )
    ax.bar(x, pct_2, width, bottom=bottom_2, label="Class 2 (clear)", color="#2E75B6")

    for i, (m0, m1, m2) in enumerate(zip(pct_0, pct_1, pct_2)):
        if m0 > 6:
            ax.text(
                i,
                m0 / 2,
                f"{m0:.1f}%",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="white",
            )
        if m1 > 6:
            ax.text(
                i,
                bottom_1[i] + m1 / 2,
                f"{m1:.1f}%",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="white",
            )
        if m2 > 6:
            ax.text(
                i,
                bottom_2[i] + m2 / 2,
                f"{m2:.1f}%",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="white",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylabel("Percentage of examples (%)")
    ax.set_title(
        "Feasibility Distribution by Dataset\n(LLM-as-a-Judge feasibility filter)"
    )
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=14, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = output_dir / "feasibility_comparison.png"
    fig.savefig(plot_path, dpi=150)
    logger.info("Saved plot to %s", plot_path)
    pdf_path = output_dir / "feasibility_comparison.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    logger.info("Saved PDF to %s", pdf_path)
    plt.close(fig)


def _write_latex_table(results: list[dict], output_dir: Path) -> None:
    """Write a LaTeX table with per-class accuracy means and 95% CIs.
    Failed (None) judgments are counted as class 0.
    """
    rows = []
    for r in results:
        per_run = r["per_run_judgments"]
        run_pct_0: list[float] = []
        run_pct_1: list[float] = []
        run_pct_2: list[float] = []

        for run_key in sorted(per_run.keys(), key=lambda k: int(k.split("_")[1])):
            judgments = per_run[run_key]
            n_total = len(judgments)
            n_0 = sum(1 for j in judgments if j == 0 or j is None)
            n_1 = sum(1 for j in judgments if j == 1)
            n_2 = sum(1 for j in judgments if j == 2)
            run_pct_0.append(n_0 / n_total * 100)
            run_pct_1.append(n_1 / n_total * 100)
            run_pct_2.append(n_2 / n_total * 100)

        m0, _, _, c0l, c0h = _mean_ci(run_pct_0)
        m1, _, _, c1l, c1h = _mean_ci(run_pct_1)
        m2, _, _, c2l, c2h = _mean_ci(run_pct_2)

        rows.append(
            {
                "dataset": r["dataset"],
                "class_0": f"{m0:.1f} [{c0l:.1f}, {c0h:.1f}]",
                "class_1": f"{m1:.1f} [{c1l:.1f}, {c1h:.1f}]",
                "class_2": f"{m2:.1f} [{c2l:.1f}, {c2h:.1f}]",
            }
        )

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & Class 0 (not feasible) & Class 1 (ambiguous) & Class 2 (clear) \\",
        r"\midrule",
    ]
    for row in rows:
        ds = row["dataset"].replace("_", r"\_")
        lines.append(
            f"  {ds} & {row['class_0']} & {row['class_1']} & {row['class_2']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Per-class feasibility percentages with 95\% confidence intervals"
            r" (t-distribution, $n=" + str(results[0]["n_judge_runs"]) + r"$ runs)."
            r" Failed judgments are counted as class 0 (not feasible).}",
            r"\label{tab:per-class-ci}",
            r"\end{table}",
        ]
    )

    tex_path = output_dir / "per_class_ci_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved LaTeX table to %s", tex_path)


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
    parser.add_argument(
        "--dataset-name", default=None,
        help="Single-dataset mode: display name for the dataset",
    )
    parser.add_argument(
        "--dataset-path", default=None,
        help="Single-dataset mode: HF Hub path (e.g. ComplexDataLab/chai-veracity)",
    )
    parser.add_argument(
        "--dataset-subset", default=None,
        help="Single-dataset mode: optional subset/config name",
    )
    parser.add_argument(
        "--dataset-split", default=None,
        help="Single-dataset mode: optional split name",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate plots and LaTeX table from existing results.json (no LLM calls)",
    )
    args = parser.parse_args()

    if args.regenerate:
        json_path = Path(args.output_dir) / "results.json"
        if not json_path.exists():
            raise FileNotFoundError(f"results.json not found at {json_path}")
        with open(json_path) as f:
            results = json.load(f)
        output_dir = Path(args.output_dir)
        _plot(results, output_dir)
        _write_latex_table(results, output_dir)
        print(f"Regenerated plots and LaTeX table in {output_dir}/")
        return

    model_name = args.model_name or os.environ["MODEL_NAME"]

    # Dataset configs
    if args.dataset_path:
        dataset_name = args.dataset_name or args.dataset_path
        datasets_config = {
            dataset_name: {
                "path": args.dataset_path,
                "subset": args.dataset_subset,
                "split": args.dataset_split,
            }
        }
        output_dir = Path(args.output_dir) / _sanitize_dirname(dataset_name)
    elif args.datasets:
        with open(args.datasets) as f:
            datasets_config = json.load(f)
        output_dir = Path(args.output_dir)
    else:
        datasets_config = DEFAULT_DATASETS
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

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
    _plot(results, output_dir)

    print(f"Output saved to {output_dir}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(main())
