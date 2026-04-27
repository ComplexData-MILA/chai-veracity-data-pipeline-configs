import asyncio
import itertools
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic

from scripts.cluster.clustering import _collect, cluster_knn_graph

from s3_data_tool import DataItem

logger = logging.getLogger(__name__)

K_VALUES = [2, 3, 5, 10, 15, 20, 30]
DROP_FRAC_VALUES = [0.0, 0.5, 0.8, 0.9, 0.95]
OUTPUT_DIR = Path("outputs/clustering_sweep")


class Result(pydantic.BaseModel):
    k: int
    drop_frac: float
    clusters: list[list[DataItem]]


def _write_text_dump(
    clusters: list[list],
    path: Path,
) -> None:
    with open(path, "w") as f:
        for i, cluster in enumerate(clusters):
            f.write(f"=== Cluster {i} (size: {len(cluster)}) ===\n")
            for item in cluster:
                text = item.data.get("text", "").strip()
                if text:
                    f.write(f"  {text}\n")
            f.write("\n")


def _plot_sweep(
    results: list[Result],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    for r in results:
        sizes = sorted((len(c) for c in r.clusters), reverse=True)
        cumsum = np.cumsum(sizes)
        label = f"k={r.k}, drop={r.drop_frac:.2f}"
        xs = range(1, len(cumsum) + 1)
        ax.plot(xs, cumsum, label=label, alpha=0.7)
        ax.scatter(xs, cumsum, s=12, alpha=0.9)

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Cumulative number of items")
    ax.set_title("Cumulative distribution of items per cluster")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Saved plot to %s", output_path)
    plt.close(fig)


async def main() -> None:
    collected = await _collect()
    if not collected.rows:
        logger.warning("No embeddings found; nothing to cluster.")
        return

    logger.info(
        "Collected %d embeddings. Sweeping %d combinations...",
        len(collected.rows),
        len(K_VALUES) * len(DROP_FRAC_VALUES),
    )

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[Result] = []

    for k, drop_frac in itertools.product(K_VALUES, DROP_FRAC_VALUES):
        label = f"k{k}_drop{int(drop_frac * 100):03d}"
        logger.info("Running: k=%d, drop_frac=%.2f", k, drop_frac)

        clusters = cluster_knn_graph(
            collected.rows, collected.embeddings, k=k, drop_frac=drop_frac
        )

        results.append(Result(k=k, drop_frac=drop_frac, clusters=clusters))

        txt_path = output_dir / f"{label}.txt"
        _write_text_dump(clusters, txt_path)
        logger.info("  -> %d clusters, text dump: %s", len(clusters), txt_path)

    table = pd.DataFrame(
        [(r.k, r.drop_frac, len(r.clusters)) for r in results],
        columns=["k", "drop_frac", "n_clusters"],
    ).pivot_table(index="k", columns="drop_frac", values="n_clusters", sort=False)
    table.index.name = "k \\ drop_frac"
    print(table.to_string())
    print(f"Number of rows: {len(collected.rows)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
