"""Export daily annotator label statistics as a CSV file.

Computes day-by-day aggregation: for each annotator column, what percentage
of rows (where the column is non-NULL) have a given label value.

Usage:
    python scripts/export/daily_annotator_stats.py
"""

import asyncio
import csv
from collections import defaultdict
from datetime import date, datetime

from s3_data_tool import S3DataTool

# ---------------------------------------------------------------------------
# Configuration — edit these constants to change what gets analyzed
# ---------------------------------------------------------------------------

DATASET_NAME = "posts"

# Base columns to fetch from the dataset ("created_at" is required).
BASE_COLUMNS = ["created_at"]

# Annotator name → list of column names to fetch and analyze.
ANNOTATOR_COLUMNS: dict[str, list[str]] = {
    "feasibility_classifier_001": ["classifier_label"],
}

# The label value to count as "positive" when computing percentages.
LABEL_OF_INTEREST = "LABEL_2"

OUTPUT_PATH = "output/daily_annotator_stats.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_date(value) -> str | None:
    """Extract YYYY-MM-DD from a datetime, date, or ISO-timestamp string."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    s = str(value)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return s[:10] if len(s) >= 10 else None


def _flat_columns(annotator_columns: dict[str, list[str]]) -> list[str]:
    """Return a flat list of all column names across annotators."""
    cols: list[str] = []
    for _, columns in annotator_columns.items():
        cols.extend(columns)
    return cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    columns_of_interest = _flat_columns(ANNOTATOR_COLUMNS)

    # {date: {column_name: {"total": int, "label": int}}}
    daily: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "label": 0})
    )

    async with S3DataTool().filter_for_annotation(
        annotator_name="daily_stats_export",
        name=DATASET_NAME,
        base_columns=BASE_COLUMNS,
        annotator_columns=ANNOTATOR_COLUMNS,
    ) as generator:
        async for item in generator:
            row = item.data
            day = to_date(row.get("created_at"))
            if day is None:
                continue
            for col in columns_of_interest:
                val = row.get(col)
                if val is not None:
                    daily[day][col]["total"] += 1
                    if val == LABEL_OF_INTEREST:
                        daily[day][col]["label"] += 1

    # Write CSV
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "column", "total_non_null", "label_count", "label_pct"])
        for day in sorted(daily):
            for col, counts in daily[day].items():
                total = counts["total"]
                label_count = counts["label"]
                pct = (label_count / total * 100) if total > 0 else 0.0
                writer.writerow([day, col, total, label_count, f"{pct:.2f}"])

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
