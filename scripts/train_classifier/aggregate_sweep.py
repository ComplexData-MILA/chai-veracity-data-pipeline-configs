"""Aggregate hyperparameter sweep results from SLURM stdout logs.

For each log, extracts the hyperparameters and the epoch with lowest val_loss,
then produces a pandas table sorted by val_loss.

Usage:
  python scripts/train_classifier/aggregate_sweep.py logs/*.out
  python scripts/train_classifier/aggregate_sweep.py < combined_log.txt
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def parse_log(text: str) -> dict | None:
    """Parse a single SLURM stdout log. Returns best-epoch row or None."""
    # Extract hyperparams from "Task N params: KEY=val ..."
    params_match = re.search(r"Task\s+\d+\s+params:\s*(.+)", text)
    if not params_match:
        return None

    params = {}
    for pair in params_match.group(1).split():
        k, v = pair.split("=", 1)
        try:
            v = float(v)
            if v == int(v) and k not in ("LR", "WEIGHT_DECAY", "WARMUP_RATIO"):
                v = int(v)
        except ValueError:
            pass
        params[k] = v

    # Extract all epoch lines and find the one with minimum val_loss
    epoch_re = re.compile(
        r"epoch\s+(\d+)/(\d+)\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+val_acc=([\d.]+)"
    )
    best = None
    for m in epoch_re.finditer(text):
        val_loss = float(m.group(4))
        if best is None or val_loss < best["val_loss"]:
            best = {
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(3)),
                "val_loss": val_loss,
                "val_acc": float(m.group(5)),
            }

    if best is None:
        return None

    row = {**params, **best}

    # Extract test metrics if present
    test_match = re.search(r"test_loss=([\d.]+)\s+test_acc=([\d.]+)", text)
    if test_match:
        row["test_loss"] = float(test_match.group(1))
        row["test_acc"] = float(test_match.group(2))

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate classifier sweep logs into a sorted table"
    )
    parser.add_argument(
        "files", nargs="*", help="Log files to parse (reads stdin if none given)"
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Save to CSV file"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Add a 'source' column with this label (useful when combining sweeps)"
    )
    args = parser.parse_args()

    rows = []
    if args.files:
        for path in args.files:
            row = parse_log(Path(path).read_text())
            if row:
                if args.source:
                    row["source"] = args.source
                row["log_file"] = str(path)
                rows.append(row)
            else:
                print(f"Warning: no parseable sweep data in {path}", file=sys.stderr)
    else:
        text = sys.stdin.read()
        # Split on sentinel that separates runs in combined logs
        for chunk in re.split(r"\n(?=Task\s+\d+\s+params:)", text):
            row = parse_log(chunk)
            if row:
                rows.append(row)

    if not rows:
        print("No sweep results found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("val_loss", ascending=True).reset_index(drop=True)

    col_order = ["val_loss", "val_acc", "test_loss", "test_acc", "epoch", "train_loss",
                 "LR", "BATCH_SIZE", "WEIGHT_DECAY", "SEED"]
    if args.source:
        col_order.insert(0, "source")
    if args.files:
        col_order.append("log_file")
    available = [c for c in col_order if c in df.columns]
    extra = [c for c in df.columns if c not in available]
    df = df[available + extra]

    print(df.to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved {len(df)} rows to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
