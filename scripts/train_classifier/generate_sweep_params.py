"""Generate a params.txt file for SLURM job-array hyperparameter sweeps.

Each line is a set of KEY=value pairs that get sourced by the array job script.
"""

import argparse
import itertools
from pathlib import Path


def float_range(s: str) -> list[float]:
    """Parse a float range: '1e-5,5e-5,1e-4' or '1e-5'."""
    return [float(x.strip()) for x in s.split(",")]


def int_range(s: str) -> list[int]:
    """Parse an int range: '8,16,32' or '16'."""
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Generate sweep params file for SLURM job array")
    parser.add_argument("--lr", type=float_range, default="5e-5", help="Comma-separated learning rates")
    parser.add_argument("--batch_size", type=int_range, default="16", help="Comma-separated batch sizes")
    parser.add_argument("--weight_decay", type=float_range, default="0.01", help="Comma-separated weight decays")
    parser.add_argument("--seed", type=int_range, default="42", help="Comma-separated random seeds")
    parser.add_argument("--epochs", type=int_range, default="10", help="Comma-separated epoch counts")
    parser.add_argument("--warmup_ratio", type=float_range, default="0.1", help="Comma-separated warmup ratios")
    parser.add_argument("--max_length", type=int_range, default="512", help="Comma-separated max token lengths")
    parser.add_argument("--output", type=str, default="params.txt", help="Output file path")
    args = parser.parse_args()

    keys = ["LR", "BATCH_SIZE", "WEIGHT_DECAY", "SEED", "EPOCHS", "WARMUP_RATIO", "MAX_LENGTH"]
    value_lists = [
        args.lr,
        args.batch_size,
        args.weight_decay,
        args.seed,
        args.epochs,
        args.warmup_ratio,
        args.max_length,
    ]

    # Only sweep over parameters with multiple values
    sweep_keys = [k for k, v in zip(keys, value_lists) if len(v) > 1]
    sweep_values = [v for v in value_lists if len(v) > 1]
    fixed = {k: v[0] for k, v in zip(keys, value_lists) if len(v) == 1}

    if not sweep_keys:
        sweep_keys = keys
        sweep_values = value_lists
        fixed = {}

    lines = []
    for combo in itertools.product(*sweep_values):
        params = dict(fixed)
        for k, v in zip(sweep_keys, combo):
            params[k] = v
        # Format: space-separated KEY=value pairs (sourced by bash)
        line = " ".join(f"{k}={v}" for k, v in params.items())
        lines.append(line)

    output_path = Path(args.output)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} experiments to {output_path}")
    if fixed:
        print(f"  fixed: {fixed}")
    print(f"  sweep over: {sweep_keys}")


if __name__ == "__main__":
    main()
