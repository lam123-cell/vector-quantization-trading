"""Export tq_error (distortion) statistics to CSV for thesis reporting.

Usage example (Colab):
    python scripts/export_tq_metrics.py --dataset drl_dataset_turbo_train.csv --output /content/drive/MyDrive/thesis/tq_metrics_turbo.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export tq_error statistics from dataset")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset CSV")
    parser.add_argument("--output", type=Path, default=repo_root / "tq_metrics.csv", help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    df = pd.read_csv(args.dataset)
    if "tq_error" not in df.columns:
        raise ValueError("Dataset does not contain 'tq_error' column")

    stats = df["tq_error"].describe()
    out_df = stats.to_frame().T
    out_df["median"] = df["tq_error"].median()

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"✓ Saved tq_error stats: {args.output}")

    # attempt copy to Drive (Colab)
    try:
        import shutil
        drive_path = Path("/content/drive/MyDrive/thesis")
        drive_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(args.output), str(drive_path / args.output.name))
        print(f"[*] Copied tq metrics to Drive: {drive_path / args.output.name}")
    except Exception:
        print("[!] Drive copy for tq metrics skipped or failed")


if __name__ == "__main__":
    main()
