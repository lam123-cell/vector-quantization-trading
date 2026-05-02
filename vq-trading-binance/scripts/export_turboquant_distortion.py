from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset_master.csv"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "figures" / "chapter_3"


ORIGINAL_COLUMNS = [
    "n_log_return",
    "n_return_5",
    "n_log_volume",
    "n_candle_body",
    "n_rsi",
    "n_macd",
    "n_macd_signal",
    "n_volatility",
    "n_atr",
]

RECON_COLUMNS = [
    "tq_xhat_log_return",
    "tq_xhat_return_5",
    "tq_xhat_log_volume",
    "tq_xhat_candle_body",
    "tq_xhat_rsi",
    "tq_xhat_macd",
    "tq_xhat_macd_signal",
    "tq_xhat_volatility",
    "tq_xhat_atr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TurboQuant distortion chart from dataset_master.csv"
    )
    parser.add_argument(
        "--dataset-master",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to dataset_master.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the exported figure and CSV",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="hinh_3_4_distortion.png",
        help="Filename for the exported figure",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="hinh_3_4_distortion_summary.csv",
        help="Filename for the exported summary CSV",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Maximum number of points to plot (uniformly sampled if needed)",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_master: Path) -> Path:
    if not dataset_master.exists():
        raise FileNotFoundError(f"Missing dataset_master.csv: {dataset_master}")
    return dataset_master


def load_dataset(dataset_master: Path) -> pd.DataFrame:
    required_columns = ["time", *ORIGINAL_COLUMNS, *RECON_COLUMNS]
    dataframe = pd.read_csv(dataset_master, usecols=lambda column: column in required_columns)

    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"dataset_master.csv is missing required columns: {missing}")

    dataframe["time"] = pd.to_datetime(dataframe["time"], errors="coerce", utc=True)
    dataframe = dataframe.dropna(subset=["time"])

    for column in ORIGINAL_COLUMNS + RECON_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe = dataframe.dropna(subset=ORIGINAL_COLUMNS + RECON_COLUMNS)
    dataframe = dataframe.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def compute_distortion(dataframe: pd.DataFrame) -> pd.DataFrame:
    original = dataframe[ORIGINAL_COLUMNS].to_numpy(dtype=np.float64)
    recon = dataframe[RECON_COLUMNS].to_numpy(dtype=np.float64)

    distortion_per_row = np.mean((original - recon) ** 2, axis=1)
    per_feature = np.mean((original - recon) ** 2, axis=0)

    summary = pd.DataFrame(
        {
            "time": dataframe["time"],
            "distortion": distortion_per_row,
        }
    )

    feature_summary = pd.DataFrame(
        {
            "feature": [name.replace("n_", "") for name in ORIGINAL_COLUMNS],
            "mse": per_feature,
        }
    )
    return summary, feature_summary


def downsample_for_plot(summary: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(summary) <= max_points:
        return summary

    indices = np.linspace(0, len(summary) - 1, max_points, dtype=int)
    return summary.iloc[indices].reset_index(drop=True)


def export_outputs(
    summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    output_dir: Path,
    figure_name: str,
    csv_name: str,
    max_points: int,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / csv_name
    feature_path = output_dir / f"{Path(csv_name).stem}_by_feature.csv"
    figure_path = output_dir / figure_name

    summary.to_csv(summary_path, index=False)
    feature_summary.to_csv(feature_path, index=False)

    plot_data = downsample_for_plot(summary, max_points=max_points)
    avg_distortion = float(summary["distortion"].mean())
    median_distortion = float(summary["distortion"].median())

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(plot_data["time"], plot_data["distortion"], color="#2563eb", linewidth=1.4)
    ax.axhline(avg_distortion, color="#dc2626", linestyle="--", linewidth=1.4, label=f"Mean = {avg_distortion:.6f}")
    ax.set_title("Hình 3.4. Distortion giữa dữ liệu gốc và dữ liệu sau lượng tử hóa theo thời gian")
    ax.set_xlabel("Time")
    ax.set_ylabel("Distortion (MSE per vector)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("=" * 80)
    print("EXPORT TURBOQUANT DISTORTION")
    print("=" * 80)
    print(f"Dataset         : {summary_path}")
    print(f"Figure          : {figure_path}")
    print(f"Feature summary  : {feature_path}")
    print(f"Rows            : {len(summary):,}")
    print(f"Average MSE     : {avg_distortion:.8f}")
    print(f"Median MSE      : {median_distortion:.8f}")
    print(f"Max MSE         : {float(summary['distortion'].max()):.8f}")
    print("=" * 80)

    return figure_path, summary_path, feature_path


def main() -> int:
    args = parse_args()

    dataset_master = resolve_dataset_path(args.dataset_master)
    dataframe = load_dataset(dataset_master)
    summary, feature_summary = compute_distortion(dataframe)
    export_outputs(
        summary=summary,
        feature_summary=feature_summary,
        output_dir=args.output_dir,
        figure_name=args.figure_name,
        csv_name=args.csv_name,
        max_points=args.max_points,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())