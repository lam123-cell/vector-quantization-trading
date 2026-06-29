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
        default="distortion.png",
        help="Filename for the exported figure",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="distortion_summary.csv",
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


# Load dataset_master.csv, ensuring required columns are present 
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


def compute_distortion(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    original = dataframe[ORIGINAL_COLUMNS].to_numpy(dtype=np.float64)
    recon = dataframe[RECON_COLUMNS].to_numpy(dtype=np.float64)

    # MSE = (1/n) * Σ(x_i - x̂_i)^2
    distortion_per_row = np.mean((original - recon) ** 2, axis=1)
    per_feature = np.mean((original - recon) ** 2, axis=0)
    
    # MAE = (1/n) * Σ|x_i - x̂_i|
    mae_per_row = np.mean(np.abs(original - recon), axis=1)

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
    return summary, feature_summary, mae_per_row, distortion_per_row


def downsample_for_plot(summary: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(summary) <= max_points:
        return summary

    indices = np.linspace(0, len(summary) - 1, max_points, dtype=int)
    return summary.iloc[indices].reset_index(drop=True)


def plot_histogram(mae: np.ndarray, mse: np.ndarray, output_dir: Path) -> Path:
    """Plot Histogram - Reconstruction Error Distribution (MAE & MSE)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "reconstruction_error_histogram.png"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Reconstruction Error Distribution - TurboQuant", 
                 fontsize=14, fontweight="bold")
    
    # MAE histogram
    ax = axes[0]
    ax.hist(mae, bins=100, color="#3b82f6", alpha=0.7, edgecolor="black")
    ax.axvline(mae.mean(), color="red", linestyle="--", linewidth=2, 
               label=f"Mean = {mae.mean():.6f}")
    ax.axvline(np.median(mae), color="orange", linestyle="--", linewidth=2,
               label=f"Median = {np.median(mae):.6f}")
    ax.set_xlabel("MAE (Mean Absolute Error)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"MAE Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MSE histogram  
    ax = axes[1]
    ax.hist(mse, bins=100, color="#10b981", alpha=0.7, edgecolor="black")
    ax.axvline(mse.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean = {mse.mean():.6f}")
    ax.axvline(np.median(mse), color="orange", linestyle="--", linewidth=2,
               label=f"Median = {np.median(mse):.6f}")
    ax.set_xlabel("MSE (Mean Squared Error)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"MSE Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    
    return fig_path


def plot_comparison(dataframe: pd.DataFrame, mae: np.ndarray, output_dir: Path, 
                    feature_orig: str, feature_recon: str, max_points: int = 3000) -> Path:
    """Plot Line Chart - Original vs Reconstructed data"""
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_name = feature_orig.replace("n_", "")
    fig_path = output_dir / f"original_vs_reconstructed_{feat_name}.png"
    
    # Sample data
    if len(dataframe) > max_points:
        indices = np.linspace(0, len(dataframe) - 1, max_points, dtype=int)
        df_plot = dataframe.iloc[indices].reset_index(drop=True)
        mae_plot = mae[indices]
    else:
        df_plot = dataframe.copy()
        mae_plot = mae
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    fig.suptitle(f"Original vs Reconstructed: {feat_name}", 
                 fontsize=14, fontweight="bold")
    
    # Top: Feature comparison
    ax = axes[0]
    ax.plot(df_plot["time"], df_plot[feature_orig], 
            label="Original", linewidth=1.8, color="#2563eb", alpha=0.8)
    ax.plot(df_plot["time"], df_plot[feature_recon], 
            label="Reconstructed (TurboQuant)", linewidth=1.8, 
            color="#dc2626", alpha=0.7, linestyle="--")
    ax.fill_between(df_plot["time"], df_plot[feature_orig], 
                     df_plot[feature_recon], alpha=0.1, color="#8b5cf6")
    ax.set_ylabel("Normalized Value")
    ax.set_title("Feature Reconstruction")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # Bottom: Error over time (MAE)
    ax = axes[1]
    ax.plot(df_plot["time"], mae_plot, label="MAE", 
            linewidth=1.5, color="#f59e0b", alpha=0.8)
    ax.fill_between(df_plot["time"], mae_plot, alpha=0.2, color="#f59e0b")
    ax.axhline(mae_plot.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean MAE = {mae_plot.mean():.6f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("MAE")
    ax.set_title("Reconstruction Error Over Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    
    return fig_path


def downsample_for_plot(summary: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(summary) <= max_points:
        return summary

    indices = np.linspace(0, len(summary) - 1, max_points, dtype=int)
    return summary.iloc[indices].reset_index(drop=True)


def export_outputs(
    summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    dataframe: pd.DataFrame,
    mae: np.ndarray,
    mse: np.ndarray,
    output_dir: Path,
    figure_name: str,
    csv_name: str,
    max_points: int,
) -> tuple[Path, Path, Path, Path, Path]:
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
    ax.set_title("Distortion giữa dữ liệu gốc và dữ liệu sau lượng tử hóa theo thời gian")
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
    
    # Generate visualization for thesis 4.4.1
    print("\nGenerating visualization for thesis Section 4.4.1...")
    histogram_path = plot_histogram(mae, mse, output_dir)
    print(f"✓ Histogram saved: {histogram_path}")

    return figure_path, summary_path, feature_path, histogram_path


def main() -> int:
    args = parse_args()

    dataset_master = resolve_dataset_path(args.dataset_master)
    dataframe = load_dataset(dataset_master)
    summary, feature_summary, mae, mse = compute_distortion(dataframe)
    export_outputs(
        summary=summary,
        feature_summary=feature_summary,
        dataframe=dataframe,
        mae=mae,
        mse=mse,
        output_dir=args.output_dir,
        figure_name=args.figure_name,
        csv_name=args.csv_name,
        max_points=args.max_points,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())