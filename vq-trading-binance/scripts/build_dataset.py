from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


FeatureSetName = Literal["baseline", "turbo"]


@dataclass(frozen=True)
class DatasetSpec:
    name: FeatureSetName
    prefix: str
    output_name: str


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(name="baseline", prefix="n_", output_name="drl_dataset_baseline.csv"),
    DatasetSpec(name="turbo", prefix="tq_xhat_", output_name="drl_dataset_turbo.csv"),
)


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_time_column(columns: list[str]) -> str:
    for candidate in ("time", "timestamp", "date"):
        if candidate in columns:
            return candidate
    raise ValueError("dataset_master.csv must contain one of: time, timestamp, date")


def _collect_feature_columns(columns: list[str], prefix: str) -> list[str]:
    return sorted([column for column in columns if column.startswith(prefix)])


def _select_output_columns(columns: list[str], prefix: str) -> list[str]:
    time_column = _resolve_time_column(columns)
    feature_columns = _collect_feature_columns(columns, prefix)
    if not feature_columns:
        raise ValueError(f"dataset_master.csv missing required feature group: {prefix}*")

    ohlcv_columns = [column for column in ["open", "high", "low", "close", "volume"] if column in columns]
    return [time_column] + ohlcv_columns + feature_columns


def _load_and_clean_dataset(dataset_master: Path, output_columns: list[str]) -> pd.DataFrame:
    dataframe = pd.read_csv(dataset_master, usecols=output_columns)

    time_column = output_columns[0]
    dataframe[time_column] = pd.to_datetime(dataframe[time_column], errors="coerce", utc=True)
    dataframe = dataframe.dropna(subset=[time_column])

    numeric_columns = [column for column in output_columns if column != time_column]
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe = dataframe.dropna(subset=numeric_columns)
    dataframe = dataframe.sort_values(time_column).drop_duplicates(subset=[time_column], keep="last")
    dataframe = dataframe.reset_index(drop=True)

    dataframe = dataframe.rename(columns={time_column: "time"})
    dataframe["time"] = dataframe["time"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    dataframe["time"] = dataframe["time"].str.replace(r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True)

    return dataframe


def _chronological_split(dataframe: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    if len(dataframe) < 2:
        raise ValueError("Need at least 2 rows to create a train/test split")

    split_index = int(len(dataframe) * train_ratio)
    split_index = max(1, min(split_index, len(dataframe) - 1))

    train_frame = dataframe.iloc[:split_index].reset_index(drop=True)
    test_frame = dataframe.iloc[split_index:].reset_index(drop=True)
    return train_frame, test_frame


def _write_dataset_exports(base_frame: pd.DataFrame, output_path: Path, train_ratio: float) -> None:
    train_frame, test_frame = _chronological_split(base_frame, train_ratio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_frame.to_csv(output_path, index=False)

    train_path = output_path.with_name(f"{output_path.stem}_train.csv")
    test_path = output_path.with_name(f"{output_path.stem}_test.csv")
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")
    print(f"Rows: total={len(base_frame):,}, train={len(train_frame):,}, test={len(test_frame):,}")


def build_drl_dataset(
    dataset_master: Path,
    output_dir: Path,
    feature_set: FeatureSetName,
    train_ratio: float = 0.8,
) -> pd.DataFrame:
    spec = next(item for item in DATASET_SPECS if item.name == feature_set)

    if not dataset_master.exists():
        raise FileNotFoundError(f"Missing dataset_master.csv: {dataset_master}")

    columns = pd.read_csv(dataset_master, nrows=0).columns.tolist()
    output_columns = _select_output_columns(columns, spec.prefix)
    dataframe = _load_and_clean_dataset(dataset_master, output_columns)

    output_path = output_dir / spec.output_name
    _write_dataset_exports(dataframe, output_path, train_ratio)
    return dataframe


def build_all_datasets(
    dataset_master: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for spec in DATASET_SPECS:
        print("=" * 80)
        print(f"Building DRL dataset: {spec.name}")
        print("=" * 80)
        results[spec.name] = build_drl_dataset(
            dataset_master=dataset_master,
            output_dir=output_dir,
            feature_set=spec.name,
            train_ratio=train_ratio,
        )
    return results


def parse_args() -> argparse.Namespace:
    repo_root = _resolve_repo_root()
    parser = argparse.ArgumentParser(description="Build DRL baseline/turbo datasets from dataset_master.csv")
    parser.add_argument(
        "--dataset-master",
        type=Path,
        default=repo_root / "dataset_master.csv",
        help="Path to dataset_master.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root,
        help="Directory to write drl_dataset_*.csv files",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Chronological train split ratio",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "turbo", "both"],
        default="both",
        help="Which DRL dataset(s) to export",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "both":
        build_all_datasets(args.dataset_master, args.output_dir, args.train_ratio)
        return 0

    feature_set = args.mode
    print("=" * 80)
    print(f"Building DRL dataset: {feature_set}")
    print("=" * 80)
    build_drl_dataset(args.dataset_master, args.output_dir, feature_set, args.train_ratio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
