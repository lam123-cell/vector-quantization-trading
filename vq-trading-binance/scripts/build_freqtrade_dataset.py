"""Build freqtrade_dataset.csv for baseline and turbo strategies.

Output schema targets:
- Baseline strategy input: OHLCV + n_*
- Turbo strategy input: OHLCV + tq_xhat_*

Core tq metadata is preserved for optional analysis.
"""

import pandas as pd
import os
import shutil
from pathlib import Path


def _normalize_time_series(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    ts_num = pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="ms", errors="coerce")
    return ts.fillna(ts_num)


def _resolve_dataset_columns(columns: list[str]) -> list[str]:
    n_cols = sorted([c for c in columns if c.startswith("n_")])
    tq_xhat_cols = sorted([c for c in columns if c.startswith("tq_xhat_")])
    tq_meta_cols = [c for c in ["tq_code", "tq_regime", "tq_score", "tq_error", "tq_confidence"] if c in columns]

    selected = ["time"] + n_cols + tq_xhat_cols + tq_meta_cols
    missing_minimum = []
    if not n_cols:
        missing_minimum.append("n_*")
    if not tq_xhat_cols:
        missing_minimum.append("tq_xhat_*")
    if missing_minimum:
        raise ValueError(f"dataset_master.csv missing required groups: {', '.join(missing_minimum)}")

    return selected

def build_freqtrade_dataset(
    dataset_file='vq-trading-binance/dataset_master.csv',
    raw_ohlcv_file='vq-trading-binance/btc_buffer.csv',
    output_file='vq-trading-binance/freqtrade_dataset.csv',
    copy_to_user_data=True,
):
    """
    Build freqtrade_dataset by merging OHLCV + selected feature groups from dataset_master.csv
    
    Args:
        dataset_file: Path to computed dataset_master.csv
        raw_ohlcv_file: Path to raw OHLCV data
        output_file: Output path for freqtrade_dataset.csv
        copy_to_user_data: Also copy to freqtrade_setup/user_data/freqtrade_dataset.csv
    
    Returns:
        pd.DataFrame: merged dataset
    """
    
    print("=" * 80)
    print("BUILD FREQTRADE_DATASET.CSV")
    print("=" * 80)
    
    # 1. Load datasets
    print("\n[1] Load data...")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Missing: {dataset_file}")
    if not os.path.exists(raw_ohlcv_file):
        raise FileNotFoundError(f"Missing: {raw_ohlcv_file}")
    
    all_dataset_cols = pd.read_csv(dataset_file, nrows=0).columns.tolist()
    selected_dataset_cols = _resolve_dataset_columns(all_dataset_cols)
    df_dataset = pd.read_csv(dataset_file, usecols=selected_dataset_cols)
    df_raw = pd.read_csv(raw_ohlcv_file)

    raw_time_col = 'time' if 'time' in df_raw.columns else 'timestamp'
    if raw_time_col not in df_raw.columns:
        raise ValueError("Raw OHLCV file must have 'time' or 'timestamp' column")
    
    print(f"    - Dataset: {len(df_dataset):,} rows x {len(df_dataset.columns)} cols")
    print(f"    - Raw OHLCV: {len(df_raw):,} rows x {len(df_raw.columns)} cols")
    
    # 2. Prepare merge - robust timestamp normalization
    print("\n[2] Prepare merge...")
    df_dataset["_time_key"] = _normalize_time_series(df_dataset["time"])
    df_raw["_time_key"] = _normalize_time_series(df_raw[raw_time_col])

    df_dataset = df_dataset.dropna(subset=["_time_key"])
    df_raw = df_raw.dropna(subset=["_time_key"])
    
    print(f"    - Dataset time range: {df_dataset['_time_key'].min()} to {df_dataset['_time_key'].max()}")
    print(f"    - Raw time range: {df_raw['_time_key'].min()} to {df_raw['_time_key'].max()}")
    
    # 3. Merge (inner join - keep only overlapping times)
    print("\n[3] Merge OHLCV + features...")
    df_merged = pd.merge(
        df_dataset,
        df_raw[["_time_key", "open", "high", "low", "close", "volume"]],
        on="_time_key",
        how='inner'
    )
    
    print(f"    - Merged result: {len(df_merged):,} rows ({len(df_dataset) - len(df_merged)} rows not matched)")
    
    # 4. Arrange columns - OHLCV first, then strategy-target features
    print("\n[4] Arrange columns...")
    n_cols = sorted([c for c in df_merged.columns if c.startswith("n_")])
    tq_xhat_cols = sorted([c for c in df_merged.columns if c.startswith("tq_xhat_")])
    tq_meta_cols = [c for c in ["tq_code", "tq_regime", "tq_score", "tq_error", "tq_confidence"] if c in df_merged.columns]

    df_merged["time"] = df_merged["_time_key"].dt.strftime("%Y-%m-%d %H:%M:%S")
    col_order = ["time", "open", "high", "low", "close", "volume"] + n_cols + tq_xhat_cols + tq_meta_cols
    
    df_freqtrade = df_merged[col_order].reset_index(drop=True)
    
    print(f"    - Total columns: {len(df_freqtrade.columns)}")
    print(f"    - n_* columns: {len(n_cols)}")
    print(f"    - tq_xhat_* columns: {len(tq_xhat_cols)}")
    print(f"    - Column order:")
    for i, col in enumerate(df_freqtrade.columns[:10]):
        print(f"       [{i:2}] {col}")
    print(f"       ... ({len(df_freqtrade.columns) - 10} more columns)")
    
    # 5. Quality check
    print(f"\n[5] Quality check...")
    null_count = df_freqtrade.isnull().sum().sum()
    dup_time = df_freqtrade['time'].duplicated().sum()
    file_size = df_freqtrade.memory_usage(deep=True).sum() / 1024**2
    
    print(f"    - Total NaN: {null_count}")
    print(f"    - Duplicate timestamps: {dup_time}")
    print(f"    - Memory size: {file_size:.2f} MB")
    
    if null_count > 0:
        print(f"    WARNING: Found {null_count} NaN values!")
    if dup_time > 0:
        print(f"    WARNING: Found {dup_time} duplicate timestamps!")
    
    # 6. Sample data
    print(f"\n[6] Sample (first 3 rows):")
    cols_sample = [
        'time',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'n_log_return',
        'n_rsi',
        'tq_xhat_log_return',
        'tq_xhat_rsi',
    ]
    cols_sample = [c for c in cols_sample if c in df_freqtrade.columns]
    print(df_freqtrade.head(3)[cols_sample].to_string())
    
    # 7. Save
    print(f"\n[7] Save...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_freqtrade.to_csv(output_file, index=False)
    output_size = os.path.getsize(output_file) / 1024**2
    
    print(f"    - Saved: {output_file}")
    print(f"    - File size: {output_size:.2f} MB")

    if copy_to_user_data:
        output_path = Path(output_file).resolve()
        user_data_target = output_path.parent / "freqtrade_setup" / "user_data" / "freqtrade_dataset.csv"
        user_data_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_path, user_data_target)
        print(f"    - Copied to: {user_data_target}")
    
    print("\n" + "=" * 80)
    print("DONE! freqtrade_dataset.csv is ready for Freqtrade")
    print("=" * 80)
    
    return df_freqtrade


if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # Get script directory
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    
    # Build with default paths
    try:
        df = build_freqtrade_dataset(
            dataset_file=str(root_dir / 'dataset_master.csv'),
            raw_ohlcv_file=str(root_dir / 'btc_buffer.csv'),
            output_file=str(root_dir / 'freqtrade_dataset.csv'),
            copy_to_user_data=True,
        )
        print("\nSuccess! You can now use freqtrade_dataset.csv with Freqtrade")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
