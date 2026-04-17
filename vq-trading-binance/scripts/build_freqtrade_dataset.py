"""
Build freqtrade_dataset.csv by merging OHLCV + all features from dataset.csv

This script:
1. Loads raw OHLCV data from data/raw/
2. Loads computed features from dataset.csv
3. Merges them (inner join on timestamp)
4. Outputs freqtrade_dataset.csv with OHLCV + all indicators + quantization results

File: freqtrade_dataset.csv structure:
  - time (string: YYYY-MM-DD HH:MM:SS)
  - open, high, low, close, volume (OHLCV for Freqtrade)
  - f_* columns (9 raw features)
  - n_* columns (9 normalized features)
  - tq_* columns (quantization results)
"""

import pandas as pd
import os
from pathlib import Path

def build_freqtrade_dataset(
    dataset_file='vq-trading-binance/dataset.csv',
    raw_ohlcv_file='vq-trading-binance/btc_buffer.csv',
    output_file='vq-trading-binance/freqtrade_dataset.csv'
):
    """
    Build freqtrade_dataset by merging OHLCV + dataset.csv
    
    Args:
        dataset_file: Path to computed dataset.csv
        raw_ohlcv_file: Path to raw OHLCV data
        output_file: Output path for freqtrade_dataset.csv
    
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
    
    df_dataset = pd.read_csv(dataset_file)
    df_raw = pd.read_csv(raw_ohlcv_file)

    raw_time_col = 'time' if 'time' in df_raw.columns else 'timestamp'
    if raw_time_col not in df_raw.columns:
        raise ValueError("Raw OHLCV file must have 'time' or 'timestamp' column")
    
    print(f"    - Dataset: {len(df_dataset):,} rows x {len(df_dataset.columns)} cols")
    print(f"    - Raw OHLCV: {len(df_raw):,} rows x {len(df_raw.columns)} cols")
    
    # 2. Prepare merge - create merge keys
    print("\n[2] Prepare merge...")
    df_dataset['_time_key'] = df_dataset['time']
    df_raw['_raw_time'] = df_raw[raw_time_col]
    
    print(f"    - Dataset time range: {df_dataset['time'].min()} to {df_dataset['time'].max()}")
    print(f"    - Raw time range: {df_raw[raw_time_col].min()} to {df_raw[raw_time_col].max()}")
    
    # 3. Merge (inner join - keep only overlapping times)
    print("\n[3] Merge OHLCV + features...")
    df_merged = pd.merge(
        df_dataset,
        df_raw[['_raw_time', 'open', 'high', 'low', 'close', 'volume']],
        left_on='_time_key',
        right_on='_raw_time',
        how='inner'
    )
    
    print(f"    - Merged result: {len(df_merged):,} rows ({len(df_dataset) - len(df_merged)} rows not matched)")
    
    # 4. Arrange columns - OHLCV first (right after time)
    print("\n[4] Arrange columns...")
    col_order = ['time', 'open', 'high', 'low', 'close', 'volume']
    remaining_cols = [c for c in df_merged.columns 
                      if c not in col_order and c not in ['_time_key', '_raw_time']]
    col_order.extend(remaining_cols)
    
    df_freqtrade = df_merged[col_order].reset_index(drop=True)
    
    print(f"    - Total columns: {len(df_freqtrade.columns)}")
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
    cols_sample = ['time', 'open', 'high', 'low', 'close', 'volume', 'f_log_return', 'tq_regime', 'tq_score']
    cols_sample = [c for c in cols_sample if c in df_freqtrade.columns]
    print(df_freqtrade.head(3)[cols_sample].to_string())
    
    # 7. Save
    print(f"\n[7] Save...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_freqtrade.to_csv(output_file, index=False)
    output_size = os.path.getsize(output_file) / 1024**2
    
    print(f"    - Saved: {output_file}")
    print(f"    - File size: {output_size:.2f} MB")
    
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
            dataset_file=str(root_dir / 'dataset.csv'),
            raw_ohlcv_file=str(root_dir / 'btc_buffer.csv'),
            output_file=str(root_dir / 'freqtrade_dataset.csv')
        )
        print("\nSuccess! You can now use freqtrade_dataset.csv with Freqtrade")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
