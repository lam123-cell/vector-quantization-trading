"""
Export freqtrade_dataset.csv to Freqtrade OHLCV CSV format.

Output columns:
    date,open,high,low,close,volume

Default output:
    vq-trading-binance/freqtrade_setup/user_data/data/binance/BTC_USDT-1m.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _pair_to_filename(pair: str) -> str:
    # BTC/USDT -> BTC_USDT
    return pair.replace("/", "_").upper()


def export_ohlcv_csv(input_csv: Path, output_csv: Path) -> int:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[required].copy()
    out = out.rename(columns={"time": "date"})

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    before = len(out)
    out = out.dropna(subset=["date"])
    dropped = before - len(out)
    if dropped > 0:
        print(f"[!] Dropped {dropped} rows with invalid date")

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    before_num = len(out)
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    dropped_num = before_num - len(out)
    if dropped_num > 0:
        print(f"[!] Dropped {dropped_num} rows with invalid OHLCV")

    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, date_format="%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print("EXPORT FREQTRADE OHLCV CSV")
    print("=" * 80)
    print(f"Input : {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Rows  : {len(out):,}")
    if len(out) > 0:
        print("First :", out.iloc[0].to_dict())
        print("Last  :", out.iloc[-1].to_dict())
    print("=" * 80)

    return len(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export OHLCV CSV for Freqtrade")
    parser.add_argument(
        "--input",
        default="vq-trading-binance/freqtrade_dataset.csv",
        help="Input CSV path (default: vq-trading-binance/freqtrade_dataset.csv)",
    )
    parser.add_argument(
        "--pair",
        default="BTC/USDT",
        help="Pair symbol for output filename (default: BTC/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        default="1m",
        help="Timeframe suffix for output filename (default: 1m)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Explicit output CSV path",
    )

    args = parser.parse_args()

    input_csv = Path(args.input)
    if args.output:
        output_csv = Path(args.output)
    else:
        pair_file = _pair_to_filename(args.pair)
        output_csv = Path(
            f"vq-trading-binance/freqtrade_setup/user_data/data/binance/{pair_file}-{args.timeframe}.csv"
        )

    export_ohlcv_csv(input_csv=input_csv, output_csv=output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
