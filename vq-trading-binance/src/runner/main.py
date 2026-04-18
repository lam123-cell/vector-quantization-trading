import asyncio
import atexit
import os
import pandas as pd

from src.runner.config import Config
from src.runner.pipeline import Pipeline
from src.data.binance_kline_stream import BinanceKlineStream
from src.data.data_writer import DataWriter


config = Config()

writer = DataWriter(
    candle_path=config.DATA_PATH,
    dataset_path=config.DATASET_PATH,
    batch_size=config.SAVE_INTERVAL
)

pipeline = Pipeline(config)


def _load_existing_dataset_times(dataset_path):
    if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) == 0:
        return set()

    try:
        df = pd.read_csv(dataset_path, usecols=["time"])
        if "time" not in df.columns:
            return set()
        return set(pd.to_datetime(df["time"], utc=True).astype(str).tolist())
    except Exception:
        return set()


def _load_dataset_max_time(dataset_path):
    if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) == 0:
        return None

    try:
        df = pd.read_csv(dataset_path, usecols=["time"])
        if "time" not in df.columns or len(df) == 0:
            return None
        t = pd.to_datetime(df["time"], errors="coerce", utc=True).dropna()
        return t.max() if len(t) > 0 else None
    except Exception:
        return None


def backfill_historical_dataset():
    source_df = pipeline._read_source_dataframe()
    if source_df is None or len(source_df) == 0:
        print("[*] No historical source to backfill")
        return

    source_max_time = source_df["time"].max()
    dataset_max_time = _load_dataset_max_time(config.DATASET_PATH)

    if dataset_max_time is not None and dataset_max_time >= source_max_time:
        print(f"[*] Historical already covered until {dataset_max_time}. Skip backfill.")
        return

    existing_times = _load_existing_dataset_times(config.DATASET_PATH)
    added = 0
    scanned = 0

    for result in pipeline.iter_historical_results(min_time_exclusive=dataset_max_time):
        scanned += 1
        t_key = str(pd.to_datetime(result["time"], utc=True))
        if t_key in existing_times:
            if scanned % 5000 == 0:
                print(f"[*] Backfill progress: scanned={scanned}, added={added}")
            continue

        writer.add_feature(result)
        existing_times.add(t_key)
        added += 1

        if scanned % 5000 == 0:
            print(f"[*] Backfill progress: scanned={scanned}, added={added}")

    writer.flush_all()
    print(f"[*] Historical backfill added {added} rows to dataset")


def pretty(feature):
    names = [
        "log_return", "return_5", "log_volume", "candle_body",
        "rsi", "macd", "macd_signal", "volatility", "atr"
    ]
    return {k: round(v, 4) for k, v in zip(names, feature)}


async def handle_kline(candle):
    # ❗ CHỈ xử lý candle đóng
    if not candle.get("is_closed", True):
        return

    closed_candle = {
        "time": candle["time"],
        "open": candle["open"],
        "high": candle["high"],
        "low": candle["low"],
        "close": candle["close"],
        "volume": candle["volume"],
        "is_closed": True
    }

    # =========================
    # PIPELINE
    # =========================
    result = pipeline.add_candle(closed_candle)

    # =========================
    # SAVE CANDLE
    # =========================
    writer.add_candle(closed_candle)

    if result is None:
        return

    # 🔥 FIX: add time vào result
    result["time"] = closed_candle["time"]

    print("\n=== NEW FEATURE ===")
    print("Raw:", pretty(result["feature_raw"]))
    print("Norm:", pretty(result["feature_norm"]))

    # =========================
    # SAVE DATASET
    # =========================
    writer.add_feature(result)

    # =========================
    # TurboQuant
    # =========================
    if config.USE_TURBO:
        print("TQ code:", result["tq_code"])
        print("TQ regime:", result["tq_regime"])
        print("TQ score:", round(result["tq_score"], 4))
        print("TQ error:", round(result["tq_error"], 4))
        print("TQ confidence:", round(result["tq_confidence"], 4))


async def main():
    print("=== INIT ===")

    pipeline.load_data()
    pipeline.fit_scaler()
    backfill_historical_dataset()

    print("=== START STREAM ===")

    stream = BinanceKlineStream(config.SYMBOL, config.TIMEFRAME)
    await stream.start(handle_kline)


# 🔥 đảm bảo không mất data khi crash
atexit.register(writer.flush_all)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] Stopped by user (Ctrl+C)")