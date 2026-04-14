import asyncio
import atexit

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

    print("=== START STREAM ===")

    stream = BinanceKlineStream(config.SYMBOL, config.TIMEFRAME)
    await stream.start(handle_kline)


# 🔥 đảm bảo không mất data khi crash
atexit.register(writer.flush_all)


if __name__ == "__main__":
    asyncio.run(main())