import asyncio
import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.candle_buffer import CandleBuffer
from src.feature.feature_engineer import FeatureEngineer
from src.data.binance_kline_stream import BinanceKlineStream
from src.quantization.turbo_quant import TurboQuant

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "src", "data", "btc_buffer.csv")

buffer = CandleBuffer(max_size=100)
preprocessor = FeatureEngineer()
tq = TurboQuant(n_clusters=8, dim=5)


async def initialize():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

        for _, row in df.iterrows():
            buffer.add_candle({
                "time": pd.to_datetime(row["time"]),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "is_closed": True
            })

    print("[*] Buffer:", buffer.size())

    if buffer.size() < 50:
        return

    raw_features = []
    df = buffer.get_data()

    for i in range(35, len(df)):
        f = preprocessor.compute_features(df.iloc[:i])
        if f is not None:
            raw_features.append(f)

    if len(raw_features) < 20:
        return

    # fit scaler
    preprocessor.fit_scaler(raw_features)

    # normalize
    features = [preprocessor.normalize_features(f) for f in raw_features]

    # init turbo quant
    tq.initialize(features)


async def handle_kline(candle):
    if not candle["is_closed"]:
        return

    buffer.add_candle({
        "time": pd.to_datetime(candle["time"], unit="ms"),
        "open": candle["open"],
        "high": candle["high"],
        "low": candle["low"],
        "close": candle["close"],
        "volume": candle["volume"],
        "is_closed": True
    })

    print("\n=== TURBO QUANT ===")

    if buffer.is_ready(50):
        df = buffer.get_data()
        feature = preprocessor.compute_features(df)

        if feature is None:
            return

        feature_norm = preprocessor.normalize_features(feature)

        if not tq.is_initialized:
            print("[DEBUG] TQ not ready")
            return

        state = tq.encode(feature_norm)

        print("Feature:", feature_norm)
        print("State:", state)


async def main():
    await initialize()
    stream = BinanceKlineStream("btcusdt", "1m")
    await stream.start(handle_kline)


if __name__ == "__main__":
    asyncio.run(main())