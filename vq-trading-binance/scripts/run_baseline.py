import asyncio
import os
import pandas as pd
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.candle_buffer import CandleBuffer
from src.feature.feature_engineer import FeatureEngineer
from src.data.binance_kline_stream import BinanceKlineStream

np.set_printoptions(suppress=True, precision=4)

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "src", "data", "btc_buffer.csv")

# =========================
# GLOBAL OBJECT
# =========================
buffer = CandleBuffer(max_size=100)
preprocessor = FeatureEngineer()

# =========================
# INITIALIZE
# =========================

def pretty_named(feature):
    names = [
        "log_return",
        "return_5",
        "log_volume",
        "candle_body",
        "rsi",
        "macd",
        "macd_signal",
        "volatility",
        "atr"
    ]
    return {k: round(v, 4) for k, v in zip(names, feature)}

async def initialize():
    print("\n========== INITIALIZE ==========")

    # ===== LOAD CSV =====
    if os.path.exists(DATA_PATH):
        print("[*] Loading data from CSV...")
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
    else:
        print("[!] CSV not found")

    print(f"[*] Buffer size after load: {buffer.size()}")

    # ===== CHECK DATA =====
    if buffer.size() < 50:
        print("[!] Not enough data to train scaler (<50)")
        return

    print("[*] Building features for scaler training...")

    raw_features = []
    df = buffer.get_data()

    # ===== BUILD FEATURES =====
    for i in range(35, len(df)):
        sub_df = df.iloc[:i]
        f = preprocessor.compute_features(sub_df)

        if f is not None:
            raw_features.append(f)

    print(f"[*] Collected {len(raw_features)} feature vectors")

    if len(raw_features) == 0:
        print("[!] No valid features → cannot fit scaler")
        return

    # ===== FIT SCALER =====
    preprocessor.fit_scaler(raw_features)

    print("\n📊 SCALER STATS")
    print("Mean:", pretty_named(preprocessor.scaler.mean_))
    print("Std: ", pretty_named(preprocessor.scaler.scale_))

    print("========== INIT DONE ==========\n")


# =========================
# SAVE CSV
# =========================
def save_to_csv():
    if buffer.size() == 0:
        print("[!] Buffer empty → skip saving CSV")
        return

    df = buffer.get_data()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"[*] Saved CSV (size={buffer.size()})")


# =========================
# STREAM HANDLER
# =========================
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

    print("\n=== NEW KLINE ===")
    print(f"Buffer size: {buffer.size()}")

    if not buffer.is_ready(50):
        print(f"[DEBUG] Not enough data: {buffer.size()}/50")
        return

    df = buffer.get_data()
    feature = preprocessor.compute_features(df)

    if feature is None:
        print("[DEBUG] Feature = None")
        return

    # ===== CHECK DIMENSION =====
    if len(feature) != 9:
        print("[ERROR] Feature dim mismatch:", len(feature))
        return

    print("[DEBUG] Scaler fitted:", preprocessor.is_fitted)

    feature_norm = preprocessor.normalize_features(feature)

    # ===== DEBUG PRINT =====
    print("Raw feature:", pretty_named(feature))
    print("Norm feature:", pretty_named(feature_norm))

    print("[DEBUG] Feature shape:", len(feature))
    print("[DEBUG] Norm shape:", len(feature_norm))

    if buffer.size() % 10 == 0:
        save_to_csv()

# =========================
# MAIN
# =========================
async def main():
    await initialize()

    print("[*] Starting stream...")
    print("Buffer after init:", buffer.size())

    stream = BinanceKlineStream("btcusdt", "1m")
    await stream.start(handle_kline)


if __name__ == "__main__":
    asyncio.run(main())