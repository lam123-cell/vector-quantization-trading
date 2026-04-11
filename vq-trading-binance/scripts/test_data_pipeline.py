import asyncio
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.binance_trade_stream import BinanceTradeStream
from src.data.binance_historical_fetcher import BinanceHistoricalFetcher
from src.data.candle_builder import CandleBuilder
from src.data.candle_buffer import CandleBuffer
from src.data.feature_engineer import FeatureEngineer


DATA_PATH = "data/btc_buffer.csv"

buffer = CandleBuffer(max_size=100)
builder = CandleBuilder(interval_ms=5000)
preprocessor = FeatureEngineer()


# =========================
# LOAD CSV OR FETCH
# =========================
async def initialize():
    fetcher = BinanceHistoricalFetcher("BTCUSDT")

    if os.path.exists(DATA_PATH):
        print("[*] Loading from CSV...")
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
        print("[*] No CSV → Fetching from Binance...")
        history = await fetcher.get_historical_data(limit=100)

        # ✅ CHECK Ở ĐÂY
        if history and len(history) > 0:
            for candle in history:
                buffer.add_candle(candle)

            save_to_csv()
        else:
            print("[!] Fetch failed → buffer is empty, skip saving CSV")

    print(f"[*] Buffer size: {buffer.size()}")
    if buffer.size() == 0:
        print("[!] WARNING: Buffer is empty → features will not work")

# =========================
# SAVE CSV
# =========================
def save_to_csv():
    if buffer.size() == 0:
        print("[!] Buffer empty → skip saving CSV")
        return

    df = buffer.get_data()
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print("[*] Saved buffer to CSV")


# =========================
# HANDLE STREAM
# =========================
async def handle_trade(trade):
    candle = builder.update(trade)

    if candle:
        buffer.add_candle(candle)

        print("\n=== NEW CANDLE ===")
        print(candle)

        # =========================
        # FEATURE ENGINEERING
        # =========================
        if buffer.is_ready(35):   # 🔥 nâng lên 35 cho MACD
            df = buffer.get_data()

            try:
                feature = preprocessor.compute_features(df)

                if feature is not None:
                    print("Feature:", feature)
                else:
                    print("[DEBUG] Feature = None (chưa đủ dữ liệu sạch)")

            except Exception as e:
                print("[!] Feature error:", e)

        else:
            print(f"[DEBUG] Not enough data: {buffer.size()}/35")

        # =========================
        # SAVE CSV
        # =========================
        if buffer.size() % 10 == 0:
            save_to_csv()


# =========================
# MAIN
# =========================
async def main():
    await initialize()

    stream = BinanceTradeStream("BTCUSDT")
    await stream.start_stream(handle_trade)


if __name__ == "__main__":
    asyncio.run(main())