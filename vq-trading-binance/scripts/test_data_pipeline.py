import asyncio
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.binance_historical_fetcher import BinanceHistoricalFetcher
from src.data.candle_buffer import CandleBuffer
from src.data.feature_engineer import FeatureEngineer
from src.data.binance_kline_stream import BinanceKlineStream

DATA_PATH = "data/btc_buffer.csv"

buffer = CandleBuffer(max_size=100)
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
    print(candle)

    if buffer.is_ready(35):
        df = buffer.get_data()
        feature = preprocessor.compute_features(df)

        print("Feature:", feature)
    else:
        print(f"[DEBUG] Not enough data: {buffer.size()}/35")
        
# =========================
# MAIN
# =========================

async def main():
    stream = BinanceKlineStream("btcusdt", "1m")

    await stream.start(handle_kline)


if __name__ == "__main__":
    asyncio.run(main())
