import aiohttp
import pandas as pd
class BinanceHistoricalFetcher:
    """
    Load historical kline to warm-up buffer
    """

    def __init__(self, symbol="BTCUSDT", interval="1m"):
        self.symbol = symbol

    async def get_historical_data(self, limit=100):
        url = f"https://api.binance.com/api/v3/klines?symbol={self.symbol}&interval={self.interval}&limit={limit}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()

                    candles = []
                    for k in data:
                        candle = {
                            "time": pd.to_datetime(k[0], unit="ms"),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                            "is_closed": True
                        }
                        candles.append(candle)

                    return candles

        except Exception as e:
            print(f"[!] Error fetching historical data: {e}")
            return []