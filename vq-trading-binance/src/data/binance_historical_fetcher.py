import aiohttp
import pandas as pd
import aiohttp

class BinanceHistoricalFetcher:
    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self, symbol):
        self.symbol = symbol

    async def get_historical_data(self, interval="1m", limit=200):
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(self.BASE_URL, params=params) as resp:
                data = await resp.json()

        candles = []

        for k in data:
            candles.append({
                "time": pd.to_datetime(k[0], unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "is_closed": bool(k[6]) 
            })

        return candles