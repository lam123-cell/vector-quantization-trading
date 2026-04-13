import asyncio
import json
import websockets

class BinanceKlineStream:
    def __init__(self, symbol="btcusdt", interval="1m"):
        self.url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"

    async def start(self, callback):
        async with websockets.connect(self.url) as ws:
            print("[*] Connected to Binance Kline Stream")

            while True:
                msg = await ws.recv()
                data = json.loads(msg)

                k = data['k']

                candle = {
                    "time": k['t'],
                    "open": float(k['o']),
                    "high": float(k['h']),
                    "low": float(k['l']),
                    "close": float(k['c']),
                    "volume": float(k['v']),
                    "is_closed": k['x']  # 🔥 quan trọng
                }

                await callback(candle)