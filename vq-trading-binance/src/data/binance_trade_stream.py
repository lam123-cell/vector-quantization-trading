import asyncio
import json
import websockets
import aiohttp
import pandas as pd


class BinanceTradeStream:
    """
    Real-time trade stream from Binance (aggTrade)
    """

    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol.lower()

    async def start_stream(self, callback):
        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@aggTrade"

        while True:
            try:
                print(f"[*] Connecting to {url} ...")

                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20
                ) as ws:

                    print(f"[*] Connected to TRADE stream: {self.symbol}")

                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)

                        trade = {
                            "symbol": data["s"],
                            "price": float(data["p"]),
                            "volume": float(data["q"]),
                            "time": data["T"]
                        }

                        await callback(trade)

            except Exception as e:
                print(f"[!] Stream Error: {e}")
                print("[*] Reconnecting in 5 seconds...")
                await asyncio.sleep(5)


