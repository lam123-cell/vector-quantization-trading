import asyncio
import json
import websockets


class BinanceKlineStream:
    def __init__(self, symbol="btcusdt", interval="1m"):
        self.url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"

    async def start(self, callback):
        retry_seconds = 3

        while True:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
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
                            "is_closed": k['x']  # closed candle only
                        }

                        await callback(candle)

            except asyncio.CancelledError:
                print("[*] Stream task cancelled")
                raise
            except KeyboardInterrupt:
                print("[*] Stream interrupted by user")
                raise
            except Exception as e:
                # DNS/network hiccups are common in long-running websocket jobs.
                print(f"[!] Stream error: {e}. Reconnecting in {retry_seconds}s...")
                await asyncio.sleep(retry_seconds)