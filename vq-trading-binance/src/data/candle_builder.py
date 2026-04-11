import pandas as pd

class CandleBuilder:
    """
    Build OHLCV candles from trade stream
    """

    def __init__(self, interval_ms=1000):  # default 1s candle
        self.interval = interval_ms
        self.current_candle = None

    def update(self, trade):
        t = trade["time"]
        price = trade["price"]
        volume = trade["volume"]

        bucket = t - (t % self.interval)

        # First candle
        if self.current_candle is None:
            self.current_candle = {
                "time": pd.to_datetime(bucket, unit="ms"),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
                "is_closed": False
            }
            return None

        # Same candle
        if bucket == int(self.current_candle["time"].timestamp() * 1000):
            self.current_candle["high"] = max(self.current_candle["high"], price)
            self.current_candle["low"] = min(self.current_candle["low"], price)
            self.current_candle["close"] = price
            self.current_candle["volume"] += volume
            return None

        # New candle → close old one
        finished = self.current_candle.copy()
        finished["is_closed"] = True

        # Create new candle
        self.current_candle = {
            "time": pd.to_datetime(bucket, unit="ms"),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume,
            "is_closed": False
        }

        return finished
