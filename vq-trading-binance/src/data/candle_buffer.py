import pandas as pd


class CandleBuffer:
    """
    Sliding window buffer for candles (streaming-friendly)
    """

    def __init__(self, max_size=100):
        self.max_size = max_size
        self.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        self.df = pd.DataFrame(columns=self.columns)

    def add_candle(self, candle):
        if candle is None or not candle.get("is_closed", False):
            return

        # chỉ lấy đúng columns
        # Normalize time: accept ms timestamps or ISO-like strings and store as UTC
        t = candle["time"]
        if isinstance(t, (int, float)):
            time_ts = pd.to_datetime(t, unit="ms", utc=True)
        else:
            time_ts = pd.to_datetime(t, utc=True)

        row = {
            "time": time_ts,
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"]
        }
        
        # append nhanh hơn concat
        self.df.loc[len(self.df)] = row

        # giữ size cố định
        if len(self.df) > self.max_size:
            self.df = self.df.iloc[-self.max_size:].reset_index(drop=True)

    def get_data(self):
        return self.df.copy()

    def is_ready(self, min_required=30):
        return len(self.df) >= min_required

    def size(self):
        return len(self.df)

    def clear(self):
        self.df = pd.DataFrame(columns=self.columns)