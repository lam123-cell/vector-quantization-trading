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
        row = {col: candle[col] for col in self.columns}

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