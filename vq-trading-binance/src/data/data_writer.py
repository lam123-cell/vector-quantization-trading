import os
import pandas as pd


class DataWriter:
    def __init__(self, candle_path, dataset_path, batch_size=50):
        self.candle_path = candle_path
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        self._candle_buffer = []
        self._dataset_buffer = []

        # =========================
        # FEATURE NAMES (🔥 readable)
        # =========================
        self.feature_names = [
            "log_return",
            "return_5",
            "log_volume",
            "candle_body",
            "rsi",
            "macd",
            "macd_signal",
            "volatility",
            "atr"
        ]

        # =========================
        # SCHEMA
        # =========================
        self.candle_columns = [
            "time", "open", "high", "low", "close", "volume"
        ]

        self.dataset_columns = (
            ["time"] +
            [f"f_{name}" for name in self.feature_names] +
            [f"n_{name}" for name in self.feature_names] +
            ["tq_code", "tq_regime", "tq_score", "tq_error", "tq_confidence"]
        )

    # =========================
    # 🔥 TIME NORMALIZER (QUAN TRỌNG NHẤT)
    # =========================
    def _normalize_time(self, t):
        try:
            # nếu là timestamp (ms)
            if isinstance(t, (int, float)):
                return pd.to_datetime(t, unit="ms")

            # nếu là string / datetime
            return pd.to_datetime(t)
        except Exception:
            return None

    # =========================
    # ADD CANDLE
    # =========================
    def add_candle(self, candle):
        if not candle.get("is_closed", True):
            return

        row = {
            "time": self._normalize_time(candle["time"]),
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "volume": float(candle["volume"]),
        }

        self._candle_buffer.append(row)

        if len(self._candle_buffer) >= self.batch_size:
            self._flush_candles()

    # =========================
    # ADD FEATURE
    # =========================
    def add_feature(self, result):
        if result is None:
            return

        row = {
            "time": self._normalize_time(result["time"])
        }

        # =========================
        # RAW FEATURE
        # =========================
        for name, value in zip(self.feature_names, result["feature_raw"]):
            row[f"f_{name}"] = float(value)

        # =========================
        # NORMALIZED FEATURE
        # =========================
        for name, value in zip(self.feature_names, result["feature_norm"]):
            row[f"n_{name}"] = float(value)

        # =========================
        # TURBO QUANT
        # =========================
        row["tq_code"] = result.get("tq_code")
        row["tq_regime"] = result.get("tq_regime")
        row["tq_score"] = result.get("tq_score")
        row["tq_error"] = result.get("tq_error")
        row["tq_confidence"] = result.get("tq_confidence")

        self._dataset_buffer.append(row)

        if len(self._dataset_buffer) >= self.batch_size:
            self._flush_dataset()

    # =========================
    # SAVE CANDLES
    # =========================
    def _flush_candles(self):
        if not self._candle_buffer:
            return

        df = pd.DataFrame(self._candle_buffer)

        df = df[self.candle_columns]

        os.makedirs(os.path.dirname(self.candle_path), exist_ok=True)

        df.to_csv(
            self.candle_path,
            mode="a",
            header=not os.path.exists(self.candle_path),
            index=False
        )

        self._candle_buffer.clear()
        print(f"[*] Flushed candles ({len(df)})")

    # =========================
    # SAVE DATASET
    # =========================
    def _flush_dataset(self):
        if not self._dataset_buffer:
            return

        df = pd.DataFrame(self._dataset_buffer)

        # đảm bảo đủ cột
        for col in self.dataset_columns:
            if col not in df:
                df[col] = None

        df = df[self.dataset_columns]

        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

        df.to_csv(
            self.dataset_path,
            mode="a",
            header=not os.path.exists(self.dataset_path),
            index=False
        )

        self._dataset_buffer.clear()
        print(f"[*] Flushed dataset ({len(df)})")

    # =========================
    # FORCE SAVE
    # =========================
    def flush_all(self):
        self._flush_candles()
        self._flush_dataset()