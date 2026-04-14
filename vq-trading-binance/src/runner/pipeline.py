import os
from time import time
import pandas as pd
import numpy as np

from src.data.candle_buffer import CandleBuffer
from src.feature.feature_engineer import FeatureEngineer
from src.quantization.turboquant_core import TurboQuant


class Pipeline:
    def __init__(self, config):
        self.config = config

        self.buffer = CandleBuffer(max_size=config.BUFFER_SIZE)
        self.preprocessor = FeatureEngineer()

        # TurboQuant optional
        self.tq = None
        if config.USE_TURBO:
            self.tq = TurboQuant(
                feature_dim=config.FEATURE_DIM,
                levels=config.TQ_LEVELS,
                value_range=config.TQ_RANGE
            )

    # =========================
    # LOAD CSV (warmup buffer)
    # =========================
    def load_data(self):
        if not os.path.exists(self.config.DATA_PATH):
            print("[!] CSV not found:", self.config.DATA_PATH)
            return

        df = pd.read_csv(self.config.DATA_PATH)

        for _, row in df.iterrows():
            self.buffer.add_candle({
                "time": pd.to_datetime(row["time"]),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "is_closed": True
            })

        print(f"[*] Loaded buffer size: {self.buffer.size()}")

    # =========================
    # FIT SCALER
    # =========================
    def fit_scaler(self):
        if self.buffer.size() < self.config.MIN_DATA:
            print("[!] Not enough data to fit scaler")
            return

        df = self.buffer.get_data()
        raw_features = []

        for i in range(35, len(df)):
            sub_df = df.iloc[:i]
            f = self.preprocessor.compute_features(sub_df)

            if f is not None:
                raw_features.append(f)

        if len(raw_features) == 0:
            print("[!] No features to train scaler")
            return

        self.preprocessor.fit_scaler(raw_features)
        print("[*] Scaler fitted")

    # =========================
    # PROCESS ONE STEP
    # =========================
    def process(self):
        if not self.buffer.is_ready(self.config.MIN_DATA):
            return None

        df = self.buffer.get_data()
        feature = self.preprocessor.compute_features(df)

        if feature is None:
            return None

        # =========================
        # NORMALIZATION
        # =========================
        feature_norm = self.preprocessor.normalize_features(feature)

        # 🔥 best practice (ổn định hơn clip)
        feature_norm = np.tanh(feature_norm / 3)

        result = {
            "time": df.iloc[-1]["time"],
            "feature_raw": feature,
            "feature_norm": feature_norm,
        }

        # =========================
        # TurboQuant
        # =========================
        if self.tq is not None:
            tq_out = self.tq.quantize(feature_norm)

            result.update({
                "tq_code": tq_out["code"],
                "tq_indices": tq_out["indices"],
                "tq_regime": tq_out["regime"],
                "tq_score": tq_out["score"],
                "tq_error": tq_out["error"],
                "tq_confidence": tq_out["confidence"],
            })

        return result

    # =========================
    # ADD NEW CANDLE
    # =========================
    def add_candle(self, candle):
        if not candle["is_closed"]:
            return None

        self.buffer.add_candle(candle)
        return self.process()