import os
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
    def _read_source_dataframe(self):
        if not os.path.exists(self.config.DATA_PATH):
            return None

        df = pd.read_csv(self.config.DATA_PATH)
        time_col = "time" if "time" in df.columns else "timestamp"

        if time_col not in df.columns:
            print("[!] Missing time column in CSV (expected 'time' or 'timestamp')")
            return None

        needed = [time_col, "open", "high", "low", "close", "volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[!] Missing columns in CSV: {missing}")
            return None

        out = df[needed].copy()
        if time_col != "time":
            out = out.rename(columns={time_col: "time"})

        out["time"] = pd.to_datetime(out["time"])
        out = out.sort_values("time").reset_index(drop=True)
        return out

    # =========================
    # LOAD CSV (warmup buffer)
    # =========================
    def load_data(self):
        df = self._read_source_dataframe()
        if df is None:
            print("[!] CSV not found:", self.config.DATA_PATH)
            return

        for _, row in df.iterrows():
            self.buffer.add_candle({
                "time": row["time"],
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
        source_df = self._read_source_dataframe()
        if source_df is not None:
            df = source_df
        else:
            df = self.buffer.get_data()

        if len(df) < self.config.MIN_DATA:
            print("[!] Not enough data to fit scaler")
            return

        raw_features = []

        max_fit_samples = 5000
        stride = max(1, len(df) // max_fit_samples)
        print(f"[*] Fitting scaler with stride={stride} on {len(df)} rows")

        temp_buffer = CandleBuffer(max_size=self.config.BUFFER_SIZE)
        for idx, row in df.iterrows():
            temp_buffer.add_candle({
                "time": row["time"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "is_closed": True,
            })

            if not temp_buffer.is_ready(self.config.MIN_DATA):
                continue

            if idx % stride != 0:
                continue

            sub_df = temp_buffer.get_data()
            f = self.preprocessor.compute_features(sub_df)
            if f is not None:
                raw_features.append(f)

        if len(raw_features) == 0:
            print("[!] No features to train scaler")
            return

        self.preprocessor.fit_scaler(raw_features)
        print("[*] Scaler fitted")

    # =========================
    # ITER HISTORICAL FEATURES
    # =========================
    def iter_historical_results(self):
        df = self._read_source_dataframe()
        if df is None:
            return

        temp_buffer = CandleBuffer(max_size=self.config.BUFFER_SIZE)

        for _, row in df.iterrows():
            temp_buffer.add_candle({
                "time": row["time"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "is_closed": True,
            })

            if not temp_buffer.is_ready(self.config.MIN_DATA):
                continue

            sub_df = temp_buffer.get_data()
            feature = self.preprocessor.compute_features(sub_df)
            if feature is None:
                continue

            feature_norm = self.preprocessor.normalize_features(feature)
            feature_norm = np.tanh(feature_norm / 3)

            result = {
                "time": row["time"],
                "feature_raw": feature,
                "feature_norm": feature_norm,
            }

            if self.tq is not None:
                tq_out = self.tq.quantize(feature_norm)
                result.update({
                    "tq_code": tq_out["code"],
                    "tq_indices": tq_out["indices"],
                    "tq_xhat": tq_out["x_hat"],
                    "tq_regime": tq_out["regime"],
                    "tq_score": tq_out["score"],
                    "tq_error": tq_out["error"],
                    "tq_confidence": tq_out["confidence"],
                })

            yield result

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
                "tq_xhat": tq_out["x_hat"],
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