import os
import numpy as np
import pandas as pd
from src.runner.config import Config

class DatasetBuilder:
    def __init__(
        self,
        master_path,
        output_dir="datasets",
        seq_len=50,
        horizon=10,
        threshold=0.001
    ):
        self.master_path = master_path
        self.output_dir = output_dir
        self.seq_len = seq_len
        self.horizon = horizon
        self.threshold = threshold

        self.lstm_dir = os.path.join(output_dir, "lstm")
        os.makedirs(self.lstm_dir, exist_ok=True)

        self.df = None

    # =========================
    # LOAD
    # =========================
    def load(self):
        df = pd.read_csv(self.master_path)

        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)

        self.df = df
        print(f"[*] Loaded: {len(df)} rows")

    # =========================
    # LABEL (NO LEAK + TRADING)
    # =========================
    def add_label(self):
        df = self.df.copy()

        # future return
        df["future_return"] = (
            df["close"].shift(-self.horizon) / df["close"] - 1
        )

        th = self.threshold

        def label_fn(x):
            if pd.isna(x):
                return np.nan
            if x > th:
                return 2  # BUY
            elif x < -th:
                return 0  # SELL
            else:
                return 1  # HOLD

        df["label"] = df["future_return"].apply(label_fn)

        self.df = df

        print(f"[*] Label done (threshold={th}, horizon={self.horizon})")

    # =========================
    # CLEAN
    # =========================
    def clean(self):
        df = self.df

        base_cols = [c for c in df.columns if c.startswith("n_")]
        tq_cols = [c for c in df.columns if c.startswith("tq_xhat_")]

        required_cols = ["label"] + base_cols + tq_cols

        before = len(df)

        df = df.dropna(subset=required_cols).reset_index(drop=True)

        self.df = df

        print(f"[*] Clean: {before} → {len(df)}")

    # =========================
    # DEBUG LABEL DIST
    # =========================
    def debug_labels(self):
        print("\n===== LABEL DIST =====")

        counts = self.df["label"].value_counts().sort_index()
        total = len(self.df)

        for k in [0, 1, 2]:
            v = counts.get(k, 0)
            print(f"{k}: {v} ({v/total:.4f})")

        print("======================\n")

    # =========================
    # BUILD SEQUENCE (NO LEAK)
    # =========================
    def _build_seq(self, features, labels):
        X, y = [], []

        for i in range(len(features) - self.seq_len - self.horizon):
            # input: past only
            X.append(features[i:i + self.seq_len])

            # label: future AFTER sequence
            y.append(labels[i + self.seq_len])

        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64)
        )

    # =========================
    # BUILD LSTM DATA
    # =========================
    def build_lstm(self):
        df = self.df

        print(f"[*] Build LSTM seq_len={self.seq_len}")

        labels = df["label"].values.astype(np.int64)

        # ========= BASELINE =========
        base_cols = [c for c in df.columns if c.startswith("n_")]

        X_base, y_base = self._build_seq(
            df[base_cols].values,
            labels
        )

        np.save(f"{self.lstm_dir}/lstm_baseline_X.npy", X_base)
        np.save(f"{self.lstm_dir}/lstm_baseline_y.npy", y_base)

        print(f"[+] baseline: {X_base.shape}")

        # ========= TQ =========
        tq_cols = [c for c in df.columns if c.startswith("tq_xhat_")]

        X_tq, y_tq = self._build_seq(
            df[tq_cols].values,
            labels
        )

        np.save(f"{self.lstm_dir}/lstm_tq_X.npy", X_tq)
        np.save(f"{self.lstm_dir}/lstm_tq_y.npy", y_tq)

        print(f"[+] tq: {X_tq.shape}")

    # =========================
    # RUN
    # =========================
    def build_all(self):
        self.load()
        self.add_label()
        self.clean()
        self.debug_labels()
        self.build_lstm()


if __name__ == "__main__":
    DatasetBuilder(
        master_path=Config.DATASET_PATH,
        seq_len=50,
        horizon=10,
        threshold=0.002
    ).build_all()