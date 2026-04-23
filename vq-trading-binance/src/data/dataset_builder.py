import os
import numpy as np
import pandas as pd


class DatasetBuilder:
    def __init__(self, master_path, output_dir="datasets"):
        self.master_path = master_path
        self.output_dir = output_dir

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
    # LABEL (QUANTILE - CLEAN)
    # =========================
    def add_label(self, horizon=5):
        df = self.df.copy()

        df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1

        q_low = df["future_return"].quantile(0.33)
        q_high = df["future_return"].quantile(0.66)

        def label_fn(x):
            if x > q_high:
                return 2   # BUY
            elif x < q_low:
                return 0   # SELL
            else:
                return 1   # HOLD

        df["label"] = df["future_return"].apply(label_fn)

        self.df = df

        print(f"[*] Label (quantile)")
        print(f"    low  = {q_low:.6f}")
        print(f"    high = {q_high:.6f}")

    # =========================
    # DEBUG LABEL
    # =========================
    def debug_labels(self, df, name):
        print(f"\n===== LABEL DIST ({name}) =====")
        counts = df["label"].value_counts().sort_index()
        total = len(df)

        for k in [0, 1, 2]:
            v = counts.get(k, 0)
            print(f"{k}: {v} ({v/total:.4f})")

        print("==============================")

    # =========================
    # SEQUENCE BUILDER
    # =========================
    def _build_seq(self, data, labels, seq_len):
        X, y = [], []

        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(labels[i+seq_len])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    # =========================
    # BUILD BASELINE (n_*)
    # =========================
    def build_baseline(self, seq_len):
        cols = [c for c in self.df.columns if c.startswith("n_")]

        df = self.df.dropna(subset=["label"] + cols).reset_index(drop=True)

        self.debug_labels(df, "BASELINE")

        X, y = self._build_seq(
            df[cols].values,
            df["label"].values,
            seq_len
        )

        np.save(f"{self.lstm_dir}/lstm_baseline_X.npy", X)
        np.save(f"{self.lstm_dir}/lstm_baseline_y.npy", y)

        print(f"[+] Baseline saved: {X.shape}")

    # =========================
    # BUILD TQ (tq_xhat_*)
    # =========================
    def build_tq(self, seq_len):
        cols = [c for c in self.df.columns if c.startswith("tq_xhat_")]

        df = self.df.dropna(subset=["label"] + cols).reset_index(drop=True)

        self.debug_labels(df, "TQ")

        X, y = self._build_seq(
            df[cols].values,
            df["label"].values,
            seq_len
        )

        np.save(f"{self.lstm_dir}/lstm_tq_X.npy", X)
        np.save(f"{self.lstm_dir}/lstm_tq_y.npy", y)

        print(f"[+] TQ saved: {X.shape}")

    # =========================
    # MAIN PIPELINE
    # =========================
    def build_all(self, seq_len=50):
        self.load()
        self.add_label()

        # ❌ KHÔNG balance
        # ❌ KHÔNG trộn feature
        # ❌ KHÔNG dùng chung clean

        self.build_baseline(seq_len)
        self.build_tq(seq_len)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    builder = DatasetBuilder("dataset_master.csv")
    builder.build_all(seq_len=50)