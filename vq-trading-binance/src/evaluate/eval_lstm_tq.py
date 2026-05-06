import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from collections import Counter
import time

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from src.models.lstm import LSTMModel
from src.utils.dataset_loader import SequenceDataset


# =========================
# CONFIG
# =========================
DATA_DIR = "datasets/lstm"
MODEL_PATH = "results/lstm/lstm_tq.pt"
SCALER_PATH = f"{DATA_DIR}/tq_scaler.joblib"

SEQ_LEN = 50
HORIZON = 10
THRESHOLD = 0.0025

CONF_THRESHOLD = 0.55   # 🔥 quan trọng
NOISE_FILTER = 0.002    # 🔥 lọc market noise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128


# =========================
# BUILD DF (MATCH BUILDER)
# =========================
def build_df():
    df = pd.read_csv("datasets/dataset_master.csv")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    df["future_return"] = df["close"].shift(-HORIZON) / df["close"] - 1

    def label_fn(x):
        if pd.isna(x):
            return np.nan
        if x > THRESHOLD:
            return 2
        elif x < -THRESHOLD:
            return 0
        else:
            return 1

    df["label"] = df["future_return"].apply(label_fn)

    cols = [c for c in df.columns if c.startswith("tq_")]
    df = df.dropna(subset=["label"] + cols).reset_index(drop=True)

    return df


# =========================
# LOAD DATA
# =========================
dataset = SequenceDataset(
    f"{DATA_DIR}/lstm_tq_X.npy",
    f"{DATA_DIR}/lstm_tq_y.npy",
    scale=True,
    split_ratio=0.8,
    scaler_path=SCALER_PATH
)

split = int(0.8 * len(dataset))
val_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

labels = dataset.y.numpy()

print("\n===== VALID LABEL DIST =====")
print(Counter(labels[split:]))


# =========================
# MODEL
# =========================
input_dim = dataset[0][0].shape[-1]

model = LSTMModel(input_dim=input_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# =========================
# INFERENCE
# =========================
start_time = time.time()

preds_all = []
trues_all = []
confs_all = []

with torch.no_grad():
    for X, y in val_loader:
        X = X.to(DEVICE)

        out = model(X)
        probs = torch.softmax(out, dim=1)
        conf, preds = torch.max(probs, dim=1)

        preds_all.extend(preds.cpu().numpy())
        confs_all.extend(conf.cpu().numpy())
        trues_all.extend(y.numpy())

latency = (time.time() - start_time) / len(preds_all)


# =========================
# DEBUG CONFIDENCE
# =========================
print("\n===== CONFIDENCE DEBUG =====")
print(f"Max: {np.max(confs_all):.4f}")
print(f"Min: {np.min(confs_all):.4f}")
print(f"Avg: {np.mean(confs_all):.4f}")


# =========================
# METRICS (ONE BLOCK)
# =========================
acc = accuracy_score(trues_all, preds_all)
f1 = f1_score(trues_all, preds_all, average="macro")
precision = precision_score(trues_all, preds_all, average="macro")
recall = recall_score(trues_all, preds_all, average="macro")

print("\n================ METRICS ================")
print(f"Accuracy          : {acc:.4f}")
print(f"F1 Macro          : {f1:.4f}")
print(f"Precision Macro   : {precision:.4f}")
print(f"Recall Macro      : {recall:.4f}")
print(f"Prediction Latency: {latency:.6f} sec/sample")

print("\n===== PRED DISTRIBUTION =====")
print(Counter(preds_all))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(trues_all, preds_all, digits=4))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(trues_all, preds_all))


# =========================
# BACKTEST
# =========================
df = build_df()

capital = 1.0
fee = 0.00075

trades = 0
wins = 0
returns = []
equity_curve = []

for i, (pred, conf) in enumerate(zip(preds_all, confs_all)):
    idx = split + i

    entry = idx + SEQ_LEN
    exit_ = entry + HORIZON

    if exit_ >= len(df):
        break

    p0 = df.loc[entry, "close"]
    p1 = df.loc[exit_, "close"]

    ret = (p1 - p0) / p0

    # ===== FILTER =====
    if conf < CONF_THRESHOLD:
        continue

    if abs(ret) < NOISE_FILTER:
        continue

    if pred == 2:
        trade = ret - 2 * fee
    elif pred == 0:
        trade = -ret - 2 * fee
    else:
        continue

    trades += 1
    returns.append(trade)

    if trade > 0:
        wins += 1

    capital *= (1 + trade)
    equity_curve.append(capital)


# ===== DRAW DOWN =====
equity_curve = np.array(equity_curve)

if len(equity_curve) > 0:
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
else:
    max_dd = 0


# =========================
# BACKTEST METRICS
# =========================
print("\n================ BACKTEST ================")
print(f"Trades           : {trades}")
print(f"Winrate          : {wins / trades if trades else 0:.4f}")
print(f"Total Return     : {capital - 1:.4f}")
print(f"Avg Trade        : {np.mean(returns) if returns else 0:.6f}")
print(f"Max Drawdown     : {max_dd:.4f}")

np.save("results/tq_preds.npy", preds_all)
np.save("results/tq_trues.npy", trues_all)
np.save("results/tq_returns.npy", returns)
np.save("results/tq_confs.npy", confs_all)