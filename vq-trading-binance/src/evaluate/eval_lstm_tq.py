import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from src.models.lstm import LSTMModel
from src.utils.dataset_loader import SequenceDataset


# =========================
# CONFIG (MUST MATCH BUILDER)
# =========================
DATA_DIR = "datasets/lstm"
MODEL_PATH = "results/lstm/lstm_baseline.pt"
SCALER_PATH = f"{DATA_DIR}/baseline_scaler.joblib"

SEQ_LEN = 50
HORIZON = 10          # ✅ FIX
THRESHOLD = 0.0025    # ✅ phải match builder

CONF_THRESHOLD = 0.55  # ✅ giảm xuống

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128


# =========================
# BUILD DF (SAME AS BUILDER)
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

    cols = [c for c in df.columns if c.startswith("n_")]
    df = df.dropna(subset=["label"] + cols).reset_index(drop=True)

    return df


# =========================
# LOAD DATA
# =========================
dataset = SequenceDataset(
    f"{DATA_DIR}/lstm_baseline_X.npy",
    f"{DATA_DIR}/lstm_baseline_y.npy",
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


# =========================
# DEBUG CONFIDENCE
# =========================
print("\n===== CONF DEBUG =====")
print("Max:", np.max(confs_all))
print("Min:", np.min(confs_all))
print("Avg:", np.mean(confs_all))


# =========================
# METRICS
# =========================
print("\n===== METRICS =====")
print("Accuracy:", accuracy_score(trues_all, preds_all))
print("F1 macro:", f1_score(trues_all, preds_all, average="macro"))

print("\n===== REPORT =====")
print(classification_report(trues_all, preds_all, digits=4))

print("\n===== CONFUSION =====")
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

    if abs(ret) < 0.002:  # noise filter
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


print("\n===== BACKTEST =====")
print("Trades:", trades)
print("Winrate:", wins / trades if trades else 0)
print("Return:", capital - 1)
print("Avg trade:", np.mean(returns) if returns else 0)