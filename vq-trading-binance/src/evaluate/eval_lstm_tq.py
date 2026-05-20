import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
import time

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128


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

val_ds = torch.utils.data.Subset(
    dataset,
    range(split, len(dataset))
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

labels = dataset.y.numpy()

print("\n===== VALID LABEL DIST =====")
print(Counter(labels[split:]))


# =========================
# MODEL
# =========================
input_dim = dataset[0][0].shape[-1]

model = LSTMModel(
    input_dim=input_dim
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

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

        preds_all.extend(
            preds.cpu().numpy()
        )

        confs_all.extend(
            conf.cpu().numpy()
        )

        trues_all.extend(
            y.numpy()
        )

latency = (
    time.time() - start_time
) / len(preds_all)


# =========================
# CONFIDENCE DEBUG
# =========================
print("\n===== CONFIDENCE DEBUG =====")

print(f"Max: {np.max(confs_all):.4f}")
print(f"Min: {np.min(confs_all):.4f}")
print(f"Avg: {np.mean(confs_all):.4f}")


# =========================
# METRICS
# =========================
acc = accuracy_score(
    trues_all,
    preds_all
)

f1 = f1_score(
    trues_all,
    preds_all,
    average="macro"
)

precision = precision_score(
    trues_all,
    preds_all,
    average="macro"
)

recall = recall_score(
    trues_all,
    preds_all,
    average="macro"
)

balanced_acc = balanced_accuracy_score(
    trues_all,
    preds_all
)

print("\n================ METRICS ================")

print(f"Accuracy          : {acc:.4f}")
print(f"F1 Macro          : {f1:.4f}")
print(f"Precision Macro   : {precision:.4f}")
print(f"Recall Macro      : {recall:.4f}")
print(f"Balanced Accuracy : {balanced_acc:.4f}")

print(
    f"Prediction Latency: "
    f"{latency:.6f} sec/sample"
)

print("\n===== PRED DISTRIBUTION =====")
print(Counter(preds_all))

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        trues_all,
        preds_all,
        digits=4
    )
)

print("\n===== CONFUSION MATRIX =====")

print(
    confusion_matrix(
        trues_all,
        preds_all
    )
)


# =========================
# SAVE
# =========================
np.save(
    "results/tq_preds.npy",
    preds_all
)

np.save(
    "results/tq_trues.npy",
    trues_all
)

np.save(
    "results/tq_confs.npy",
    confs_all
)

np.save(
    "results/tq_latency.npy",
    np.array([latency], dtype=np.float32)
)
