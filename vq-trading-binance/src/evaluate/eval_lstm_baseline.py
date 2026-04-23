import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter

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
MODEL_PATH = "lstm_baseline.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128


# =========================
# LOAD DATA
# =========================
dataset = SequenceDataset(
    f"{DATA_DIR}/lstm_baseline_X.npy",
    f"{DATA_DIR}/lstm_baseline_y.npy"
)

labels = dataset.y.numpy()

# TIME SPLIT (same as train)
split = int(0.8 * len(dataset))
val_indices = np.arange(split, len(dataset))

val_ds = torch.utils.data.Subset(dataset, val_indices)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\n===== VALIDATION LABEL DIST =====")
print(Counter(labels[val_indices]))


# =========================
# LOAD MODEL
# =========================
input_dim = dataset[0][0].shape[-1]

model = LSTMModel(input_dim=input_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# =========================
# INFERENCE
# =========================
all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in val_loader:
        X = X.to(DEVICE)

        out = model(X)
        preds = torch.argmax(out, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())


# =========================
# METRICS
# =========================
print("\n===== PREDICTION DISTRIBUTION =====")
print(Counter(all_preds))

acc = accuracy_score(all_labels, all_preds)
f1_macro = f1_score(all_labels, all_preds, average="macro")
f1_weighted = f1_score(all_labels, all_preds, average="weighted")

precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)

print("\n===== SUMMARY METRICS =====")
print(f"Accuracy       : {acc:.4f}")
print(f"F1 (macro)     : {f1_macro:.4f}")
print(f"F1 (weighted)  : {f1_weighted:.4f}")
print(f"Precision macro: {precision_macro:.4f}")
print(f"Recall macro   : {recall_macro:.4f}")


print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(all_labels, all_preds, digits=4, zero_division=0))


print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(all_labels, all_preds))