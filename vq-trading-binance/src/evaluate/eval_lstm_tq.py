import torch
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

from src.models.lstm import LSTMModel
from src.utils.dataset_loader import SequenceDataset


DATA_DIR = "datasets/lstm"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# LOAD DATA
# =========================
dataset = SequenceDataset(
    f"{DATA_DIR}/lstm_tq_X.npy",
    f"{DATA_DIR}/lstm_tq_y.npy"
)

split = int(0.8 * len(dataset))

val_ds = torch.utils.data.Subset(dataset, list(range(split, len(dataset))))
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)


# =========================
# LOAD MODEL
# =========================
input_dim = dataset[0][0].shape[-1]

model = LSTMModel(input_dim=input_dim).to(DEVICE)
model.load_state_dict(torch.load("lstm_tq.pt"))
model.eval()


# =========================
# EVAL
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
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(all_labels, all_preds, digits=4))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(all_labels, all_preds))