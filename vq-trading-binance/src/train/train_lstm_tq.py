# src/train/train_lstm_tq.py

import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.models.lstm import LSTMModel
from src.utils.dataset_loader import SequenceDataset


DATA_DIR = "datasets/lstm"
RESULT_DIR = "results/lstm"
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RESULT_DIR, "lstm_tq.pt")
SCALER_PATH = os.path.join(DATA_DIR, "tq_scaler.joblib")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 10
LR = 5e-4


# =========================
# LOAD
# =========================
dataset = SequenceDataset(
    f"{DATA_DIR}/lstm_tq_X.npy",
    f"{DATA_DIR}/lstm_tq_y.npy",
    scale=True,
    split_ratio=0.8,
    scaler_path=f"{DATA_DIR}/scaler.joblib"
)

labels = dataset.y.numpy()

print("\n===== LABEL DIST =====")
print(Counter(labels))


# =========================
# CLASS WEIGHT (compute on TRAIN split)
# =========================
train_size = int(0.8 * len(dataset))
train_labels = labels[:train_size]

class_counts = np.bincount(train_labels)

weights = 1.0 / class_counts
weights = weights / weights.sum()

# boost BUY/SELL
weights[0] *= 1.5
weights[2] *= 1.5

class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

print("[DEBUG] class weights (train):", class_weights)


# =========================
# SPLIT (FIX BUG)
# =========================

train_idx = list(range(0, train_size))
val_idx = list(range(train_size, len(dataset)))

train_ds = torch.utils.data.Subset(dataset, train_idx)
val_ds = torch.utils.data.Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# MODEL
# =========================
input_dim = dataset[0][0].shape[-1]

print("[DEBUG] input_dim:", input_dim)

model = LSTMModel(
    input_dim=input_dim,
    hidden_dim=128,
    dropout=0.3
).to(DEVICE)


criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# early stopping
best_val_f1 = -1.0
patience = 5
no_improve = 0


# =========================
# TRAIN
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    # ===== VALIDATION =====
    model.eval()
    correct = 0
    total = 0
    preds_all = []
    trues_all = []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            out = model(X)
            preds = torch.argmax(out, dim=1)

            preds_all.extend(preds.cpu().numpy())
            trues_all.extend(y.cpu().numpy())

            correct += (preds == y).sum().item()
            total += len(y)

    acc = correct / total

    val_f1 = f1_score(trues_all, preds_all, average='macro')

    print(f"\n[Epoch {epoch+1}] Loss: {total_loss:.2f} | Val Acc: {acc:.4f} | Val F1(macro): {val_f1:.4f}")
    print("Pred dist:", Counter(preds_all))
    print("Val classification report:\n", classification_report(trues_all, preds_all, digits=4))

    # scheduler + early stopping
    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        no_improve = 0
        torch.save(model.state_dict(), MODEL_PATH)  # save best
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"Early stopping (no improvement {patience} epochs)")
        break


torch.save(model.state_dict(), MODEL_PATH)
print(f"[+] Saved model: {MODEL_PATH}")