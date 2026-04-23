import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter

from src.models.lstm import LSTMModel
from src.utils.dataset_loader import SequenceDataset


# =========================
# CONFIG
# =========================
DATA_DIR = "datasets/lstm"
RESULT_DIR = "results/lstm"
os.makedirs(RESULT_DIR, exist_ok=True)

save_path = os.path.join(RESULT_DIR, "lstm_baseline.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 20
LR = 5e-4


# =========================
# LOAD DATA
# =========================
dataset = SequenceDataset(
    f"{DATA_DIR}/lstm_baseline_X.npy",
    f"{DATA_DIR}/lstm_baseline_y.npy"
)

labels = dataset.y.numpy()
print("\n===== LABEL DIST =====")
print(Counter(labels))


# =========================
# SPLIT (TIME SERIES)
# =========================
train_size = int(0.8 * len(dataset))

train_indices = np.arange(0, train_size)
val_indices = np.arange(train_size, len(dataset))

train_ds = torch.utils.data.Subset(dataset, train_indices)
val_ds = torch.utils.data.Subset(dataset, val_indices)


# =========================
# SAMPLER (CÂN BẰNG DATA)
# =========================
train_labels = labels[train_indices]

class_counts = np.bincount(train_labels)

weights = 1.0 / np.sqrt(class_counts)

# 🔥 boost nhẹ class BUY
if len(weights) > 2:
    weights[2] *= 1.3

weights = weights / weights.sum()

print("[DEBUG] class weights:", weights)

sample_weights = weights[train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# =========================
# DATALOADER
# =========================
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# MODEL
# =========================
input_dim = dataset[0][0].shape[-1]

model = LSTMModel(
    input_dim=input_dim,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3
).to(DEVICE)


# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# =========================
# TRAIN LOOP
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

    # chỉ monitor nhẹ
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")


# =========================
# SAVE
# =========================
torch.save(model.state_dict(), save_path)
print(f"[+] Saved model: {save_path}")