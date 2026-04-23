import numpy as np
import torch
from torch.utils.data import Dataset


# =========================
# LOAD DATA (numpy)
# =========================
def load_lstm_data(X_path, y_path, split_ratio=0.8):
    X = np.load(X_path)
    y = np.load(y_path)

    split = int(len(X) * split_ratio)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    return X_train, y_train, X_test, y_test


# =========================
# PYTORCH DATASET
# =========================
class SequenceDataset(Dataset):
    def __init__(self, X_path, y_path):
        X = np.load(X_path)
        y = np.load(y_path)

        print("[DEBUG] X dtype:", X.dtype)
        print("[DEBUG] y dtype:", y.dtype)

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]