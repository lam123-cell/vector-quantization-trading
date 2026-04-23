import os
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from sklearn.preprocessing import StandardScaler
    import joblib
except Exception:
    StandardScaler = None
    joblib = None


# =========================
# LOAD DATA (numpy)
# =========================
def load_lstm_data(X_path, y_path, split_ratio=0.8, scale=False, scaler_path=None):
    """Load LSTM data from .npy files.

    If ``scale`` is True, fit a StandardScaler on the training portion (first
    ``split_ratio`` of samples) and transform the whole dataset. If ``scaler_path``
    is provided the fitted scaler will be saved there. If a scaler already exists
    at ``scaler_path``, it will be loaded and used instead of fitting.
    Returns: X_train, y_train, X_test, y_test (numpy arrays)
    """
    X = np.load(X_path)
    y = np.load(y_path)

    if not scale:
        split = int(len(X) * split_ratio)
        X_train = X[:split]
        y_train = y[:split]
        X_test = X[split:]
        y_test = y[split:]
        return X_train, y_train, X_test, y_test

    if StandardScaler is None:
        raise RuntimeError('scikit-learn is required for scaling but is not installed')

    split = int(len(X) * split_ratio)

    # fit or load scaler on training portion
    N, S, F = X.shape

    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(X[:split].reshape(-1, F))
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)

    # transform whole dataset
    X_flat = X.reshape(-1, F)
    X_scaled = scaler.transform(X_flat).reshape(N, S, F)

    X_train = X_scaled[:split].astype(np.float32)
    y_train = y[:split]
    X_test = X_scaled[split:].astype(np.float32)
    y_test = y[split:]

    return X_train, y_train, X_test, y_test


# =========================
# PYTORCH DATASET
# =========================
class SequenceDataset(Dataset):
    def __init__(self, X_path, y_path, scale=False, split_ratio=0.8, scaler_path=None):
        """PyTorch dataset for (N, seq_len, features) sequences.

        If ``scale`` is True the loader will fit a scaler on the first
        ``split_ratio`` portion (train) and transform the whole X. The scaler
        will be saved to ``scaler_path`` if provided.
        """
        X = np.load(X_path)
        y = np.load(y_path)

        if scale:
            if StandardScaler is None:
                raise RuntimeError('scikit-learn is required for scaling but is not installed')

            split = int(len(X) * split_ratio)
            N, S, F = X.shape

            if scaler_path and os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                scaler = StandardScaler()
                scaler.fit(X[:split].reshape(-1, F))
                if scaler_path:
                    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                    joblib.dump(scaler, scaler_path)

            X = scaler.transform(X.reshape(-1, F)).reshape(N, S, F)

        print("[DEBUG] X dtype:", X.dtype)
        print("[DEBUG] y dtype:", y.dtype)

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]