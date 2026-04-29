import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib


class SequenceDataset(Dataset):
    def __init__(self, X_path, y_path, scale=False, split_ratio=0.8, scaler_path=None):
        X = np.load(X_path)
        y = np.load(y_path)

        if scale:
            split = int(len(X) * split_ratio)
            N, S, F = X.shape

            if scaler_path is None:
                raise ValueError("scaler_path must be provided when scale=True")

            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                scaler = StandardScaler()
                scaler.fit(X[:split].reshape(-1, F))

                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                joblib.dump(scaler, scaler_path)

            X = scaler.transform(X.reshape(-1, F)).reshape(N, S, F)

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]