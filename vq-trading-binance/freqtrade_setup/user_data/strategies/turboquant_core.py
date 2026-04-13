import numpy as np


class TurboQuant:
    """
    Data-oblivious TurboQuant implementation.

    Pipeline:
    1) Random rotation via QR decomposition of a Gaussian matrix.
    2) Scalar quantization on each rotated dimension using a fixed codebook.
    3) Dequantization by mapping indices back to scalar centroids.
    4) Inverse rotation to recover the approximate original vector.

    This version does NOT train on data and does NOT use k-means.
    """

    def __init__(self, feature_dim=3, levels=16, value_range=(-1.0, 1.0), seed=42):
        self.feature_dim = int(feature_dim)
        self.levels = int(levels)
        self.low = float(value_range[0])
        self.high = float(value_range[1])
        self.rng = np.random.default_rng(seed)

        # Fixed scalar codebook shared by every dimension.
        self.scalar_codebook = np.linspace(self.low, self.high, self.levels)

        # Build one random orthonormal rotation matrix at initialization.
        self.rotation_matrix = self._build_random_rotation()

    def _build_random_rotation(self):
        """Create an orthonormal random rotation matrix using QR decomposition."""
        gaussian = self.rng.normal(size=(self.feature_dim, self.feature_dim))
        q, r = np.linalg.qr(gaussian)

        # Fix signs so the matrix is a valid rotation basis and deterministic for the seed.
        diag = np.sign(np.diag(r))
        diag[diag == 0] = 1.0
        q = q * diag
        return q

    def _quantize_scalar(self, value):
        """Map one scalar to the nearest fixed centroid in [-1, 1]."""
        idx = int(np.argmin(np.abs(self.scalar_codebook - value)))
        return idx, float(self.scalar_codebook[idx])

    def dequantize(self, indices):
        """
        Reconstruct an approximate vector from scalar quantization indices.

        Parameters
        ----------
        indices : array-like, shape (feature_dim,)
            Scalar codebook indices for each rotated dimension.

        Returns
        -------
        np.ndarray
            Reconstructed x_hat in the original space.
        """
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} indices, got {indices.size}")

        # Decode scalar values in the rotated space.
        y_hat = self.scalar_codebook[indices]

        # Apply inverse rotation (transpose of an orthonormal matrix).
        x_hat = self.rotation_matrix.T @ y_hat
        return x_hat

    def quantize(self, x):
        """
        Quantize a single vector using the required TurboQuant pipeline.

        Returns a dict with:
        - indices: scalar codebook indices per dimension
        - y: rotated vector before scalar quantization
        - y_hat: dequantized rotated vector
        - x_hat: reconstructed vector in original space
        - error_norm: ||x - x_hat||_2
        - score: simple regime score derived from x_hat
        - regime: {-1, 0, 1} based on score
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != self.feature_dim:
            raise ValueError(f"Expected vector of size {self.feature_dim}, got {x.size}")

        # Data-oblivious implementation: just clip to the supported working range.
        x = np.nan_to_num(x, nan=0.0, posinf=self.high, neginf=self.low)
        x = np.clip(x, self.low, self.high)

        # 1) Random rotation.
        y = self.rotation_matrix @ x
        y = np.clip(y, self.low, self.high)

        # 2) Scalar quantization dimension-by-dimension.
        indices = np.zeros(self.feature_dim, dtype=np.int64)
        y_hat = np.zeros(self.feature_dim, dtype=np.float64)
        for i in range(self.feature_dim):
            idx, centroid = self._quantize_scalar(y[i])
            indices[i] = idx
            y_hat[i] = centroid

        # 3) Dequantization and inverse rotation.
        x_hat = self.dequantize(indices)

        # Simple heuristic score for trading logic.
        score = self._score_from_reconstruction(x_hat)
        regime = self._regime_from_score(score)
        error_norm = float(np.linalg.norm(x - x_hat))

        return {
            "indices": indices,
            "y": y,
            "y_hat": y_hat,
            "x_hat": x_hat,
            "error_norm": error_norm,
            "score": float(score),
            "regime": int(regime),
        }

    def quantize_batch(self, matrix):
        """Quantize a batch of vectors row by row."""
        x = np.asarray(matrix, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"Expected shape (n, {self.feature_dim}), got {x.shape}")

        if x.shape[0] == 0:
            return {
                "indices": np.empty((0, self.feature_dim), dtype=np.int64),
                "codes": np.empty((0,), dtype=np.int64),
                "y_hat": np.empty((0, self.feature_dim), dtype=np.float64),
                "x_hat": np.empty((0, self.feature_dim), dtype=np.float64),
                "error_norm": np.empty((0,), dtype=np.float64),
                "score": np.empty((0,), dtype=np.float64),
                "regime": np.empty((0,), dtype=np.int8),
            }

        indices_list = []
        y_hat_list = []
        x_hat_list = []
        error_list = []
        score_list = []
        regime_list = []

        for row in x:
            result = self.quantize(row)
            indices_list.append(result["indices"])
            y_hat_list.append(result["y_hat"])
            x_hat_list.append(result["x_hat"])
            error_list.append(result["error_norm"])
            score_list.append(result["score"])
            regime_list.append(result["regime"])

        indices = np.asarray(indices_list, dtype=np.int64)
        codes = np.ravel_multi_index(indices.T, dims=(self.levels,) * self.feature_dim)

        return {
            "indices": indices,
            "codes": codes.astype(np.int64),
            "y_hat": np.asarray(y_hat_list, dtype=np.float64),
            "x_hat": np.asarray(x_hat_list, dtype=np.float64),
            "error_norm": np.asarray(error_list, dtype=np.float64),
            "score": np.asarray(score_list, dtype=np.float64),
            "regime": np.asarray(regime_list, dtype=np.int8),
        }

    def _score_from_reconstruction(self, x_hat):
        """Turn the reconstructed vector into a simple trading score."""
        # x_hat = [rsi_norm, macdhist_norm, ret_1] after reconstruction.
        return (-0.45 * x_hat[0]) + (0.40 * x_hat[1]) + (0.15 * x_hat[2])

    def _regime_from_score(self, score):
        if score > 0.05:
            return 1
        if score < -0.05:
            return -1
        return 0
