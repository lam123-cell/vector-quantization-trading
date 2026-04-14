import numpy as np


class TurboQuant:
    """
    TurboQuant for streaming financial features (9D).

    Compatible with:
    - Freqtrade
    - DRL (state representation)
    - LSTM (sequence modeling)
    """

    def __init__(self, feature_dim=9, levels=16, value_range=(-3, 3), seed=42):
        self.feature_dim = feature_dim
        self.levels = levels
        self.low, self.high = value_range
        self.rng = np.random.default_rng(seed)

        # Scalar quantization codebook
        self.scalar_codebook = np.linspace(self.low, self.high, self.levels)

        # Random orthonormal rotation
        self.rotation_matrix = self._build_rotation()

    # =========================
    # ROTATION
    # =========================
    def _build_rotation(self):
        A = self.rng.normal(size=(self.feature_dim, self.feature_dim))
        Q, R = np.linalg.qr(A)

        diag = np.sign(np.diag(R))
        diag[diag == 0] = 1.0
        Q = Q * diag

        return Q

    # =========================
    # CORE QUANTIZATION
    # =========================
    def quantize(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)

        if x.shape[0] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim}, got {x.shape[0]}")

        # Clean + clip
        x = np.nan_to_num(x)
        x = np.clip(x, self.low, self.high)

        # 1. Rotate
        y = self.rotation_matrix @ x
        y = np.clip(y, self.low, self.high)

        # 2. Scalar quantization
        indices = np.argmin(
            np.abs(self.scalar_codebook[:, None] - y[None, :]),
            axis=0
        )

        y_hat = self.scalar_codebook[indices]

        # 3. Inverse rotation
        x_hat = self.rotation_matrix.T @ y_hat

        # 4. Metrics
        error = np.linalg.norm(x - x_hat)

        score = self._score(x_hat)
        regime = self._regime(score)
        confidence = 1.0 / (1.0 + error)

        return {
            "indices": indices,
            "code": self._to_code(indices),
            "x_hat": x_hat,
            "error": error,
            "score": score,
            "regime": regime,
            "confidence": confidence
        }

    # =========================
    # BATCH
    # =========================
    def quantize_batch(self, X):
        X = np.asarray(X, dtype=float)

        outputs = [self.quantize(x) for x in X]

        return {
            "codes": np.array([o["code"] for o in outputs]),
            "states": np.array([o["indices"] for o in outputs]),
            "reconstruction": np.array([o["x_hat"] for o in outputs]),
            "error": np.array([o["error"] for o in outputs]),
            "score": np.array([o["score"] for o in outputs]),
            "regime": np.array([o["regime"] for o in outputs]),
            "confidence": np.array([o["confidence"] for o in outputs]),
        }

    # =========================
    # ENCODE (FOR DRL)
    # =========================
    def encode(self, x):
        return self.quantize(x)

    def encode_batch(self, X):
        return self.quantize_batch(X)

    # =========================
    # HELPER
    # =========================
    def _to_code(self, indices):
        return np.ravel_multi_index(indices, (self.levels,) * self.feature_dim)

    # =========================
    # 🔥 SCORE FUNCTION (9D)
    # =========================
    def _score(self, x):
        """
        x = [
            log_return,
            return_5,
            log_volume,
            candle_body,
            rsi,
            macd,
            macd_signal,
            volatility,
            atr
        ]
        """

        return (
            0.25 * x[0] +   # log_return (short-term)
            0.25 * x[1] +   # return_5 (momentum)
            0.15 * x[3] +   # candle_body (pressure)
            0.15 * x[5] +   # macd (trend)
            -0.10 * x[7]    # volatility (risk penalty)
        )

    def _regime(self, score):
        if score > 0.1:
            return 1   # BUY
        elif score < -0.1:
            return -1  # SELL
        else:
            return 0   # HOLD