import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    # =========================
    # FIT SCALER
    # =========================
    def fit_scaler(self, features):
        self.scaler.fit(features)
        self.is_fitted = True
        print("[*] Scaler fitted")

    # =========================
    # COMPUTE FEATURE
    # =========================
    def compute_features(self, df):
        df = df.copy()

        if df is None or len(df) < 35:
            return None

        try:
            # =========================
            # PRICE FEATURES
            # =========================
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))

            # 🔥 NEW: short-term momentum (5-step return)
            df['return_5'] = df['close'].pct_change(5)

            # =========================
            # VOLUME FEATURES
            # =========================
            df['log_volume'] = np.log(df['volume'] + 1e-8)

            # =========================
            # CANDLE PRESSURE FEATURE (NEW)
            # =========================
            df['candle_body'] = (df['close'] - df['open']) / df['open']

            # =========================
            # RSI
            # =========================
            df['rsi'] = ta.rsi(df['close'], length=14)

            # =========================
            # MACD
            # =========================
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)

            if macd is None or macd.empty:
                return None

            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']

            # =========================
            # VOLATILITY FEATURES
            # =========================

            df['volatility'] = df['log_return'].rolling(window=14).std()

            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

            # =========================
            # FINAL FEATURE VECTOR
            # =========================
            last_row = df[
                [
                    'log_return',
                    'return_5',        # 🔥 NEW
                    'log_volume',
                    'candle_body',     # 🔥 NEW
                    'rsi',
                    'macd',
                    'macd_signal',
                    'volatility',
                    'atr'
                ]
            ].iloc[-1]

            # =========================
            # CLEAN CHECK
            # =========================
            if last_row.isnull().any():
                return None

            return last_row.values.astype(float)

        except Exception as e:
            print("[!] FeatureEngineer error:", e)
            return None

    # =========================
    # NORMALIZE
    # =========================
    def normalize_features(self, vector):
        if not self.is_fitted:
            return vector

        return self.scaler.transform([vector])[0]