import pandas as pd
import pandas_ta as ta
import numpy as np


class FeatureEngineer:

    def compute_features(self, df):

        # ✅ FIX 1: copy để tránh mutate
        df = df.copy()

        # ✅ FIX 2: đủ dữ liệu cho MACD
        if df is None or len(df) < 35:
            return None

        try:
            # 1. Log return
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))

            # 2. RSI
            df['rsi'] = ta.rsi(df['close'], length=14)

            # 3. MACD
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)

            if macd is None or 'MACDS_12_26_9' not in macd.columns:
                return None

            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDS_12_26_9']

            # 4. ATR
            atr = ta.atr(df['high'], df['low'], df['close'], length=14)

            if atr is None:
                return None

            df['atr_norm'] = atr / df['close']

            # 5. drop NaN
            features_df = df[
                ['log_return', 'rsi', 'macd', 'macd_signal', 'atr_norm']
            ].dropna()

            if features_df.empty:
                return None

            return features_df.iloc[-1].values

        except Exception as e:
            print("[!] FeatureEngineer error:", e)
            return None

    def normalize_features(self, vector):
        return vector