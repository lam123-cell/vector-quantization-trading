import pandas as pd
import pandas_ta as ta
import numpy as np

class FeatureEngineer:

    def compute_features(self, df):
        df = df.copy()

        if df is None or len(df) < 50:  # 🔥 tăng lên cho chắc
            return None

        try:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))

            df['rsi'] = ta.rsi(df['close'], length=14)

            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)

            if macd is None or macd.empty:
                return None

            # ✅ FIX CHUẨN
            if 'MACD_12_26_9' not in macd.columns:
                return None

            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']

            features_df = df[['log_return', 'rsi', 'macd', 'macd_signal']].dropna()

            if len(features_df) < 5:
                return None

            return features_df.iloc[-1].values

        except Exception as e:
            print("[!] FeatureEngineer error:", e)
            return None

    def normalize_features(self, vector):
        return vector