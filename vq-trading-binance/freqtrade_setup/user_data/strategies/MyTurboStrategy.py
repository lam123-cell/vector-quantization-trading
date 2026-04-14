# import sys
# from pathlib import Path
# import numpy as np
# from pandas import DataFrame
# from freqtrade.strategy.interface import IStrategy
# import joblib

# # ===== ADD ROOT PATH =====
# BASE_DIR = Path(__file__).resolve().parents[2]
# sys.path.append(str(BASE_DIR))

# # ===== IMPORT CORE =====
# from src.feature.feature_engineer import FeatureEngineer
# from freqtrade_setup.turboquant_core import TurboQuant


# class MyTurboStrategy(IStrategy):
#     INTERFACE_VERSION = 3

#     minimal_roi = {
#         "0": 0.02,
#         "30": 0.01,
#         "60": 0
#     }

#     stoploss = -0.10
#     timeframe = "15m"

#     process_only_new_candles = True
#     startup_candle_count = 50

#     def __init__(self, config: dict) -> None:
#         super().__init__(config)

#         # ===== FEATURE ENGINEER =====
#         self.fe = FeatureEngineer()

#         # 🔥 LOAD SCALER (BẮT BUỘC)
#         self.fe.scaler = joblib.load("scaler.pkl")
#         self.fe.is_fitted = True

#         # ===== TURBO QUANT =====
#         self.tq = TurboQuant(
#             feature_dim=9,
#             levels=16,
#             value_range=(-3, 3)
#         )

#     def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

#         tq_codes = []
#         tq_scores = []
#         tq_regimes = []
#         tq_dist = []
#         tq_features = []

#         for i in range(len(dataframe)):
#             sub_df = dataframe.iloc[:i+1]

#             # ===== FEATURE =====
#             feature = self.fe.compute_features(sub_df)

#             if feature is None:
#                 tq_codes.append(0)
#                 tq_scores.append(0)
#                 tq_regimes.append(0)
#                 tq_dist.append(0)
#                 tq_features.append(np.zeros(9))
#                 continue

#             # ===== NORMALIZE =====
#             norm = self.fe.normalize_features(feature)

#             # ===== TURBO QUANT =====
#             tq = self.tq.quantize(norm)

#             tq_codes.append(np.sum(tq["indices"]))
#             tq_scores.append(tq["score"])
#             tq_regimes.append(tq["regime"])
#             tq_dist.append(tq["error_norm"])
#             tq_features.append(tq["x_hat"])

#         tq_features = np.array(tq_features)

#         dataframe["tq_code"] = tq_codes
#         dataframe["tq_score"] = tq_scores
#         dataframe["tq_regime"] = tq_regimes
#         dataframe["tq_dist"] = tq_dist

#         # unpack feature
#         for i in range(9):
#             dataframe[f"tq_f{i}"] = tq_features[:, i]

#         return dataframe

#     def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

#         dataframe.loc[
#             (
#                 (dataframe["tq_regime"] == 1)
#                 & (dataframe["tq_score"] > 0.05)
#                 & (dataframe["tq_dist"] < 1.2)
#                 & (dataframe["volume"] > 0)
#             ),
#             "enter_long"
#         ] = 1

#         return dataframe

#     def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

#         dataframe.loc[
#             (
#                 (dataframe["tq_regime"] == -1)
#                 | (dataframe["tq_score"] < -0.05)
#                 | (dataframe["tq_dist"] > 1.5)
#             ),
#             "exit_long"
#         ] = 1

#         return dataframe