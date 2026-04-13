import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy


STRATEGY_DIR = Path(__file__).resolve().parent
if str(STRATEGY_DIR) not in sys.path:
    sys.path.append(str(STRATEGY_DIR))

from turboquant_core import TurboQuant


class MyTurboStrategy(IStrategy):
    """
    TurboQuant-integrated smoke-test strategy.
    Entry/exit uses compressed regime columns (tq_*), not raw RSI thresholds.
    """

    INTERFACE_VERSION = 3

    # Keep risk controls simple for first test pass.
    minimal_roi = {
        "0": 0.02,
        "30": 0.01,
        "60": 0
    }
    stoploss = -0.10
    timeframe = "15m"

    process_only_new_candles = True
    startup_candle_count = 35

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.tq = TurboQuant(feature_dim=3, levels=16, value_range=(-1.0, 1.0))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        dataframe["ret_1"] = dataframe["close"].pct_change().fillna(0.0)
        dataframe["rsi_norm"] = (dataframe["rsi"].fillna(50.0) - 50.0) / 50.0

        macd_scale = dataframe["close"].replace(0, np.nan).ffill().bfill().fillna(1.0)
        dataframe["macdhist_norm"] = (dataframe["macdhist"].fillna(0.0) / macd_scale).clip(-1.0, 1.0)

        features = dataframe[["rsi_norm", "macdhist_norm", "ret_1"]].fillna(0.0).to_numpy(dtype=float)
        quantized = self.tq.quantize_batch(features)

        dataframe["tq_code"] = quantized["codes"]
        dataframe["tq_dist"] = quantized["error_norm"]
        dataframe["tq_regime"] = quantized["regime"]
        dataframe["tq_score"] = quantized["score"]
        dataframe["tq_rsi_hat"] = quantized["x_hat"][:, 0]
        dataframe["tq_macdhist_hat"] = quantized["x_hat"][:, 1]
        dataframe["tq_ret_hat"] = quantized["x_hat"][:, 2]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["tq_rsi_hat"] < -0.10)
                & (dataframe["tq_macdhist_hat"] > -0.05)
                & (dataframe["tq_score"] > 0.05)
                & (dataframe["tq_dist"] < 1.25)
                & (dataframe["volume"] > 0)
            ),
            "enter_long"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe["tq_rsi_hat"] > 0.10)
                    | (dataframe["tq_score"] < -0.05)
                    | (dataframe["tq_dist"] > 1.50)
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long"
        ] = 1

        return dataframe
