from pathlib import Path

class Config:
    # =========================
    # DATA
    # =========================
    SYMBOL = "btcusdt"
    TIMEFRAME = "1m"
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_PATH = BASE_DIR.joinpath("btc_buffer.csv")
    DATASET_PATH = BASE_DIR.joinpath("dataset.csv")

    # =========================
    # FEATURE
    # =========================
    FEATURE_DIM = 9

    # =========================
    # TURBO QUANT
    # =========================
    USE_TURBO = True
    TQ_LEVELS = 16
    TQ_RANGE = (-3, 3)

    # =========================
    # MODEL
    # =========================
    MODEL_TYPE = "baseline"
    # options:
    # "baseline"
    # "tq"
    # "lstm"
    # "drl"
    # "freqtrade"

    # =========================
    # BUFFER
    # =========================
    BUFFER_SIZE = 200
    MIN_DATA = 50
    
    # cứ 20 nến save 1 lần
    SAVE_INTERVAL = 1