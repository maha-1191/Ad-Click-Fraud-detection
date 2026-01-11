from pathlib import Path


class TrainingConfig:
    BASE_DIR = Path("fraudapp/ml_engine")
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "artifacts"
    LOG_DIR = BASE_DIR / "logs"
    SEQUENCE_LENGTH = 10

    @classmethod
    def ensure_dirs(cls):
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

    RANDOM_STATE = 42
    BATCH_SIZE = 512
    DROPOUT_RATE = 0.3

    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 3

    RNN_HIDDEN_DIM = 128
    RNN_LAYERS = 2

    DEEP_EPOCHS = 5
    DEEP_LR = 1e-3

    SMOTE_RATIO = 0.2

    XGB_ESTIMATORS = 300
    XGB_MAX_DEPTH = 6
    XGB_LR = 0.05
    XGB_SUBSAMPLE = 0.8
    XGB_COLSAMPLE = 0.8



