"""
training_config.py
------------------
Centralized configuration for training.

Defines all hyperparameters to ensure
reproducibility and paper alignment.

NOTE:
- CNN and RNN are trained JOINTLY
- Input dimensions are detected at runtime
"""

from pathlib import Path


class TrainingConfig:
    # --------------------------------------------------
    # PATHS
    # --------------------------------------------------
    BASE_DIR = Path("fraudapp/ml_engine")
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "artifacts"
    LOG_DIR = BASE_DIR / "logs"
    SEQUENCE_LENGTH = 10


    @classmethod
    def ensure_dirs(cls):
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # GENERAL
    # --------------------------------------------------
    RANDOM_STATE = 42
    BATCH_SIZE = 512
    DROPOUT_RATE = 0.3

    # --------------------------------------------------
    # CNN CONFIG
    # --------------------------------------------------
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 3

    # --------------------------------------------------
    # RNN CONFIG
    # --------------------------------------------------
    RNN_HIDDEN_DIM = 128
    RNN_LAYERS = 2

    # --------------------------------------------------
    # JOINT CNN + RNN TRAINING
    # --------------------------------------------------
    DEEP_EPOCHS = 5          # 🔥 replaces CNN_EPOCHS & RNN_EPOCHS
    DEEP_LR = 1e-3           # 🔥 shared learning rate

    # --------------------------------------------------
    # SMOTE CONFIG (training only)
    # --------------------------------------------------
    SMOTE_RATIO = 0.2     # safer for extreme imbalance

    # --------------------------------------------------
    # XGBOOST CONFIG
    # --------------------------------------------------
    XGB_ESTIMATORS = 300
    XGB_MAX_DEPTH = 6
    XGB_LR = 0.05
    XGB_SUBSAMPLE = 0.8
    XGB_COLSAMPLE = 0.8


