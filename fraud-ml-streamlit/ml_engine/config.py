"""
config.py
---------
Central configuration for ML Engine (INFERENCE & EXPLAINABILITY).

NOTE:
- Training hyperparameters are defined in training_config.py
- This config is used ONLY for inference, evaluation, and explainability
"""

from pathlib import Path


class MLConfig:
    BASE_DIR = Path("fraudapp/ml_engine")
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    LOGS_DIR = BASE_DIR / "logs"

    DEFAULT_THRESHOLD = 0.02

    SHAP_BACKGROUND_SAMPLES = 100
    SHAP_EXPLAIN_SAMPLES = 20
    SHAP_MAX_FEATURES = 10

    LIME_NUM_FEATURES = 10
    LIME_NUM_SAMPLES = 1000

    MAX_EXPLAIN_ROWS = 200

    @classmethod
    def ensure_dirs(cls):
        cls.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)




