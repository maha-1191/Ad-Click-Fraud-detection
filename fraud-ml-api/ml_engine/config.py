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
    # --------------------------------------------------
    # BASE PATHS
    # --------------------------------------------------
    BASE_DIR = Path("fraudapp/ml_engine")
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    LOGS_DIR = BASE_DIR / "logs"

    # --------------------------------------------------
    # INFERENCE CONFIG (FRAUD DETECTION)
    # --------------------------------------------------
    # ⚠️ Low threshold is REQUIRED for extreme imbalance
    DEFAULT_THRESHOLD = 0.02

    # --------------------------------------------------
    # EXPLAINABILITY CONFIG
    # --------------------------------------------------

    # 🔹 SHAP (TreeExplainer on XGBoost)
    SHAP_BACKGROUND_SAMPLES = 100     # small background for speed
    SHAP_EXPLAIN_SAMPLES = 20         # rows to explain per request
    SHAP_MAX_FEATURES = 10            # top features shown

    # 🔹 LIME (optional / future)
    LIME_NUM_FEATURES = 10
    LIME_NUM_SAMPLES = 1000

    # --------------------------------------------------
    # SAFETY LIMITS (INFERENCE)
    # --------------------------------------------------
    MAX_EXPLAIN_ROWS = 200             # hard cap to avoid overload

    @classmethod
    def ensure_dirs(cls):
        cls.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)



