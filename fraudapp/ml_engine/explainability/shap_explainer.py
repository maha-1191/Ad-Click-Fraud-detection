import shap
import numpy as np
from collections import defaultdict
from fraudapp.ml_engine.logger import get_logger

logger = get_logger(__name__)

SHAP_EXPLAIN_SAMPLES = 200
SHAP_MAX_FEATURES = 13


def get_latent_feature_label(i: int) -> str:
    """
    Maps latent feature indices to high-level
    behavioral categories.

    Multiple latent features may map to the same category.
    """

    if 1 <= i <= 30:
        return "Click Frequency & Burst Behavior"
    elif 31 <= i <= 60:
        return "Temporal & Session-Based Patterns"
    elif 61 <= i <= 90:
        return "IP-Centric Repetitive Behavior"
    elif 91 <= i <= 120:
        return "Device / OS Interaction Patterns"
    elif 121 <= i <= 150:
        return "App–Channel Usage Anomalies"
    else:
        return "Composite Behavioral Signature"


class SHAPExplainer:
    """
    SHAP explainability for XGBoost trained on
    CNN–RNN latent behavioral embeddings.

    Latent SHAP values are aggregated into
    high-level behavioral categories for
    human interpretability.
    """

    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X: np.ndarray):
        logger.debug(
            "Generating SHAP explanations (behavior-level aggregation)"
        )

        max_rows = min(len(X), SHAP_EXPLAIN_SAMPLES)
        X_sample = X[:max_rows]

        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs = np.abs(shap_values).mean(axis=0)

        top_idx = np.argsort(mean_abs)[-SHAP_MAX_FEATURES:][::-1]

        behavior_shap = defaultdict(float)

        for i in top_idx:
            behavior = get_latent_feature_label(i)
            behavior_shap[behavior] += float(mean_abs[i])

        result = [
            {
                "feature": behavior,
                "mean_shap": round(value, 4)
            }
            for behavior, value in sorted(
                behavior_shap.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        return result

















