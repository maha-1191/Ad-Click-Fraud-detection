try:
    import shap
except ImportError:
    shap = None

import numpy as np
from collections import defaultdict
from fraudapp.ml_engine.logger import get_logger

logger = get_logger(__name__)

SHAP_EXPLAIN_SAMPLES = 200
SHAP_MAX_FEATURES = 13


def get_latent_feature_label(i: int) -> str:
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
    def __init__(self, model):
        """Create a SHAP explainer for `model` if SHAP is available.

        If SHAP is not installed or creating an explainer fails, the
        instance will still be created but `self.explainer` will be
        set to ``None`` and `explain()` will return an empty list.
        """
        self.explainer = None
        if shap is None:
            logger.warning("shap package not installed; explainability disabled")
            return

        # Try common explainer constructors with graceful fallback
        try:
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception:
                # Fall back to the generic API (shap.Explainer) if present
                if hasattr(shap, "Explainer"):
                    self.explainer = shap.Explainer(model)
                else:
                    # As last resort, re-raise to be caught by outer except
                    raise
        except Exception:
            logger.exception("Failed to create SHAP explainer; explainability disabled")
            self.explainer = None

    def explain(self, X: np.ndarray):
        if self.explainer is None:
            logger.warning("SHAP explainer unavailable; returning empty explanation list")
            return []

        max_rows = min(len(X), SHAP_EXPLAIN_SAMPLES)
        X_sample = X[:max_rows]

        shap_values = None
        try:
            shap_values = self.explainer.shap_values(X_sample)
        except Exception:
            # Some SHAP explainers (shap.Explainer) return an Explanation
            # object accessible via `.values` when called directly.
            try:
                expl = self.explainer(X_sample)
                if hasattr(expl, "values"):
                    shap_values = expl.values
                else:
                    shap_values = expl
            except Exception:
                logger.exception("Failed to compute shap values")
                return []

        # Normalize possible return types: list (per-class), Explanation, ndarray
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values
        if isinstance(shap_values, list):
            # if model is binary-class, shap often returns [neg, pos]
            try:
                shap_values = shap_values[1]
            except Exception:
                shap_values = shap_values[0]

        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-SHAP_MAX_FEATURES:][::-1]

        behavior_shap = defaultdict(float)

        for i in top_idx:
            behavior = get_latent_feature_label(i)
            behavior_shap[behavior] += float(mean_abs[i])

        return [
            {"feature": behavior, "mean_shap": round(value, 4)}
            for behavior, value in sorted(
                behavior_shap.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]




















