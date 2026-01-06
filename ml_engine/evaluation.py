"""
evaluation.py
-------------
Model evaluation utilities for Ad Click Fraud Detection.

Designed for EXTREMELY IMBALANCED datasets.
"""

from typing import Dict
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from fraudapp.ml_engine.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Computes evaluation metrics for fraud detection.
    """

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> Dict[str, float]:
        """
        Evaluate classification performance.

        Returns metrics suitable for highly imbalanced fraud datasets.
        """

        

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        metrics = {
            # ⚠️ Accuracy is NOT primary in fraud detection
            "accuracy": accuracy_score(y_true, y_pred),

            # Fraud-focused metrics
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),

            # Additional critical metrics
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,

            # Counts (VERY useful for debugging & reports)
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }

        # ROC-AUC (probability-based)
        if y_prob is not None and len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        else:
            metrics["roc_auc"] = None

        

        return metrics

