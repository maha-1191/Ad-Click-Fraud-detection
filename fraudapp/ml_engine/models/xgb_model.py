import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class XGBoostFraudClassifier:
    """
    XGBoost classifier for fraud detection
    (Robust to single-class validation sets)
    """
    def __init__(
        self,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=1.0,   
    ):
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,  
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    
        
        self.is_trained = False

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    def train(self, X_train, y_train, X_val, y_val):

        use_early_stop = len(np.unique(y_val)) > 1

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)] if use_early_stop else None,
            early_stopping_rounds=30 if use_early_stop else None,
            verbose=False,
        )

        self.is_trained = True

        # Predictions
        train_probs = self.model.predict_proba(X_train)[:, 1]
        val_probs = self.model.predict_proba(X_val)[:, 1]

        train_preds = (train_probs >= 0.5).astype(int)
        val_preds = (val_probs >= 0.5).astype(int)

        metrics = {
            "train_accuracy": accuracy_score(y_train, train_preds),
            "val_accuracy": accuracy_score(y_val, val_preds),

            "train_precision": precision_score(y_train, train_preds, zero_division=0),
            "val_precision": precision_score(y_val, val_preds, zero_division=0),

            "train_recall": recall_score(y_train, train_preds, zero_division=0),
            "val_recall": recall_score(y_val, val_preds, zero_division=0),

            "train_f1": f1_score(y_train, train_preds, zero_division=0),
            "val_f1": f1_score(y_val, val_preds, zero_division=0),
        }

        # ROC-AUC (SAFE)
        if len(np.unique(y_train)) > 1:
            metrics["train_auc"] = roc_auc_score(y_train, train_probs)
        else:
            metrics["train_auc"] = None

        if len(np.unique(y_val)) > 1:
            metrics["val_auc"] = roc_auc_score(y_val, val_probs)
        else:
            metrics["val_auc"] = None

        return metrics
    
    def predict(self, X, threshold=0.6):
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        return preds, probs
    def save(self, model_dir: Path):
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.get_booster().save_model(
            str(model_dir / "xgb_model.json")
    )

    def load(self, model_dir: Path):
        self.model = xgb.XGBClassifier()
        self.model.load_model(
             str(model_dir / "xgb_model.json")
        )
        self.is_trained = True




