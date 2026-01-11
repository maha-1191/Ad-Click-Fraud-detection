from typing import Dict, Any
from collections import defaultdict

import pandas as pd
import torch
import numpy as np
import joblib
import socket
import struct

from fraudapp.ml_engine.logger import get_logger
from fraudapp.ml_engine.training.training_config import TrainingConfig
from fraudapp.ml_engine.training.model_registry import ModelRegistry
from fraudapp.ml_engine.data_pipeline.data_validation import DataValidator
from fraudapp.ml_engine.data_pipeline.preprocessing import preprocess_data
from fraudapp.ml_engine.data_pipeline.features import build_features
from fraudapp.ml_engine.data_pipeline.sequence_builder import build_sequences
from fraudapp.ml_engine.explainability.shap_explainer import SHAPExplainer

logger = get_logger(__name__)

ASSUMED_CPC_INR = 5.0

INFERENCE_BLOCK_THRESHOLD = 0.03
HIGH_RISK_THRESHOLD = 0.03
MEDIUM_RISK_THRESHOLD = 0.05


class FraudPredictor:
    """
    SEQUENCE-LEVEL FRAUD DETECTION ENGINE

    CSV
      → Preprocess
      → Feature Engineering
      → IP-based Sequence Builder
      → CNN–RNN (sequence embeddings)
      → XGBoost (fraud probability)
      → Click-level aggregation for dashboard
      → SHAP explainability
    """

    def __init__(self):
        self.config = TrainingConfig()
        self.deep_model = None

        assert (self.config.MODEL_DIR / "deep_model.pt").exists()
        assert (self.config.MODEL_DIR / "threshold.joblib").exists()

        logger.debug("Loading trained XGBoost model")
        self.xgb_model = ModelRegistry.build_xgb(self.config)
        self.xgb_model.load(self.config.MODEL_DIR)
        if not hasattr(self.xgb_model.model, "use_label_encoder"):
            self.xgb_model.model.use_label_encoder = False

        self.trained_threshold = joblib.load(
            self.config.MODEL_DIR / "threshold.joblib"
        )

    def _load_deep_model(self, input_dim: int):
        logger.debug("Loading CNN–RNN model")
        self.deep_model = ModelRegistry.build_deep_model(
            self.config, input_dim
        )
        self.deep_model.load_state_dict(
            torch.load(
                self.config.MODEL_DIR / "deep_model.pt",
                map_location="cpu"
            )
        )
        self.deep_model.eval()

    @staticmethod
    def _normalize_ip(ip_value):
        if pd.isna(ip_value):
            return None

        if isinstance(ip_value, str):
            return ip_value.strip()

        try:
            ip_int = int(ip_value)
            if 0 < ip_int <= 0xFFFFFFFF:
                return socket.inet_ntoa(struct.pack("!I", ip_int))
        except Exception:
            pass

        return str(ip_value)

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:

        DataValidator.validate(df, training=False)
        total_clicks = len(df)

        df = preprocess_data(df)

        X, _ = build_features(df, inference=True)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        df_seq = X.copy()
        df_seq["ip"] = df["ip"].values
        df_seq["click_hour"] = df["click_hour"].values

        X_seq, _, click_index_map = build_sequences(
            df_seq,
            sequence_length=self.config.SEQUENCE_LENGTH
        )

        if len(X_seq) == 0:
            raise ValueError("No sequences generated")

        if self.deep_model is None:
            self._load_deep_model(X_seq.shape[2])

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(X_seq), 512):
                batch = torch.from_numpy(
                    X_seq[i:i + 512]
                ).float()
                embeddings.append(
                    self.deep_model(batch).numpy()
                )

        embeddings = np.vstack(embeddings)

        probs = self.xgb_model.model.predict_proba(embeddings)[:, 1]

        total_sequences = len(probs)
        fraud_sequences = int((probs >= INFERENCE_BLOCK_THRESHOLD).sum())

        fraud_click_ids = set()

        for prob, click_ids in zip(probs, click_index_map):
            if prob >= INFERENCE_BLOCK_THRESHOLD:
                fraud_click_ids.update(click_ids)

        fraud_clicks = len(fraud_click_ids)
        legit_clicks = total_clicks - fraud_clicks

        ip_stats = defaultdict(lambda: {
            "fraud_clicks": set(),
            "risk_sum": 0.0
        })

        for prob, click_ids in zip(probs, click_index_map):
            if prob < INFERENCE_BLOCK_THRESHOLD:
                continue

            for idx in click_ids:
                ip = self._normalize_ip(df.iloc[idx]["ip"])
                if not ip:
                    continue

                ip_stats[ip]["fraud_clicks"].add(idx)
                ip_stats[ip]["risk_sum"] += prob

        ip_risk = []

        for ip, v in ip_stats.items():
            fraud_click_count = len(v["fraud_clicks"])
            if fraud_click_count == 0:
                continue

            avg_risk = v["risk_sum"] / fraud_click_count

            if avg_risk >= HIGH_RISK_THRESHOLD:
                level, action = "HIGH", "BLOCK"
            elif avg_risk >= MEDIUM_RISK_THRESHOLD:
                level, action = "MEDIUM", "THROTTLE"
            else:
                continue

            ip_risk.append({
                "ip": ip,
                "avg_risk_score": round(avg_risk, 3),
                "fraud_clicks": fraud_click_count,
                "risk_level": level,
                "recommended_action": action,
            })

        ip_risk.sort(key=lambda x: x["avg_risk_score"], reverse=True)

        time_stats = {
            hour: {"total_clicks": 0, "fraud_clicks": 0}
            for hour in range(24)
        }

        fraud_click_ids = set()
        for prob, click_ids in zip(probs, click_index_map):
            if prob >= INFERENCE_BLOCK_THRESHOLD:
                fraud_click_ids.update(click_ids)

        for idx, row in df.iterrows():
            hour = int(row["click_hour"])
            time_stats[hour]["total_clicks"] += 1
            if idx in fraud_click_ids:
                time_stats[hour]["fraud_clicks"] += 1

        time_trends = [
            {
                "hour": hour,
                "total_clicks": time_stats[hour]["total_clicks"],
                "fraud_clicks": time_stats[hour]["fraud_clicks"],
            }
            for hour in range(24)
        ]

        shap_summary = SHAPExplainer(
            self.xgb_model.model
        ).explain(embeddings)

        return {
            "summary": {
                "total_clicks": total_clicks,
                "fraud_clicks": fraud_clicks,
                "legit_clicks": legit_clicks,
                "total_sequences": total_sequences,
                "fraud_sequences": fraud_sequences,
                "fraud_ratio": round(
                    fraud_clicks / total_clicks, 4
                ) if total_clicks > 0 else 0.0,
                "inference_threshold": INFERENCE_BLOCK_THRESHOLD,
            },
            "ip_risk": ip_risk[:50],
            "time_trends": time_trends,
            "business_impact": {
                "assumed_cpc_inr": ASSUMED_CPC_INR,
                "estimated_fraud_waste_inr": round(
                    fraud_clicks * ASSUMED_CPC_INR, 2
                ),
            },
            "shap_summary": shap_summary,
        }











   















