"""
model_registry.py
-----------------
Central registry (factory) for ML/DL models used in the pipeline.

Architecture:
CNN → RNN → Embeddings → SMOTE → XGBoost
"""

from fraudapp.ml_engine.logger import get_logger
from fraudapp.ml_engine.training.training_config import TrainingConfig
from fraudapp.ml_engine.models.cnn_rnn import CNNRNNModel
from fraudapp.ml_engine.models.xgb_model import XGBoostFraudClassifier

logger = get_logger(__name__)


class ModelRegistry:
    """

    """

    @staticmethod
    def build_deep_model(
        config: TrainingConfig,
        input_dim: int,
    ) -> CNNRNNModel:
        """
        Build unified CNN + RNN deep model.
        """

        if input_dim <= 0:
            raise ValueError(
                f"Invalid input_dim={input_dim} for deep model"
            )

        logger.debug(f"Building CNN-RNN model (input_dim={input_dim})")

        return CNNRNNModel(
            feature_dim=input_dim,
            cnn_feature_dim=config.CNN_FILTERS,
            lstm_feature_dim=config.RNN_HIDDEN_DIM,
            num_layers=config.RNN_LAYERS,
            dropout=config.DROPOUT_RATE,
        )

    @staticmethod
    def build_xgb(config, scale_pos_weight=1.0):
        return XGBoostFraudClassifier(
            n_estimators=config.XGB_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LR,
            subsample=config.XGB_SUBSAMPLE,
            colsample_bytree=config.XGB_COLSAMPLE,
            random_state=config.RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,  
        )






