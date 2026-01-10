"""
End-to-End Fraud Detection Pipeline
CNN → RNN/LSTM → XGBoost (Hybrid)

PHASE 2:
- Real data flow
- No model training
- Safe for Django runtime
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .exceptions import PipelineException
from .logger import get_logger

from .data_pipeline.data_validation import DataValidator
from .data_pipeline.preprocessing import preprocess_data
from .data_pipeline.features import build_features
from .data_pipeline.sequence_builder import build_sequences


logger = get_logger(__name__)


class FraudDetectionPipeline:
    """
    High-level pipeline controller (Phase 2).

    Called from Django views.
    Must remain light and fast.
    """

    def __init__(self):
        logger.info("FraudDetectionPipeline initialized (Phase 2)")

    def run(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute Phase-2 pipeline steps.

        Args:
            raw_df (pd.DataFrame): Uploaded clickstream dataset

        Returns:
            dict: Structured pipeline output
        """

        try:
            logger.info("Pipeline execution started")

            # -------------------------------------------------
            # 1️⃣ Validate input dataset (INFERENCE MODE)
            # -------------------------------------------------
            logger.info("Validating dataset")
            DataValidator.validate(raw_df, training=False)

            # -------------------------------------------------
            # 2️⃣ Preprocessing
            # -------------------------------------------------
            logger.info("Preprocessing data")
            processed_df = preprocess_data(raw_df)

            # -------------------------------------------------
            # 3️⃣ Feature engineering (NO labels expected)
            # -------------------------------------------------
            logger.info("Building features")
            feature_df, _ = build_features(
                processed_df, inference=True
            )

            # -------------------------------------------------
            # 4️⃣ Sequence building
            # -------------------------------------------------
            logger.info("Building sequences")
            X_seq, _, feature_names = build_sequences(
                feature_df,
                label_column="is_attributed",
            )

            logger.info("Pipeline Phase-2 completed successfully")

            return {
                "phase": "phase_2",
                "status": "success",
                "rows_received": int(len(raw_df)),
                "rows_after_processing": int(len(feature_df)),
                "num_sequences": int(len(X_seq)),
                "sequence_shape": tuple(X_seq.shape),
                "feature_count": len(feature_names),
                "message": "Data pipeline completed successfully. Ready for inference.",
            }

        except Exception as e:
            logger.exception("Pipeline failed")
            raise PipelineException(str(e)) from e



