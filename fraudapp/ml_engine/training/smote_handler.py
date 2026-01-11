from collections import Counter

from imblearn.over_sampling import SMOTE
from fraudapp.ml_engine.logger import get_logger

logger = get_logger(__name__)


class SmoteHandler:
    def __init__(self, sampling_strategy=0.5, random_state=42):
        self.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

    def apply(self, X, y):
        counts = Counter(y)
        logger.info(f"Before SMOTE: {dict(counts)}")

        if counts[1] >= counts[0]:
            logger.warning("Fraud is not minority. Skipping SMOTE.")
            return X, y

        X_res, y_res = self.smote.fit_resample(X, y)
        return X_res, y_res

