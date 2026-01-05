"""
exceptions.py
-------------
Centralized custom exception hierarchy for the ML engine.

Used across:
- Data validation
- Preprocessing
- Feature engineering
- Training
- Inference
- Explainability
"""


class MLBaseException(Exception):
    """
    Base exception for all ML-related errors.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class PipelineException(MLBaseException):
    """
    Base exception for pipeline-level failures.
    """
    pass


class DataValidationError(PipelineException):
    """
    Raised when input data validation fails.
    """
    pass


class PreprocessingError(PipelineException):
    """
    Raised during preprocessing stage.
    """
    pass


class FeatureEngineeringError(PipelineException):
    """
    Raised during feature engineering.
    """
    pass


class ModelTrainingError(PipelineException):
    """
    Raised during model training.
    """
    pass


class InferenceError(PipelineException):
    """
    Raised during model inference / prediction.
    """
    pass


class ExplainabilityError(PipelineException):
    """
    Raised during SHAP / LIME explainability.
    """
    pass

