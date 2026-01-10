from pathlib import Path

class InferenceConfig:
    """
    Lightweight config for inference-only deployment
    """

    BASE_DIR = Path(__file__).resolve().parents[2]

    # Where trained models are stored
    MODEL_DIR = BASE_DIR / "ml_engine" / "models"

    # Must match training-time sequence length
    SEQUENCE_LENGTH = 10

