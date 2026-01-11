from pathlib import Path

class InferenceConfig:
    BASE_DIR = Path(__file__).resolve().parents[2]
    MODEL_DIR = BASE_DIR / "ml_engine" / "artifacts"
    SEQUENCE_LENGTH = 10

