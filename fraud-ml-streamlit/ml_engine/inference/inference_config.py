from pathlib import Path

class InferenceConfig:
    # fraud-ml-streamlit/
    BASE_DIR = Path(__file__).resolve().parents[2]

    # fraud-ml-streamlit/artifacts/models
    MODEL_DIR = BASE_DIR / "artifacts" / "models"

    SEQUENCE_LENGTH = 10



