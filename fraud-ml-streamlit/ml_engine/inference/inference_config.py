from pathlib import Path

class InferenceConfig:
    BASE_DIR = Path(__file__).resolve().parents[2]

    # POINT TO DJANGO ARTIFACTS (SOURCE OF TRUTH)
    MODEL_DIR = (
        BASE_DIR
        / ".."
        / ".."
        / "fraudapp"
        / "ml_engine"
        / "artifacts"
    )

    SEQUENCE_LENGTH = 10
