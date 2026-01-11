import joblib
import torch
from ml_engine.models.cnn_rnn import CNNRNNModel


class InferenceModelRegistry:

    @staticmethod
    def load_xgb(model_dir):
        model_path = model_dir / "xgb.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"XGB model not found at {model_path}")

        model = joblib.load(model_path)

        # sklearn / xgboost compatibility
        if hasattr(model, "use_label_encoder"):
            model.use_label_encoder = False

        return model

    @staticmethod
    def load_deep(model_dir, input_dim):
        model_path = model_dir / "deep_model.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Deep model not found at {model_path}")

        model = CNNRNNModel(input_dim)

        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        model.eval()
        return model




