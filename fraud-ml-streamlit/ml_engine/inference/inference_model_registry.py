
import joblib
import torch
from ml_engine.models.cnn_rnn import CNNRNNModel


class InferenceModelRegistry:

    @staticmethod
    def load_xgb(model_dir):
        model_path = model_dir / "xgb.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"xgb.joblib not found at {model_path}")

        model = joblib.load(model_path)

        # REQUIRED for backward compatibility with old-trained models
        if not hasattr(model, "use_label_encoder"):
            setattr(model, "use_label_encoder", False)

        return model

    @staticmethod
    def load_deep(model_dir, input_dim):
        model_path = model_dir / "deep_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"deep_model.pt not found at {model_path}")

        model = CNNRNNModel(input_dim)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model




