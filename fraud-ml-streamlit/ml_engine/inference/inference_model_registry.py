import joblib
import torch
from ml_engine.models.cnn_rnn import CNNRNNModel


class InferenceModelRegistry:

    @staticmethod
    def load_xgb(model_dir):
        # Load sklearn XGBClassifier (same as Django)
        model = joblib.load(model_dir / "xgb.joblib")

        # sklearn / xgboost compatibility
        if hasattr(model, "use_label_encoder"):
            model.use_label_encoder = False

        return model

    @staticmethod
    def load_deep(model_dir, input_dim):
        model = CNNRNNModel(input_dim)
        model.load_state_dict(
            torch.load(
                model_dir / "deep_model.pt",
                map_location="cpu"
            )
        )
        model.eval()
        return model



