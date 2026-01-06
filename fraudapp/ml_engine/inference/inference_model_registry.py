import joblib
import torch
from fraudapp.ml_engine.models.cnn_rnn import CNNRNNModel
from fraudapp.ml_engine.models.xgb_model import XGBModel

class InferenceModelRegistry:

    @staticmethod
    def load_xgb(model_dir):
        model = XGBModel()
        model.load(model_dir)
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
