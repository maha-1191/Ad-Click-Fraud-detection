import torch
from ml_engine.models.cnn_rnn import CNNRNNModel
from ml_engine.models.xgb_model import XGBModel


class InferenceModelRegistry:
    """
    Load-only registry for inference (no training code)
    """

    @staticmethod
    def load_xgb(model_dir):
        model = XGBModel()
        model.load(model_dir)
        if not hasattr(model.model, "use_label_encoder"):
            model.model.use_label_encoder = False

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
