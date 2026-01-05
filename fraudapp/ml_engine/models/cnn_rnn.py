import torch
import torch.nn as nn

from fraudapp.ml_engine.models.cnn_model import ClickCNN
from fraudapp.ml_engine.models.rnn_model import ClickLSTM


class CNNRNNModel(nn.Module):
    """
    Unified Deep Model:
    CNN → RNN → Embedding
    """

    def __init__(
        self,
        feature_dim: int,
        cnn_feature_dim: int = 64,
        lstm_feature_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.cnn = ClickCNN(
            feature_dim=feature_dim,
            cnn_feature_dim=cnn_feature_dim,
            dropout=dropout,
        )

        self.rnn = ClickLSTM(
            cnn_feature_dim=cnn_feature_dim,
            lstm_feature_dim=lstm_feature_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.output_dim = lstm_feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, features)

        Returns:
            embeddings: (batch, lstm_feature_dim)
        """
        x = self.cnn(x)
        x = self.rnn(x)
        return x
