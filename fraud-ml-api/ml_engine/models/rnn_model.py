"""
RNN (LSTM) module for Hybrid Deep Learning Technique (HDLT)

Role:
- Capture temporal dependencies in CNN-extracted click features
- Output sequence-aware embeddings for XGBoost classifier

Pipeline:
Raw Clicks → CNN → LSTM → XGBoost
"""

import torch
import torch.nn as nn


class ClickLSTM(nn.Module):
    """
    LSTM network for temporal modeling of click sequences.

    Input shape:
        (batch_size, time_steps, cnn_feature_dim)

    Output shape:
        (batch_size, lstm_feature_dim)
    """

    def __init__(
        self,
        cnn_feature_dim: int,
        lstm_feature_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.cnn_feature_dim = cnn_feature_dim
        self.lstm_feature_dim = lstm_feature_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_feature_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.output_dim = (
            lstm_feature_dim * 2 if bidirectional else lstm_feature_dim
        )

        self.layer_norm = nn.LayerNorm(self.output_dim)

        # IMPORTANT: Proper weight initialization
        self._init_weights()

    def _init_weights(self):
        """
        Xavier initialization for LSTM weights
        """
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Tensor of shape (batch_size, time_steps, cnn_feature_dim)

        Returns:
            Tensor of shape (batch_size, lstm_feature_dim)
        """

        # outputs -> (batch, time, hidden)
        # hidden  -> (num_layers * directions, batch, hidden)
        outputs, (hidden, _) = self.lstm(x)

        if self.bidirectional:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            final_hidden = torch.cat(
                (forward_hidden, backward_hidden), dim=1
            )
        else:
            final_hidden = hidden[-1]

        final_hidden = self.layer_norm(final_hidden)

        return final_hidden


# -------------------------------------------------
# Compatibility alias
# -------------------------------------------------
RNNEncoder = ClickLSTM
