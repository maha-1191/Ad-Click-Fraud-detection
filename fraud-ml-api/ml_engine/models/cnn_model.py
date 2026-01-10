"""
CNN module for Hybrid Deep Learning Technique (HDLT)

Role:
- Extract local/spatial patterns from click feature sequences
- Output learned representations for RNN input

CNN output → RNN input  
"""

import torch
import torch.nn as nn


class ClickCNN(nn.Module):
    """
    CNN for click fraud feature extraction.

    Input shape:
        (batch_size, time_steps, feature_dim)

    Output shape:
        (batch_size, time_steps, cnn_feature_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        cnn_feature_dim: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.cnn_feature_dim = cnn_feature_dim

        # CNN operates on feature channels
        self.conv1 = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=cnn_feature_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=cnn_feature_dim,
            out_channels=cnn_feature_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.batch_norm1 = nn.BatchNorm1d(cnn_feature_dim)
        self.batch_norm2 = nn.BatchNorm1d(cnn_feature_dim)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        #  IMPORTANT: Proper weight initialization
        self._init_weights()

    def _init_weights(self):
        """
        He/Kaiming initialization for Conv1D layers
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Tensor of shape (batch_size, time_steps, feature_dim)

        Returns:
            Tensor of shape (batch_size, time_steps, cnn_feature_dim)
        """

        # (batch, time, features) → (batch, features, time)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # (batch, features, time) → (batch, time, features)
        x = x.permute(0, 2, 1)

        return x


# ---- Compatibility alias ----
CNNFeatureExtractor = ClickCNN
