import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from fraudapp.ml_engine.logger import get_logger

logger = get_logger(__name__)


class DeepModelTrainer:
    """
    Trainer for unified CNN + RNN deep model (SUPERVISED).

    Trains CNN-RNN using a temporary classifier head.
    Only CNN-RNN weights are saved (classifier is discarded).
    """

    def __init__(
        self,
        deep_model,
        device="cpu",
        lr=1e-3,
    ):
        self.device = device
        self.model = deep_model.to(device)

        # 🔥 Temporary classifier head (TRAINING ONLY)
        self.classifier = nn.Linear(
            deep_model.output_dim, 1
        ).to(device)

        # Optimizer (CNN + RNN + classifier)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) +
            list(self.classifier.parameters()),
            lr=lr,
        )

        # Loss will be initialized later (needs y_train)
        self.criterion = None

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=10,
        batch_size=128,
        model_dir: Path | None = None,
    ):
        logger.info("Training unified CNN-RNN deep model")

        # ------------------------------
        # 🔥 CLASS IMBALANCE HANDLING
        # ------------------------------
        pos = float(y_train.sum())
        neg = float(len(y_train) - pos)

        pos_weight = neg / max(pos, 1.0)

        logger.info(
            f"Deep model pos_weight={pos_weight:.2f} "
            f"(pos={int(pos)}, neg={int(neg)})"
        )

        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device)
        )

        # ------------------------------
        # DATA LOADERS
        # ------------------------------
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        # NOTE: validation loader currently unused for loss
        # (we rely on XGBoost for final validation)
        # Keeping it for future extensions
        _ = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            ),
            batch_size=batch_size,
        )

        # ------------------------------
        # TRAIN LOOP
        # ------------------------------
        for epoch in range(epochs):
            self.model.train()
            self.classifier.train()

            total_loss = 0.0

            for X, y in train_loader:
                X = X.to(self.device)
                y = y.to(self.device).unsqueeze(1)

                self.optimizer.zero_grad()

                embeddings = self.model(X)
                logits = self.classifier(embeddings)

                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_loss:.4f}"
            )

        # ------------------------------
        # SAVE TRAINED DEEP MODEL ONLY
        # ------------------------------
        if model_dir:
            model_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                self.model.state_dict(),
                model_dir / "deep_model.pt",
            )

            logger.info("Saved deep_model.pt")

    # --------------------------------------------------
    # EMBEDDING EXTRACTION
    # --------------------------------------------------
    def extract_embeddings(self, X, batch_size=1024):
        """
        Extract embeddings in batches (memory-safe).
        Used for XGBoost training & inference.
        """
        self.model.eval()

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i : i + batch_size]

                batch_tensor = torch.tensor(
                    batch, dtype=torch.float32
                ).to(self.device)

                emb = self.model(batch_tensor)
                all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)





