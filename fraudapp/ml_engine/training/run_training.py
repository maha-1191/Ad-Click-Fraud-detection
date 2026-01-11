import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import StratifiedShuffleSplit

from fraudapp.ml_engine.logger import get_logger
from fraudapp.ml_engine.training.training_config import TrainingConfig
from fraudapp.ml_engine.training.model_registry import ModelRegistry
from fraudapp.ml_engine.training.deep_trainer import DeepModelTrainer
from fraudapp.ml_engine.training.smote_handler import SmoteHandler

from fraudapp.ml_engine.data_pipeline.preprocessing import preprocess_data
from fraudapp.ml_engine.data_pipeline.features import build_features
from fraudapp.ml_engine.data_pipeline.sequence_builder import build_sequences
from fraudapp.ml_engine.evaluation import ModelEvaluator

logger = get_logger(__name__)


def main():
    logger.info("Starting FULL training pipeline")

    config = TrainingConfig()
    config.ensure_dirs()

    TRAIN_CSV_PATH = (
        Path(__file__).resolve()
        .parents[1]
        / "data_data"
        / "train.csv"
    )

    df = pd.read_csv(TRAIN_CSV_PATH)

    if "is_attributed" not in df.columns:
        raise ValueError("Training CSV must contain 'is_attributed'")

    df = preprocess_data(df)

    X, y = build_features(df, inference=False)

    df_seq = X.copy()
    df_seq["ip"] = df["ip"].values
    df_seq["is_attributed"] = y.values

    X_seq, y_seq, _ = build_sequences(
        df_seq,
        sequence_length=config.SEQUENCE_LENGTH
    )

    if len(X_seq) == 0:
        raise ValueError("Not enough data to build sequences")

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, val_idx = next(sss.split(X_seq, y_seq))

    X_train, X_val = X_seq[train_idx], X_seq[val_idx]
    y_train_seq, y_val_seq = y_seq[train_idx], y_seq[val_idx]

    logger.info(
        f"Sequences | Train={len(X_train)} | Val={len(X_val)} | "
        f"Fraud Ratio={y_train_seq.mean():.4f}"
    )

    input_dim = X_train.shape[2]

    deep_model = ModelRegistry.build_deep_model(
        config=config,
        input_dim=input_dim
    )

    trainer = DeepModelTrainer(
        deep_model=deep_model,
        device="cpu",
        lr=config.DEEP_LR
    )

    trainer.fit(
        X_train=X_train,
        y_train=y_train_seq,
        X_val=X_val,
        y_val=y_val_seq,
        epochs=config.DEEP_EPOCHS,
        batch_size=config.BATCH_SIZE,
        model_dir=config.MODEL_DIR
    )

    logger.info("CNN-RNN training completed")
    logger.info("Saved deep_model.pt")

    emb_train = trainer.extract_embeddings(X_train)
    emb_val = trainer.extract_embeddings(X_val)

    y_train_emb = y_train_seq[:len(emb_train)]
    y_val_emb = y_val_seq[:len(emb_val)]

    fraud_count = int((y_train_emb == 1).sum())

    if config.SMOTE_RATIO and fraud_count >= 6:
        smote = SmoteHandler(
            sampling_strategy=config.SMOTE_RATIO,
            random_state=config.RANDOM_STATE
        )
        X_bal, y_bal = smote.apply(emb_train, y_train_emb)
    else:
        X_bal, y_bal = emb_train, y_train_emb

    neg = (y_bal == 0).sum()
    pos = (y_bal == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    xgb_model = ModelRegistry.build_xgb(
        config=config,
        scale_pos_weight=scale_pos_weight
    )

    logger.info("XGBoost training started")

    xgb_model.train(
        X_train=X_bal,
        y_train=y_bal,
        X_val=emb_val,
        y_val=y_val_emb
    )

    xgb_model.save(config.MODEL_DIR)
    logger.info("Saved xgb_model.json")

    train_probs = xgb_model.model.predict_proba(X_bal)[:, 1]
    val_probs = xgb_model.model.predict_proba(emb_val)[:, 1]

    train_preds = (train_probs >= 0.5).astype(int)
    val_preds = (val_probs >= 0.5).astype(int)

    train_metrics = ModelEvaluator.evaluate(
        y_true=y_bal,
        y_pred=train_preds,
        y_prob=train_probs
    )

    val_metrics = ModelEvaluator.evaluate(
        y_true=y_val_emb,
        y_pred=val_preds,
        y_prob=val_probs
    )

    logger.info(
        "[XGBOOST METRICS] "
        f"TRAIN | Acc={train_metrics['accuracy']:.4f} | "
        f"Prec={train_metrics['precision']:.4f} | "
        f"Recall={train_metrics['recall']:.4f} | "
        f"F1={train_metrics['f1_score']:.4f} | "
        f"AUC={train_metrics['roc_auc']:.4f}"
    )

    logger.info(
        "[XGBOOST METRICS] "
        f"VAL   | Acc={val_metrics['accuracy']:.4f} | "
        f"Prec={val_metrics['precision']:.4f} | "
        f"Recall={val_metrics['recall']:.4f} | "
        f"F1={val_metrics['f1_score']:.4f} | "
        f"AUC={val_metrics['roc_auc']:.4f}"
    )

    best_f1 = 0.0
    best_threshold = 0.5

    for t in np.arange(0.1, 0.95, 0.05):
        preds = (val_probs >= t).astype(int)
        metrics = ModelEvaluator.evaluate(
            y_true=y_val_emb,
            y_pred=preds,
            y_prob=val_probs
        )

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_threshold = t

    joblib.dump(
        best_threshold,
        config.MODEL_DIR / "threshold.joblib"
    )

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()





