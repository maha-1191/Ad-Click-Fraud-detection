import numpy as np
import pandas as pd
from typing import Tuple, List


class SequenceBuilderException(Exception):
    pass


class SequenceBuilder:
    """
    Builds fixed-length click sequences for CNN-RNN models.

    Output:
        X_seq            -> (num_sequences, time_steps, feature_dim)
        y_seq            -> (num_sequences,)
        click_index_map  -> List[np.ndarray]
    """

    def __init__(
        self,
        sequence_length: int,
        group_column: str = "ip",
        time_column: str | None = None,
        label_column: str = "is_attributed",
        fraud_ratio_threshold: float = 0.2,   # ✅ IMPORTANT
    ):
        self.sequence_length = sequence_length
        self.group_column = group_column
        self.time_column = time_column
        self.label_column = label_column
        self.fraud_ratio_threshold = fraud_ratio_threshold

    def build(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:

        try:
            df = df.copy()

            # ---------------- SAFETY CHECK ----------------
            if self.group_column not in df.columns:
                raise SequenceBuilderException(
                    f"Group column '{self.group_column}' missing"
                )

            # ---------------- SORT BY TIME ----------------
            if self.time_column and self.time_column in df.columns:
                df = df.sort_values(self.time_column)

            # ---------------- FEATURE COLUMNS ----------------
            feature_cols = [
                col for col in df.columns
                if col not in {self.group_column, self.label_column}
                and pd.api.types.is_numeric_dtype(df[col])
            ]

            if not feature_cols:
                raise SequenceBuilderException(
                    "No numeric feature columns for sequence building"
                )

            sequences = []
            labels = []
            click_index_map: List[np.ndarray] = []

            # ---------------- BUILD SEQUENCES ----------------
            for _, group in df.groupby(self.group_column):

                values = group[feature_cols].values

                label_values = (
                    group[self.label_column].values
                    if self.label_column in group.columns
                    else None
                )

                if len(values) < self.sequence_length:
                    continue

                for i in range(len(values) - self.sequence_length + 1):

                    # ---- INPUT SEQUENCE ----
                    sequences.append(
                        values[i : i + self.sequence_length]
                    )

                    # ---- CLICK INDEX MAP ----
                    click_index_map.append(
                        group.index[i : i + self.sequence_length].to_numpy()
                    )

                    # ---- ✅ FIXED SEQUENCE LABELING ----
                    if label_values is not None:
                        fraud_ratio = np.mean(
                            label_values[i : i + self.sequence_length]
                        )
                        labels.append(
                            int(fraud_ratio >= self.fraud_ratio_threshold)
                        )

            if not sequences:
                raise SequenceBuilderException(
                    "Not enough data to build sequences"
                )

            X_seq = np.asarray(sequences, dtype=np.float32)

            y_seq = (
                np.asarray(labels, dtype=np.int32)
                if labels
                else np.zeros(len(X_seq), dtype=np.int32)
            )

            return X_seq, y_seq, click_index_map

        except Exception as e:
            raise SequenceBuilderException(str(e))


# ======================================================
# PIPELINE WRAPPER (TRAIN + INFERENCE SAFE)
# ======================================================
def build_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    group_column: str = "ip",
    label_column: str = "is_attributed",
):
    """
    Wrapper used by training & inference.
    """

    builder = SequenceBuilder(
        sequence_length=sequence_length,
        group_column=group_column,
        label_column=label_column,
        fraud_ratio_threshold=0.2,  # ✅ DO NOT REMOVE
    )

    return builder.build(df)








