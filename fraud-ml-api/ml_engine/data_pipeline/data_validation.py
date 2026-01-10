class DataValidationException(Exception):
    pass


class DataValidator:
    """
    Validates RAW TalkingData dataset
    (before preprocessing / feature engineering)
    """

    # --------------------------------------------------
    # RAW TalkingData columns
    # --------------------------------------------------
    RAW_FEATURE_COLUMNS = {
        "ip",
        "app",
        "device",
        "os",
        "channel",
        "click_time",
    }

    TARGET_COLUMN = "is_attributed"

    MIN_ROWS_INFERENCE = 5
    MIN_ROWS_TRAINING = 50

    @staticmethod
    def validate(df, training: bool = False):
        """
        Validate input dataframe.

        Args:
            df (pd.DataFrame): Raw input data
            training (bool): True for training, False for inference
        """

        # -------------------------------
        # 1️⃣ Empty check
        # -------------------------------
        if df is None or df.empty:
            raise DataValidationException("Dataset is empty")

        # -------------------------------
        # 2️⃣ Minimum rows check
        # -------------------------------
        min_rows = (
            DataValidator.MIN_ROWS_TRAINING
            if training
            else DataValidator.MIN_ROWS_INFERENCE
        )

        if len(df) < min_rows:
            raise DataValidationException(
                f"Dataset too small. "
                f"Found {len(df)} rows, "
                f"required at least {min_rows}"
            )

        # -------------------------------
        # 3️⃣ Raw feature columns check
        # -------------------------------
        missing_features = (
            DataValidator.RAW_FEATURE_COLUMNS - set(df.columns)
        )

        if missing_features:
            raise DataValidationException(
                f"Missing required raw columns: {missing_features}"
            )

        # -------------------------------
        # 4️⃣ Target column (training only)
        # -------------------------------
        if training and DataValidator.TARGET_COLUMN not in df.columns:
            raise DataValidationException(
                "Training data must contain 'is_attributed' column"
            )

        # -------------------------------
        # 5️⃣ Basic label sanity (optional but good)
        # -------------------------------
        if training:
            invalid_labels = set(df[DataValidator.TARGET_COLUMN].unique()) - {0, 1}
            if invalid_labels:
                raise DataValidationException(
                    f"Invalid label values found: {invalid_labels}"
                )

        return True


