
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class FeatureEngineeringException(Exception):
    pass


# ======================================================
# features
# ======================================================
FEATURE_COLUMNS = [
    "app",
    "device",
    "os",
    "channel",
    "click_hour",
    "click_day",
    "click_dayofweek",
    "ip_click_count",
    "ip_hour_click_count",
    "ip_day_click_count",
    "app_channel_count",
    "device_os_count",
    "ip_app_count",
    "ip_device_count",
]


class FeatureEngineer:
    """
    FINAL feature engineering for Ad Click Fraud Detection.

    IMPORTANT RULES:
    - ip is NOT a feature
    - ip MUST be present for sequence grouping
    - Feature count = 14 
    """

    TARGET_COLUMN = "is_attributed"

    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            # ---------------- REQUIRED COLUMNS ----------------
            required = {
                "ip", "app", "device", "os", "channel",
                "click_hour", "click_day", "click_dayofweek"
            }
            missing = required - set(df.columns)
            if missing:
                raise FeatureEngineeringException(
                    f"Missing required columns: {missing}"
                )

            # ---------------- IP BEHAVIOR ----------------
            df["ip_click_count"] = df.groupby("ip")["ip"].transform("count")

            df["ip_hour_click_count"] = (
                df.groupby(["ip", "click_hour"])["ip"].transform("count")
            )

            df["ip_day_click_count"] = (
                df.groupby(["ip", "click_day"])["ip"].transform("count")
            )

            # ---------------- COMBINATION ABUSE ----------------
            df["app_channel_count"] = (
                df.groupby(["app", "channel"])["app"].transform("count")
            )

            df["device_os_count"] = (
                df.groupby(["device", "os"])["device"].transform("count")
            )

            df["ip_app_count"] = (
                df.groupby(["ip", "app"])["ip"].transform("count")
            )

            df["ip_device_count"] = (
                df.groupby(["ip", "device"])["ip"].transform("count")
            )

            # ---------------- FINAL FEATURE FRAME ----------------
            X = df[["ip"] + FEATURE_COLUMNS].copy()

            X.replace([np.inf, -np.inf], 0, inplace=True)
            X.fillna(0, inplace=True)

            return X

        except Exception as e:
            raise FeatureEngineeringException(str(e))


# ======================================================
# PIPELINE WRAPPER
# ======================================================
def build_features(
    df: pd.DataFrame,
    inference: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

    df = df.copy()
    y = None

    # ---------------- TARGET ----------------
    if not inference and FeatureEngineer.TARGET_COLUMN in df.columns:
        y = df[FeatureEngineer.TARGET_COLUMN].astype(int)
        df = df.drop(columns=[FeatureEngineer.TARGET_COLUMN])

    # ---------------- FEATURES ----------------
    X = FeatureEngineer.transform(df)

    # HARD ASSERTIONS 
    assert "ip" in X.columns, "ip column missing — sequence builder will fail"
    assert list(X.columns[1:]) == FEATURE_COLUMNS, (
        f"Feature mismatch!\nExpected: {FEATURE_COLUMNS}\nGot: {list(X.columns[1:])}"
    )

    return X, y


