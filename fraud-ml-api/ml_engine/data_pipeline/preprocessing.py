import pandas as pd
import socket
import struct


class PreprocessingException(Exception):
    pass


def ip_to_int(ip):
    """
    Convert IPv4 string to integer.
    Invalid or malformed IPs -> 0
    Example: 192.168.0.101 -> 3232235621
    """
    try:
        return struct.unpack("!I", socket.inet_aton(str(ip)))[0]
    except Exception:
        return 0


class DataPreprocessor:
    """
    Paper-aligned preprocessing for TalkingData Ad-Tracking dataset.

    Input  : Raw clickstream data
    Output : Clean numeric dataframe ready for feature engineering
    """

    TARGET_COLUMN = "is_attributed"

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            # --------------------------------------------------
            # 1️⃣ Normalize column names
            # --------------------------------------------------
            df.columns = (
                df.columns.astype(str)
                .str.strip()
                .str.lower()
            )

            # Optional alias handling
            if "ip_address" in df.columns and "ip" not in df.columns:
                df.rename(columns={"ip_address": "ip"}, inplace=True)

            # --------------------------------------------------
            # 2️⃣ Handle timestamp -> temporal features
            # --------------------------------------------------
            if "click_time" not in df.columns:
                raise PreprocessingException(
                    "Missing required column 'click_time'"
                )

            df["click_time"] = pd.to_datetime(
                df["click_time"], errors="coerce"
            )

            df["click_hour"] = df["click_time"].dt.hour
            df["click_day"] = df["click_time"].dt.day
            df["click_dayofweek"] = df["click_time"].dt.dayofweek

            # Drop raw timestamp after extraction
            df.drop(columns=["click_time"], inplace=True)

            # --------------------------------------------------
            # 3️⃣ IP address normalization
            # --------------------------------------------------
            if "ip" not in df.columns:
                raise PreprocessingException(
                    "Missing required column 'ip'"
                )

            df["ip"] = df["ip"].apply(ip_to_int).astype("int64")

            # --------------------------------------------------
            # 4️⃣ Cast categorical identifiers to numeric
            # --------------------------------------------------
            categorical_cols = ["app", "device", "os", "channel"]

            for col in categorical_cols:
                if col not in df.columns:
                    raise PreprocessingException(
                        f"Missing required column '{col}'"
                    )

                df[col] = pd.to_numeric(
                    df[col], errors="coerce"
                ).fillna(0).astype("int64")

            # --------------------------------------------------
            # 5️⃣ Fill missing numeric values
            # --------------------------------------------------
            numeric_cols = df.select_dtypes(
                include=["int64", "float64"]
            ).columns

            df[numeric_cols] = df[numeric_cols].fillna(0)

            # NOTE:
            # - Target column (is_attributed) is kept if present
            # - Feature engineering happens later

            return df

        except Exception as e:
            raise PreprocessingException(str(e))


# --------------------------------------------------
# Convenience wrapper
# --------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    return DataPreprocessor.preprocess(df)



