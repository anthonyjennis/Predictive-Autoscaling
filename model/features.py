"""Feature builder shared by training (model/train.py) and live prediction (model/forecaster.py).

Row t of the output describes the moment just after hour t has been observed; the target is the
load of hour t+1. Using one function for both paths means training and prediction can't drift apart.
"""
import pandas as pd

# feature name -> how many rows back from t (lag_k = load at hour t+1-k)
LAGS = {"lag_1": 0, "lag_2": 1, "lag_3": 2, "lag_24": 23, "lag_168": 167}
FEATURES = list(LAGS) + ["roll_mean_24", "hour", "dow"]
MIN_HISTORY = 168  # rows needed for the longest lag


def make_features(df):
    """df: columns timestamp (hourly, sorted) and rps. Returns a DataFrame with FEATURES, same index."""
    s = df["rps"].astype(float)
    X = pd.DataFrame(index=df.index)
    for name, back in LAGS.items():
        X[name] = s.shift(back)
    X["roll_mean_24"] = s.rolling(24).mean()
    target_time = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=1)
    X["hour"] = target_time.dt.hour
    X["dow"] = target_time.dt.dayofweek
    return X[FEATURES]


def make_target(df):
    """Load of the next hour (NaN on the last row)."""
    return df["rps"].astype(float).shift(-1)