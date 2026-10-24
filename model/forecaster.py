"""Trained forecaster with the interface the controller expects (see the Phase 0 contract).

    fc = Forecaster("model/artifacts")
    fc.predict(history, quantile=0.9)   # history: DataFrame(timestamp, rps), >= 168 rows
"""
from pathlib import Path

import lightgbm as lgb

try:
    from model.features import MIN_HISTORY, make_features
except ImportError:  # when run from inside model/
    from features import MIN_HISTORY, make_features


class Forecaster:
    def __init__(self, artifact_dir="model/artifacts"):
        d = Path(artifact_dir)
        self.models = {}
        for f in sorted(d.glob("lgbm_q*.txt")):
            q = int(f.stem.split("_q")[1]) / 100
            self.models[q] = lgb.Booster(model_file=str(f))
        if not self.models:
            raise FileNotFoundError(f"no lgbm_q*.txt models in {d} - run model/train.py first")

    def predict(self, history, quantile=0.9):
        """Forecast req/s for the hour AFTER the last row of `history`, at the given quantile."""
        if quantile not in self.models:
            raise ValueError(f"no model for quantile {quantile}; have {sorted(self.models)}")
        h = history[["timestamp", "rps"]].tail(MIN_HISTORY + 24).reset_index(drop=True)
        if len(h) < MIN_HISTORY:
            raise ValueError(f"need at least {MIN_HISTORY} hours of history, got {len(h)}")
        X = make_features(h).iloc[[-1]]
        return max(0.0, float(self.models[quantile].predict(X)[0]))