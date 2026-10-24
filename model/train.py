"""Train the LightGBM quantile forecasters and evaluate them against two baselines.

Usage (from the repo root):  python model/train.py

Reads   data/trace_hourly.csv, config.yaml
Writes  model/artifacts/lgbm_q50.txt, lgbm_q70.txt, lgbm_q90.txt, lgbm_q95.txt
        results/model_metrics.csv       accuracy + coverage table (report)
        results/test_forecasts.csv      every test hour: actual, baselines, all quantiles (simulator input)
        results/forecast_test_week.png  actual vs P50 with the P50-P90 band (report / video)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.features import FEATURES, make_features, make_target  # noqa: E402

ART = ROOT / "model" / "artifacts"
RESULTS = ROOT / "results"
QUANTILES = [0.5, 0.7, 0.9, 0.95]
VALID_DAYS = 7  # last week of the training period, used only for early stopping
PARAMS = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=20, feature_fraction=0.9,
              bagging_fraction=0.9, bagging_freq=1, verbose=-1, seed=42)


def pinball(y, pred, q):
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    trace = pd.read_csv(ROOT / "data" / "trace_hourly.csv", parse_dates=["timestamp"])
    test_start = pd.Timestamp(cfg["trace"]["test_start"])

    X, y = make_features(trace), make_target(trace)
    target_time = trace["timestamp"] + pd.Timedelta(hours=1)
    ok = X.notna().all(axis=1) & y.notna()

    # Split by the time of the hour being PREDICTED, so no test hour leaks into training
    is_test = ok & (target_time >= test_start)
    is_train_all = ok & (target_time < test_start)
    valid_start = test_start - pd.Timedelta(days=VALID_DAYS)
    is_valid = is_train_all & (target_time >= valid_start)
    is_train = is_train_all & (target_time < valid_start)
    print(f"train rows {is_train.sum()}  valid rows {is_valid.sum()}  test rows {is_test.sum()}")

    ART.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    dtrain = lgb.Dataset(X[is_train], y[is_train], categorical_feature=["hour", "dow"], free_raw_data=False)
    dvalid = lgb.Dataset(X[is_valid], y[is_valid], reference=dtrain)

    out = pd.DataFrame({"timestamp": target_time[is_test].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "actual": y[is_test],
                        "naive": X.loc[is_test, "lag_1"],             # next hour = this hour
                        "seasonal_naive": X.loc[is_test, "lag_168"]})  # next hour = same hour last week
    for q in QUANTILES:
        params = dict(PARAMS, objective="quantile", alpha=q, metric="quantile")
        booster = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        booster.save_model(str(ART / f"lgbm_q{int(q * 100)}.txt"), num_iteration=booster.best_iteration)
        out[f"q{int(q * 100)}"] = np.maximum(0, booster.predict(X[is_test], num_iteration=booster.best_iteration))
        print(f"q={q:.2f}: best iteration {booster.best_iteration}")
    out.to_csv(RESULTS / "test_forecasts.csv", index=False)

    # Accuracy of point forecasts (P50 vs baselines) and coverage of each quantile
    rows = []
    for name in ["naive", "seasonal_naive", "q50"]:
        e = out["actual"] - out[name]
        rows.append({"model": "LightGBM P50" if name == "q50" else name,
                     "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                     "MAPE_%": (e.abs() / out["actual"]).mean() * 100})
    acc = pd.DataFrame(rows)
    cov = pd.DataFrame([{"quantile": q,
                         "coverage_%": (out["actual"] <= out[f"q{int(q * 100)}"]).mean() * 100,
                         "pinball_loss": pinball(out["actual"], out[f"q{int(q * 100)}"], q)}
                        for q in QUANTILES])
    with open(RESULTS / "model_metrics.csv", "w", newline="") as f:
        acc.round(4).to_csv(f, index=False)
        f.write("\n")
        cov.round(4).to_csv(f, index=False)

    # Report figure: first test week
    wk = out.head(24 * 7)
    t = pd.to_datetime(wk["timestamp"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(t, wk["q50"], wk["q90"], alpha=0.25, label="P50 to P90 band")
    ax.plot(t, wk["q50"], lw=1.2, label="P50 forecast")
    ax.plot(t, wk["actual"], lw=1.2, color="black", label="actual")
    ax.set(title="Next-hour forecast vs actual, first test week", ylabel="req/s")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "forecast_test_week.png", dpi=150)
    plt.close(fig)

    print("\nPoint accuracy on the test period (lower is better):")
    print(acc.round(3).to_string(index=False))
    print("\nQuantile coverage (target: coverage close to the quantile):")
    print(cov.round(3).to_string(index=False))
    print(f"\nwrote {ART}, results/model_metrics.csv, results/test_forecasts.csv, results/forecast_test_week.png")


if __name__ == "__main__":
    main()