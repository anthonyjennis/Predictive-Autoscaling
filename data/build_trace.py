"""Turn raw hourly pageviews into the project's traffic trace.

Usage (from the repo root, after fetch_pageviews.py):
    python data/build_trace.py

Reads   data/raw_pageviews.csv, config.yaml
Writes  data/trace_hourly.csv   timestamp, views, rps
        data/live_slice.csv     timestamp, views, rps, phase   (168 'history' hours + 24 'live' hours)
        results/data_full_series.png, results/data_one_week.png, results/data_hour_of_day.png
Updates config.yaml -> trace.test_start
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_pageviews.csv"
TRACE = ROOT / "data" / "trace_hourly.csv"
SLICE = ROOT / "data" / "live_slice.csv"
CONFIG = ROOT / "config.yaml"
RESULTS = ROOT / "results"
HISTORY_HOURS = 168  # the model's longest lag is one week


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    C = cfg["capacity_per_replica_rps"]
    test_days = cfg["trace"]["test_days"]
    peak_replicas = cfg["trace"]["peak_replicas_target"]

    # 1. Load and put on a complete hourly grid
    raw = pd.read_csv(RAW, parse_dates=["timestamp"])
    s = raw.set_index("timestamp")["views"].sort_index()
    s = s[~s.index.duplicated()].asfreq("h")
    missing = int(s.isna().sum())
    s = s.interpolate(limit=3)  # only fill short gaps
    if s.isna().any():
        raise SystemExit(f"{int(s.isna().sum())} hours still missing after filling gaps of up to 3 hours - "
                         "check the raw data before continuing")

    # 2. Train/test split: the last `test_days` full days are the test period
    last_day = s.index.max().normalize()
    test_start = last_day - pd.Timedelta(days=test_days - 1)
    test = s[s.index >= test_start]

    # 3. Scale views -> requests/s so the busiest test hour needs `peak_replicas` replicas
    k = (peak_replicas * C) / test.max()
    trace = pd.DataFrame({"views": s.round().astype(int), "rps": (s * k).round(3)})
    trace.index.name = "timestamp"
    trace.to_csv(TRACE, date_format="%Y-%m-%dT%H:%M:%SZ")

    # 4. Live-run slice: the test day (UTC) with the biggest swing in load, plus one week of history
    rps = trace["rps"]
    best_day, best_swing = None, -1.0
    for day in pd.date_range(test_start, last_day, freq="D"):
        day_vals = rps[day: day + pd.Timedelta(hours=23)]
        has_history = (day - pd.Timedelta(hours=HISTORY_HOURS)) >= rps.index.min()
        if len(day_vals) == 24 and has_history and day_vals.max() - day_vals.min() > best_swing:
            best_day, best_swing = day, day_vals.max() - day_vals.min()
    hist_start = best_day - pd.Timedelta(hours=HISTORY_HOURS)
    sl = trace[hist_start: best_day + pd.Timedelta(hours=23)].copy()
    sl["phase"] = ["history"] * HISTORY_HOURS + ["live"] * 24
    sl.to_csv(SLICE, date_format="%Y-%m-%dT%H:%M:%SZ")

    # 5. Write test_start back into config.yaml (text replace keeps the comments)
    ts = test_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    text = re.sub(r'test_start:\s*(null|"[^"]*")', f'test_start: "{ts}"', CONFIG.read_text())
    CONFIG.write_text(text)

    # 6. Report plots
    RESULTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(rps.index, rps.values, lw=0.6)
    ax.axvline(test_start, color="red", ls="--", lw=1, label="test period starts")
    ax.set(title=f"{cfg['trace']['project']} hourly traffic, scaled to req/s", ylabel="req/s")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / "data_full_series.png", dpi=150); plt.close(fig)

    week = rps[test_start: test_start + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(week.index, week.values, lw=1.2)
    ax.set(title="One week of the test period", ylabel="req/s")
    fig.tight_layout(); fig.savefig(RESULTS / "data_one_week.png", dpi=150); plt.close(fig)

    by_hour = rps.groupby(rps.index.hour).mean()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(by_hour.index, by_hour.values)
    ax.set(title="Average load by hour of day (UTC)", xlabel="hour (UTC)", ylabel="req/s")
    fig.tight_layout(); fig.savefig(RESULTS / "data_hour_of_day.png", dpi=150); plt.close(fig)

    # 7. Summary to paste back
    daily = rps.groupby(rps.index.date)
    swing_ratio = (daily.max() / daily.min()).median()
    print(f"rows: {len(trace)}  ({rps.index.min()} -> {rps.index.max()})")
    print(f"missing hours filled: {missing}")
    print(f"test_start: {ts}   test hours: {len(test)}")
    print(f"scale k: {k:.3e}   peak test rps: {rps[test_start:].max():.1f}   min test rps: {rps[test_start:].min():.1f}")
    print(f"typical daily peak/trough ratio: {swing_ratio:.2f}")
    print(f"replicas needed at test peak / trough: {rps[test_start:].max() / C:.1f} / {rps[test_start:].min() / C:.1f}")
    print(f"live slice: {best_day:%Y-%m-%d} (UTC), swing {best_swing:.1f} req/s -> {SLICE.name}")


if __name__ == "__main__":
    main()