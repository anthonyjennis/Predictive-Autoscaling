"""Charts + checks for the full-length live runs (report section: live results).

Usage (from the repo root, after summarise_live.py):  python sim/plot_live.py

Reads   results/live_<policy>_YYYYmmdd-HHMM.csv   (newest per policy, mock runs ignored)
        results/sim_hourly.csv                    (simulator decisions for the same trace day)
        results/locust_<policy>_stats_history.csv (latency over time, from --csv-full-history)
        results/locust_<policy>_failures.csv      (what the failed requests were)
Writes  results/live_runs.png      live replicas vs simulated replicas vs what the load needed, per policy
        results/live_latency.png   p95 latency over the run, per policy (failures marked)
        results/live_vs_sim.csv    does the live system make the same decisions as the simulator?
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
POLICIES = ["static", "reactive", "predictive", "hybrid"]


def newest_log(policy):
    logs = sorted(f for f in RESULTS.glob(f"live_{policy}_*.csv")
                  if re.fullmatch(rf"live_{policy}_\d{{8}}-\d{{4}}\.csv", f.name))
    return pd.read_csv(logs[-1], parse_dates=["timestamp"]) if logs else None


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    C, step_s = cfg["capacity_per_replica_rps"], cfg["step_seconds"]
    sim = pd.read_csv(RESULTS / "sim_hourly.csv")
    sim["timestamp"] = pd.to_datetime(sim["timestamp"], utc=True)
    live = {p: df for p in POLICIES if (df := newest_log(p)) is not None}
    if not live:
        raise SystemExit("no live_<policy>_YYYYmmdd-HHMM.csv files in results/")
    for df in live.values():
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # 1. Live vs simulated decisions for the same trace day
    rows = []
    fig, axes = plt.subplots(len(live), 1, figsize=(10, 2.6 * len(live)), sharex=True, sharey=True, squeeze=False)
    for ax, (p, df) in zip(axes[:, 0], live.items()):
        s = sim[sim["policy"] == p].set_index("timestamp").reindex(df["timestamp"])
        hours = df["timestamp"].dt.hour
        ax.plot(hours, df["actual_rps"] / C, color="black", lw=1.2, label="replicas needed (measured load / C)")
        ax.step(hours, df["replicas"], where="mid", lw=2, label="live replicas")
        if s["replicas"].notna().any():
            ax.step(hours, s["replicas"].values, where="mid", lw=1.2, ls="--", label="simulator replicas")
        over = df["violation_share"] > 0
        ax.scatter(hours[over], df.loc[over, "replicas"], color="red", s=30, zorder=3, label="over capacity")
        ax.set_title(p, loc="left", fontsize=10)
        ax.set_ylabel("replicas")
        rows.append({
            "policy": p,
            "hours": len(df),
            "same_replicas_as_sim_%": round((df["replicas"].values == s["replicas"].values).mean() * 100, 1),
            "within_1_replica_%": round((np.abs(df["replicas"].values - s["replicas"].values) <= 1).mean() * 100, 1),
            "live_replica_hours": int(df["replicas"].sum()),
            "sim_replica_hours": int(s["replicas"].sum()) if s["replicas"].notna().all() else np.nan,
            "live_cost_usd": round(df["cost"].sum(), 3),
            "sim_cost_usd": round(s["cost"].sum(), 3),
            "max_load_error_vs_trace_rps": round((df["actual_rps"].values - s["actual_rps"].values).__abs__().max(), 2),
        })
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=4)
    axes[-1, 0].set_xlabel(f"trace hour of {df['timestamp'].iloc[0]:%Y-%m-%d} (1 hour = {step_s} s live)")
    fig.suptitle("Live runs on EC2: provisioned replicas vs what the measured load needed")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(RESULTS / "live_runs.png", dpi=150)
    plt.close(fig)
    comp = pd.DataFrame(rows)
    comp.to_csv(RESULTS / "live_vs_sim.csv", index=False)

    # 2. Latency over the run (Locust's p95 over a rolling ~10 s window)
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    for p in live:
        f = RESULTS / f"locust_{p}_stats_history.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        h = h[(h["Name"] == "Aggregated") & (h["User Count"] > 0)].copy()
        h["p95"] = pd.to_numeric(h["95%"], errors="coerce")
        h["trace_hour"] = (h["Timestamp"] - h["Timestamp"].iloc[0]) / step_s
        line, = ax.plot(h["trace_hour"], h["p95"], lw=1, label=f"{p} p95")
        fail = h["Total Failure Count"].diff().fillna(0) > 0
        ax.scatter(h.loc[fail, "trace_hour"], h.loc[fail, "p95"], marker="x", color=line.get_color(), s=40)
        plotted = True
    if plotted:
        ax.axhline(cfg["sla_ms"], color="grey", ls=":", label=f"SLA {cfg['sla_ms']} ms")
        ax.set_yscale("log")
        ax.set(xlabel="trace hour (approx., from Locust start)", ylabel="p95 latency, ms (log)",
               title="Latency during the live runs (x = moment a request failed)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "live_latency.png", dpi=150)
    plt.close(fig)

    # 3. What the failures were
    pd.set_option("display.width", 200)
    print("Live vs simulator, same trace day:")
    print(comp.to_string(index=False))
    print("\nFailed requests by error type:")
    for p in live:
        f = RESULTS / f"locust_{p}_failures.csv"
        if f.exists():
            fl = pd.read_csv(f)
            for _, r in fl.iterrows():
                print(f"  {p:10s} {int(r['Occurrences']):4d} x  {str(r['Error'])[:110]}")
            if fl.empty:
                print(f"  {p:10s}    0")
    print("\nwrote results/live_runs.png, live_latency.png, live_vs_sim.csv")


if __name__ == "__main__":
    main()