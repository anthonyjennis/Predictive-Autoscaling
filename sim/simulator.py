"""Offline evaluation of all four policies over the full 21-day test period.

Usage (from the repo root, after model/train.py):  python sim/simulator.py

For every test hour, each policy decides that hour's replicas using only what was known at the end of
the previous hour, via the SAME policies.decide() and step_metrics() the live controller uses.

Reads   config.yaml, data/trace_hourly.csv, results/test_forecasts.csv
Writes  results/summary_sim.csv        main table: 4 policies x all metrics (q = 0.9, margin = 1.1)
        results/sweep_sim.csv          cost vs violations across margins (reactive) and quantiles (predictive/hybrid)
        results/tradeoff_sweep.png     the trade-off chart ("what if the prediction is wrong?")
        results/policies_week.png      one test week, load vs replicas, one panel per policy
        results/mispredictions.csv     the 5 worst P90 under-forecasts and what each policy did
        results/sim_hourly.csv         every hour x policy (for any extra charts)
"""
import copy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controller"))
from policies import POLICIES, decide, static_replicas, step_metrics  # noqa: E402

RESULTS = ROOT / "results"


def run(policy, fc, cfg, q=0.9, margin=None):
    """Simulate one policy over the test period. Returns one row per hour."""
    c = copy.deepcopy(cfg)
    if margin is not None:
        c["margin"] = margin
    c["static_replicas"] = static_replicas(fc["actual"].max(), c)
    col = f"q{int(round(q * 100))}"
    rows = []
    for _, h in fc.iterrows():
        # h["naive"] is the load of the previous hour = what reactive saw; h[col] = forecast made for this hour
        replicas = decide(policy, 0, h["naive"], 0.0, h[col], c)
        m = step_metrics(h["actual"], replicas, c)
        rows.append({"timestamp": h["timestamp"], "policy": policy, "actual_rps": h["actual"],
                     "forecast": h[col], "replicas": replicas, **m})
    return pd.DataFrame(rows)


def summarise(df, cfg):
    """One summary row: the five metrics the brief asks for (accuracy is reported from model_metrics.csv)."""
    served_share = 1 - (df["violation_share"] * df["actual_rps"]).sum() / df["actual_rps"].sum()
    return {
        "avg_replicas": df["replicas"].mean(),
        "avg_utilisation_%": df["util"].mean() * 100,
        "hours_over_capacity_%": (df["violation_share"] > 0).mean() * 100,
        "requests_over_capacity_%": (1 - served_share) * 100,
        "replica_hours": df["replicas"].sum(),
        "cost_usd": df["cost"].sum(),
        "energy_kwh": df["energy_wh"].sum() / 1000,
    }


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    fc = pd.read_csv(RESULTS / "test_forecasts.csv", parse_dates=["timestamp"])
    q0, m0 = cfg["quantile"], cfg["margin"]

    # 1. Main comparison at the configured settings
    runs = {p: run(p, fc, cfg, q=q0) for p in POLICIES}
    summary = pd.DataFrame([{"policy": p, **summarise(df, cfg)} for p, df in runs.items()])
    summary.round(4).to_csv(RESULTS / "summary_sim.csv", index=False)
    pd.concat(runs.values()).to_csv(RESULTS / "sim_hourly.csv", index=False)

    # 2. Trade-off sweep: each policy family across its own "safety knob"
    sweep = []
    for mg in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        sweep.append({"family": "reactive", "knob": f"margin {mg}", **summarise(run("reactive", fc, cfg, margin=mg), cfg)})
    for fam in ["predictive", "hybrid"]:
        for q in [0.5, 0.7, 0.9, 0.95]:
            for mg in [1.0, 1.1]:
                sweep.append({"family": fam, "knob": f"q{int(q * 100)} margin {mg}",
                              **summarise(run(fam, fc, cfg, q=q, margin=mg), cfg)})
    sweep.append({"family": "static", "knob": f"margin {m0}", **summarise(runs["static"], cfg)})
    sweep = pd.DataFrame(sweep)
    sweep.round(4).to_csv(RESULTS / "sweep_sim.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    style = {"reactive": ("o", "-"), "predictive": ("s", "-"), "hybrid": ("^", "-"), "static": ("*", "")}
    for fam, g in sweep.groupby("family", sort=False):
        mk, ls = style[fam]
        if fam in ("predictive", "hybrid"):
            for mg, gg in g.groupby(g["knob"].str.split(" margin ").str[1]):
                gg = gg.sort_values("cost_usd")
                ax.plot(gg["cost_usd"], gg["requests_over_capacity_%"], marker=mk, ls="-" if mg == "1.0" else "--",
                        label=f"{fam} (q 0.5 to 0.95, margin {mg})")
        else:
            g = g.sort_values("cost_usd")
            ax.plot(g["cost_usd"], g["requests_over_capacity_%"], marker=mk, ls=ls or "none", ms=10 if fam == "static" else 6,
                    label=f"{fam}" + (" (margin 1.0 to 1.5)" if fam == "reactive" else ""))
    ax.set(xlabel="cost over the 21-day test period (USD)", ylabel="requests over capacity (%)",
           title="Lower-left is better: cost vs SLA risk for each policy's safety setting")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "tradeoff_sweep.png", dpi=150)
    plt.close(fig)

    # 3. One test week, load vs replicas per policy
    C = cfg["capacity_per_replica_rps"]
    fig, axes = plt.subplots(len(POLICIES), 1, figsize=(11, 9), sharex=True, sharey=True)
    for ax, p in zip(axes, POLICIES):
        wk = runs[p].head(24 * 7)
        ax.plot(wk["timestamp"], wk["actual_rps"] / C, color="black", lw=1, label="replicas needed (load / C)")
        ax.step(wk["timestamp"], wk["replicas"], where="mid", lw=1.5, label="replicas provisioned")
        over = wk["actual_rps"] / C > wk["replicas"]
        ax.scatter(wk.loc[over, "timestamp"], wk.loc[over, "replicas"], color="red", s=12, zorder=3,
                   label="under-provisioned hour")
        ax.set_title(p, loc="left", fontsize=10)
        ax.set_ylabel("replicas")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("First test week: provisioned replicas vs what the load needed")
    fig.tight_layout()
    fig.savefig(RESULTS / "policies_week.png", dpi=150)
    plt.close(fig)

    # 4. Misprediction analysis: worst P90 under-forecasts, and over-forecast waste
    col = f"q{int(round(q0 * 100))}"
    worst = (fc.assign(under_by=fc["actual"] - fc[col]).nlargest(5, "under_by"))
    mis = []
    for _, h in worst.iterrows():
        row = {"timestamp": h["timestamp"], "actual_rps": round(h["actual"], 2), "p90_forecast": round(h[col], 2),
               "under_by_rps": round(h["under_by"], 2), "replicas_needed": round(h["actual"] / C, 2)}
        for p in ["reactive", "predictive", "hybrid"]:
            r = runs[p].loc[runs[p]["timestamp"] == h["timestamp"]].iloc[0]
            row[f"{p}_replicas"] = r["replicas"]
            row[f"{p}_over_capacity_%"] = round(r["violation_share"] * 100, 1)
        mis.append(row)
    mis = pd.DataFrame(mis)
    mis.to_csv(RESULTS / "mispredictions.csv", index=False)
    pred = runs["predictive"]
    needed = np.ceil(pred["actual_rps"] / C - 1e-9).clip(lower=cfg["min_replicas"])
    waste = (pred["replicas"] - needed).clip(lower=0).sum()

    pd.set_option("display.width", 200)
    print(f"Main comparison (q = {q0}, margin = {m0}), 21-day test period:")
    print(summary.round(3).to_string(index=False))
    print("\nWorst P90 under-forecasts and what each policy did:")
    print(mis.to_string(index=False))
    print(f"\nPredictive over-provisioning: {waste:.0f} replica-hours above what the load needed "
          f"({waste / pred['replicas'].sum() * 100:.1f}% of its replica-hours)")
    print("\nwrote results/summary_sim.csv, sweep_sim.csv, tradeoff_sweep.png, policies_week.png, "
          "mispredictions.csv, sim_hourly.csv")


if __name__ == "__main__":
    main()