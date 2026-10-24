"""Summarise the full-length live runs into one table (same metrics as the simulator + real latency).

Usage (from the repo root, after copying the live logs back from EC2):  python sim/summarise_live.py

For each policy it uses the NEWEST results/live_<policy>_YYYYmmdd-HHMM.csv (controller log, mock runs ignored)
and results/locust_<policy>_stats.csv (Locust's latency summary for the same run).
Writes results/summary_live.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SLA_MS = 300
PCTS = [50, 66, 75, 80, 90, 95, 98, 99, 99.9, 99.99, 100]


def share_slower_than(row, sla_ms):
    """Approximate % of requests slower than sla_ms by interpolating Locust's percentile columns."""
    values = [float(row[f"{p}%"]) for p in PCTS]
    if sla_ms <= values[0]:
        return 50.0  # the median is already above the SLA: at least half are slower (lower bound)
    if sla_ms >= values[-1]:
        return 0.0
    pct_at_sla = float(np.interp(sla_ms, values, PCTS))  # percentile rank of the SLA value
    return 100 - pct_at_sla


def main():
    rows = []
    for policy in ["static", "reactive", "predictive", "hybrid"]:
        logs = sorted(f for f in RESULTS.glob(f"live_{policy}_*.csv")
                      if re.fullmatch(rf"live_{policy}_\d{{8}}-\d{{4}}\.csv", f.name))
        if not logs:
            continue
        log = pd.read_csv(logs[-1])
        row = {
            "policy": policy,
            "controller_log": logs[-1].name,
            "steps": len(log),
            "avg_load_rps": log["actual_rps"].mean(),
            "avg_replicas": log["replicas"].mean(),
            "avg_utilisation_%": log["util"].mean() * 100,
            "hours_over_capacity_%": (log["violation_share"] > 0).mean() * 100,
            "replica_hours": log["replicas"].sum(),
            "cost_usd": log["cost"].sum(),
            "energy_wh": log["energy_wh"].sum(),
            "measured_cpu_util_%": log["cpu_util_measured"].mean() * 100 if log["cpu_util_measured"].notna().any() else np.nan,
        }
        stats = RESULTS / f"locust_{policy}_stats.csv"
        if stats.exists():
            agg = pd.read_csv(stats).query("Name == 'Aggregated'").iloc[0]
            row.update({
                "requests": int(agg["Request Count"]),
                "failures": int(agg["Failure Count"]),
                "p50_ms": float(agg["50%"]),
                "p95_ms": float(agg["95%"]),
                "p99_ms": float(agg["99%"]),
                "max_ms": round(float(agg["Max Response Time"])),
                f"slower_than_{SLA_MS}ms_%": share_slower_than(agg, SLA_MS),
            })
        else:
            print(f"note: no {stats.name} - latency columns left empty for {policy}")
        rows.append(row)

    if not rows:
        raise SystemExit("no live_<policy>_YYYYmmdd-HHMM.csv files in results/ - copy them back from EC2 first")
    out = pd.DataFrame(rows)
    out.round(3).to_csv(RESULTS / "summary_live.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.drop(columns=["controller_log"]).round(2).to_string(index=False))
    print(f"\nwrote results/summary_live.csv  (latency % above {SLA_MS} ms is interpolated from Locust's percentiles)")


if __name__ == "__main__":
    main()