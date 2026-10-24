"""Quick checks for policies.py.  Run from the repo root:  python controller/test_policies.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policies import decide, static_replicas, step_metrics  # noqa: E402

CFG = {
    "capacity_per_replica_rps": 6, "min_replicas": 1, "max_replicas": 6, "margin": 1.1,
    "vcpu_per_replica": 0.25, "price_per_replica_hour": 0.01413,
    "energy": {"min_watts_per_vcpu": 0.71, "max_watts_per_vcpu": 3.46},
}


def check(name, got, want):
    assert got == want, f"{name}: got {got}, want {want}"
    print(f"ok  {name}: {got}")


CFG["static_replicas"] = static_replicas(30, CFG)
check("static: peak 30 rps -> ceil(30/6*1.1)=6", decide("static", 3, 10, 0.5, 10, CFG), 6)

# reactive: sized from the load we just saw
check("reactive: last hour 16 rps -> ceil(16*1.1/6)=3", decide("reactive", 3, 16, 0, 0, CFG), 3)
check("reactive: scales DOWN 6 -> 2 when load falls to 10", decide("reactive", 6, 10, 0, 0, CFG), 2)
check("reactive: never below min", decide("reactive", 2, 0, 0, 0, CFG), 1)
check("reactive: ignores the forecast", decide("reactive", 3, 16, 0, 60, CFG), 3)

# predictive: sized from the P90 forecast
check("predictive: P90 20 rps -> ceil(20*1.1/6)=4", decide("predictive", 1, 0, 0, 20, CFG), 4)
check("predictive: P90 60 rps -> capped at max 6", decide("predictive", 1, 0, 0, 60, CFG), 6)

# hybrid = max of the two
check("hybrid: forecast 25, last 10 -> predictive wins: 5", decide("hybrid", 3, 10, 0, 25, CFG), 5)
check("hybrid: forecast 10, last 25 -> reactive wins: 5", decide("hybrid", 3, 25, 0, 10, CFG), 5)
check("hybrid: can scale DOWN 6 -> 3 when both fall to 16", decide("hybrid", 6, 16, 0, 16, CFG), 3)

# metrics
m = step_metrics(36, 5, CFG)  # capacity 30, overloaded by 6 rps
check("metrics: violation share 6/36", m["violation_share"], round(6 / 36, 4))
check("metrics: util capped at 1", m["util"], 1.0)
check("metrics: cost 5 x price", m["cost"], round(5 * 0.01413, 6))
check("metrics: energy 5 x 0.25 x 3.46 W", m["energy_wh"], round(5 * 0.25 * 3.46, 4))
print("\nall policy tests passed")