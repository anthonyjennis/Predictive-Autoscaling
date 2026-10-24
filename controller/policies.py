"""Scaling policies and per-step metrics.

Shared by the live controller (controller/controller.py) AND the simulator (sim/simulator.py),
so live and simulated decisions and metrics are computed by exactly the same code.
"""
import math

POLICIES = ("static", "reactive", "predictive", "hybrid")
_EPS = 1e-9  # guards against float noise, e.g. ceil(5.0000000001) -> 6


def _ceil(x):
    return math.ceil(x - _EPS)


def clamp(n, cfg):
    return max(cfg["min_replicas"], min(cfg["max_replicas"], int(n)))


def static_replicas(peak_rps, cfg):
    """Always provision for the busiest test hour (plus margin)."""
    return clamp(_ceil(peak_rps / cfg["capacity_per_replica_rps"] * cfg["margin"]), cfg)


def reactive_replicas(last_rps, cfg):
    """Target tracking on the hour that just finished: size for the load we just SAW.

    Same formula as predictive_replicas(), but fed the last observed load instead of a forecast,
    so the comparison isolates the value of forecasting. Equivalent to Kubernetes HPA / AWS target
    tracking with a target utilisation of 1 / margin (about 91%). Scales both up and down.
    """
    return _ceil(last_rps * cfg["margin"] / cfg["capacity_per_replica_rps"])


def predictive_replicas(forecast_rps, cfg):
    """Provision for the forecast (P90 by default) plus margin."""
    return _ceil(forecast_rps * cfg["margin"] / cfg["capacity_per_replica_rps"])


def decide(policy, current_replicas, last_rps, last_util, forecast_p90, cfg):
    """Replicas for the NEXT hour.

    last_rps: load measured in the hour that just finished. forecast_p90: forecast for the next hour.
    current_replicas and last_util are kept in the signature for logging/compatibility; no policy
    depends on them, so no policy can "ratchet" (get stuck at its current size).
    cfg must contain 'static_replicas' when policy == 'static' (see static_replicas()).
    """
    if policy == "static":
        n = cfg["static_replicas"]
    elif policy == "reactive":
        n = reactive_replicas(last_rps, cfg)
    elif policy == "predictive":
        n = predictive_replicas(forecast_p90, cfg)
    elif policy == "hybrid":
        n = max(predictive_replicas(forecast_p90, cfg), reactive_replicas(last_rps, cfg))
    else:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    return clamp(n, cfg)


def utilisation(rps, replicas, cfg):
    """Offered load / capacity (can exceed 1 when overloaded)."""
    return rps / (replicas * cfg["capacity_per_replica_rps"])


def step_metrics(actual_rps, replicas, cfg):
    """Metrics for one trace hour, identical in the simulator and the live run."""
    capacity = replicas * cfg["capacity_per_replica_rps"]
    util = min(1.0, actual_rps / capacity) if capacity else 0.0
    violation = max(0.0, actual_rps - capacity) / actual_rps if actual_rps > 0 else 0.0
    e = cfg["energy"]
    watts = replicas * cfg["vcpu_per_replica"] * (
        e["min_watts_per_vcpu"] + util * (e["max_watts_per_vcpu"] - e["min_watts_per_vcpu"]))
    return {
        "capacity_rps": round(capacity, 3),
        "util": round(util, 4),
        "violation_share": round(violation, 4),          # share of requests beyond capacity
        "cost": round(replicas * cfg["price_per_replica_hour"], 6),   # USD for one trace hour
        "energy_wh": round(watts * 1.0, 4),               # watts x 1 trace hour (modelled)
    }