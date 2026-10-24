"""Live autoscaling controller.

Runs on the EC2 host (from the repo root). Every `step_seconds` (= one trace hour):
  1. measure the request rate of the step that just ended (nginx stub_status counter)
  2. log that step (replicas, forecast made for it, actual load, metrics)
  3. forecast the next hour (P50 + P90) from the history
  4. decide the replica count with policies.decide() and apply it via docker compose

Usage:
  python controller/controller.py --policy reactive                     # real run on EC2, dummy forecaster
  python controller/controller.py --policy hybrid --forecaster model    # real run with the trained model
  python controller/controller.py --policy hybrid --mock --step-seconds 1   # local test: no Docker, no nginx

Writes results/live_<policy>_<YYYYmmdd-HHMM>.csv (one row per step).
"""
import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "controller"))
from policies import POLICIES, decide, static_replicas, step_metrics, utilisation  # noqa: E402

LOG_FIELDS = ["step", "timestamp", "policy", "actual_rps", "forecast_p50", "forecast_p90", "replicas",
              "capacity_rps", "util", "violation_share", "cost", "energy_wh", "cpu_util_measured",
              "wall_clock"]


# ---------------------------------------------------------------- environment: real vs mock
class RealEnv:
    """Talks to nginx and Docker on the EC2 host."""

    def __init__(self, cfg):
        live = cfg["live"]
        self.deploy_dir = ROOT / live["deploy_dir"]
        self.api, self.nginx = live["api_service"], live["nginx_service"]
        self.status_url = live["nginx_status_url"]
        self.vcpu = cfg["vcpu_per_replica"]

    def request_count(self):
        """Total requests nginx has handled. stub_status line 3: ' accepts handled requests'."""
        r = requests.get(self.status_url, timeout=5)
        if r.status_code != 200:
            raise SystemExit(f"{self.status_url} returned HTTP {r.status_code}. If it's 403, nginx's "
                             "'allow' rule must include the Docker network (e.g. allow 172.16.0.0/12;), "
                             "because host requests reach the container from the Docker bridge, not 127.0.0.1.")
        return int(r.text.splitlines()[2].split()[2])

    def _compose(self, *args):
        return subprocess.run(["docker", "compose", *args], cwd=self.deploy_dir,
                              capture_output=True, text=True, check=True)

    def scale(self, n):
        self._compose("up", "-d", "--scale", f"{self.api}={n}", "--no-recreate")
        self._compose("exec", "-T", self.nginx, "nginx", "-s", "reload")  # nginx re-resolves the replicas

    def cpu_util(self, n):
        """Measured CPU of the api containers as a share of their combined limit (n x 0.25 vCPU)."""
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.CPUPerc}}"],
                             capture_output=True, text=True, check=True).stdout
        total = sum(float(line.split()[1].rstrip("%")) for line in out.splitlines()
                    if f"-{self.api}-" in line.split()[0])
        return round(total / (n * self.vcpu * 100), 4)


class MockEnv:
    """Pretends to be nginx + Docker: load = the trace's rps. For testing the loop on a laptop."""

    def __init__(self, live_rps):
        self.live_rps, self.step, self.count, self.t = list(live_rps), 0, 0.0, time.monotonic()

    def request_count(self):
        now = time.monotonic()
        rate = self.live_rps[min(self.step, len(self.live_rps) - 1)]
        self.count += rate * (now - self.t)
        self.t = now
        return int(self.count)

    def scale(self, n):
        print(f"      [mock] docker compose up -d --scale api={n}; nginx -s reload")

    def cpu_util(self, n):
        return None


# ---------------------------------------------------------------- forecaster
def load_forecaster(kind):
    if kind == "dummy":
        from dummy_forecaster import DummyForecaster
        return DummyForecaster()
    from model.forecaster import Forecaster
    return Forecaster(artifact_dir=str(ROOT / "model" / "artifacts"))


def forecast(fc, history):
    return round(fc.predict(history, 0.5), 3), round(fc.predict(history, 0.9), 3)


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=POLICIES)
    ap.add_argument("--forecaster", default="dummy", choices=["dummy", "model"])
    ap.add_argument("--mock", action="store_true", help="no Docker/nginx: simulate load from the trace")
    ap.add_argument("--step-seconds", type=float, help="override config step_seconds (e.g. 1 for a fast mock test)")
    ap.add_argument("--steps", type=int, help="stop after this many steps (default: all 24 live hours)")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    step_s = args.step_seconds or cfg["step_seconds"]
    C = cfg["capacity_per_replica_rps"]

    trace = pd.read_csv(ROOT / "data" / "trace_hourly.csv", parse_dates=["timestamp"])
    test = trace[trace["timestamp"] >= pd.Timestamp(cfg["trace"]["test_start"])]
    cfg["static_replicas"] = static_replicas(test["rps"].max(), cfg)

    sl = pd.read_csv(ROOT / "data" / "live_slice.csv", parse_dates=["timestamp"])
    history = sl[sl["phase"] == "history"][["timestamp", "rps"]].reset_index(drop=True)
    live = sl[sl["phase"] == "live"].reset_index(drop=True)
    if args.steps:
        live = live.head(args.steps)

    env = MockEnv(live["rps"]) if args.mock else RealEnv(cfg)
    fc = load_forecaster(args.forecaster)

    # Initial state: every policy makes its first decision from the end of the history
    p50, p90 = forecast(fc, history)
    last = history["rps"].iloc[-1]
    replicas = decide(args.policy, 0, last, 0.0, p90, cfg)
    env.scale(replicas)

    out = ROOT / "results" / f"live_{args.policy}{'_mock' if args.mock else ''}_{datetime.now():%Y%m%d-%H%M}.csv"
    out.parent.mkdir(exist_ok=True)
    print(f"policy={args.policy} forecaster={args.forecaster} steps={len(live)} step={step_s}s "
          f"start replicas={replicas} -> {out.name}")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        prev_count, prev_t = env.request_count(), time.monotonic()
        next_boundary = prev_t + step_s

        for k, row in live.iterrows():
            if args.mock:
                env.step = k
            time.sleep(max(0.0, next_boundary - time.monotonic()))
            next_boundary += step_s

            # 1. measure the step that just ended
            count, now = env.request_count(), time.monotonic()
            actual = (count - prev_count) / (now - prev_t)
            prev_count, prev_t = count, now
            cpu = env.cpu_util(replicas)

            # 2. log it (replicas and forecast are the ones that were in force during this step)
            m = step_metrics(actual, replicas, cfg)
            w.writerow({"step": k, "timestamp": row["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "policy": args.policy, "actual_rps": round(actual, 3), "forecast_p50": p50,
                        "forecast_p90": p90, "replicas": replicas, **m, "cpu_util_measured": cpu,
                        "wall_clock": datetime.now().isoformat(timespec="seconds")})
            f.flush()
            print(f"step {k:2d} {row['timestamp']:%m-%d %H:00}  load {actual:6.2f} rps  "
                  f"forecast p90 {p90:6.2f}  replicas {replicas}  util {m['util']:.2f}  "
                  f"over-capacity {m['violation_share']:.0%}")

            if k == len(live) - 1:
                break  # last step: nothing left to decide

            # 3. forecast the next hour from history + what we just measured
            history = pd.concat([history, pd.DataFrame({"timestamp": [row["timestamp"]], "rps": [actual]})],
                                ignore_index=True)
            p50, p90 = forecast(fc, history)

            # 4. decide and apply
            new = decide(args.policy, replicas, actual, utilisation(actual, replicas, cfg), p90, cfg)
            if new != replicas:
                env.scale(new)
                replicas = new

    print(f"done -> {out}")


if __name__ == "__main__":
    main()