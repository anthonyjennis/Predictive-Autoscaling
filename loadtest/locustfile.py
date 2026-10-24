"""Replays the 24 live hours of data/live_slice.csv against the service.

Each trace hour is held for STEP_SECONDS (default 30; use 10 for the video runs).
Each simulated user sends 1 request/second, so users = target req/s (rounded to a whole number).

Data run (headless, saves latency history for the report), from the repo root:
  $env:STEP_SECONDS=30
  python -m locust -f loadtest/locustfile.py --headless --host http://<ec2-ip> --csv results/locust_<policy> --csv-full-history

Video run (web UI chart on http://localhost:8089, starts immediately):
  $env:STEP_SECONDS=10
  python -m locust -f loadtest/locustfile.py --host http://<ec2-ip> --autostart
"""
import csv
import os
from pathlib import Path

from locust import HttpUser, LoadTestShape, constant_throughput, task

ROOT = Path(__file__).resolve().parents[1]
STEP_SECONDS = float(os.environ.get("STEP_SECONDS", 30))
STEPS = int(os.environ.get("STEPS", 0))  # 0 = all 24 live hours

with open(ROOT / "data" / "live_slice.csv", newline="") as f:
    LIVE_RPS = [float(r["rps"]) for r in csv.DictReader(f) if r["phase"] == "live"]
if STEPS:
    LIVE_RPS = LIVE_RPS[:STEPS]


class TraceUser(HttpUser):
    wait_time = constant_throughput(1)  # each user = 1 request/second

    @task
    def hit(self):
        self.client.get("/")


class TraceShape(LoadTestShape):
    """Step k (k = run_time // STEP_SECONDS) runs round(rps_k) users; stops after the last hour."""

    def tick(self):
        k = int(self.get_run_time() // STEP_SECONDS)
        if k >= len(LIVE_RPS):
            return None  # end of the trace -> Locust stops
        users = max(1, round(LIVE_RPS[k]))
        return users, users  # spawn rate = users/s, so each new level is reached within ~1 s