"""Download hourly Wikipedia pageviews (human readers only) from the Wikimedia REST API.

Usage (from the repo root):
    python data/fetch_pageviews.py --contact you@example.com
    python data/fetch_pageviews.py --contact you@example.com --project de.wikipedia.org --start 2026-04-01 --end 2026-10-21

Writes data/raw_pageviews.csv with columns: timestamp (UTC), views
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw_pageviews.csv"
API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/"
       "{project}/all-access/user/hourly/{start}/{end}")


def month_chunks(start, end):
    """Split [start, end] into one window per calendar month (one API call each)."""
    cur = start
    while cur <= end:
        next_month = (cur.replace(day=1) + timedelta(days=32)).replace(day=1, hour=0)
        yield cur, min(next_month - timedelta(hours=1), end)
        cur = next_month


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="en.wikipedia.org")
    p.add_argument("--start", default="2026-04-01", help="first day, YYYY-MM-DD")
    p.add_argument("--end", default="2026-10-21", help="last day, YYYY-MM-DD (inclusive)")
    p.add_argument("--contact", required=True,
                   help="your email; Wikimedia asks every API client to identify itself")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23)

    session = requests.Session()
    session.headers["User-Agent"] = f"ACC-predictive-autoscaling-student-project/1.0 ({args.contact})"

    rows = []
    for a, b in month_chunks(start, end):
        url = API.format(project=args.project, start=a.strftime("%Y%m%d%H"), end=b.strftime("%Y%m%d%H"))
        print(f"fetching {a:%Y-%m-%d} .. {b:%Y-%m-%d} ...", end=" ", flush=True)
        resp = session.get(url, timeout=60)
        if resp.status_code != 200:
            print(f"\nHTTP {resp.status_code} for {url}\n{resp.text[:500]}")
            sys.exit(1)
        items = resp.json().get("items", [])
        rows += [(it["timestamp"], it["views"]) for it in items]
        print(f"{len(items)} hours")
        time.sleep(0.5)  # be polite to the API

    df = pd.DataFrame(rows, columns=["timestamp", "views"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} : {len(df)} rows, {df.timestamp.min()} -> {df.timestamp.max()}")


if __name__ == "__main__":
    main()