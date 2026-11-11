"""Daily closes for the lab, no API keys.

    python data/fetch.py          # update both CSVs incrementally

- BTC-USD spot, Coinbase Exchange candles API (300 per request, paginated).
- Gold: Yahoo Finance GC=F front-month futures as the XAUUSD proxy - the
  basis vs spot is noise at the daily scale, documented so nobody mistakes
  it for spot ticks.

The committed CSVs were bootstrapped from the paper-trader repo's copies;
this script extends them with whatever is newer.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (montecarlo-lab data fetch)"}


def get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_coinbase_daily(start: datetime, end: datetime):
    bars = []
    page_end = end
    while page_end > start:
        page_start = max(start, page_end - timedelta(days=299))
        url = ("https://api.exchange.coinbase.com/products/BTC-USD/candles"
               f"?granularity=86400&start={page_start.isoformat()}&end={page_end.isoformat()}")
        chunk = get_json(url)
        if not chunk:
            break
        for t, low, high, open_, close, vol in chunk:
            bars.append((int(t), open_, high, low, close, vol))
        new_end = datetime.fromtimestamp(min(c[0] for c in chunk), tz=timezone.utc)
        if new_end >= page_end:
            break
        page_end = new_end
        time.sleep(0.15)
    return sorted(set(bars))


def fetch_gold_daily(start: datetime, end: datetime):
    p1 = int(start.replace(tzinfo=timezone.utc).timestamp())
    p2 = int(end.replace(tzinfo=timezone.utc).timestamp())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
           f"?period1={p1}&period2={p2}&interval=1d")
    res = get_json(url)["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    return [(t, o, h, l, c, v) for t, o, h, l, c, v in
            zip(ts, q["open"], q["high"], q["low"], q["close"], q["volume"]) if c]


def update(path: Path, fetch) -> int:
    rows = []
    if path.exists():
        with open(path) as f:
            rows = [(int(datetime.fromisoformat(r["ts"]).timestamp()),
                     float(r["open"]), float(r["high"]), float(r["low"]),
                     float(r["close"]), float(r["volume"]))
                    for r in csv.DictReader(f)]
    start = datetime.fromtimestamp(rows[-1][0] + 1, tz=timezone.utc) if rows \
        else datetime(2010, 1, 1, tzinfo=timezone.utc)
    new = fetch(start, datetime.now(timezone.utc))
    by_ts = {t: (t, o, h, l, c, v) for t, o, h, l, c, v in rows}
    for t, o, h, l, c, v in new:
        by_ts[t] = (t, o, h, l, c, v)
    rows = [by_ts[k] for k in sorted(by_ts)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for t, o, h, l, c, v in rows:
            w.writerow([datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                        f"{float(o):.2f}", f"{float(h):.2f}", f"{float(l):.2f}",
                        f"{float(c):.2f}", f"{float(v or 0):.4f}"])
    return len(new)


if __name__ == "__main__":
    n1 = update(HERE / "btc_1d.csv", fetch_coinbase_daily)
    print(f"btc: {n1} new bars")
    n2 = update(HERE / "xauusd_1d.csv", fetch_gold_daily)
    print(f"gold: {n2} new bars")
