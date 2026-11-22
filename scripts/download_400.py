# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Bulk download 20y daily bars for 400 tickers via Yahoo v8 chart API."""
import csv, datetime, json, time, urllib.request, concurrent.futures, os, sys

TICKERS = [l.strip() for l in open("data/tickers_400.txt") if l.strip()]
OUTDIR = "data/universe"
os.makedirs(OUTDIR, exist_ok=True)
P1, P2 = 1136073600, 1788739200  # 2006-01-01 .. 2026-09-07

def fetch(sym):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={P1}&period2={P2}&interval=1d&events=div%7Csplit")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            j = json.load(r)
        res = j["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
        rows = []
        for i in range(len(ts)):
            o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
            v = q["volume"][i]
            if o is None or h is None or l is None or c is None:
                continue
            if adj and adj[i]:
                f = adj[i] / c if c else 1.0
                o, h, l, c = o*f, h*f, l*f, adj[i]
            rows.append((ts[i], o, h, l, c, int(v or 0)))
        if len(rows) < 300:
            return (sym, f"too-short-{len(rows)}", 0)
        with open(f"{OUTDIR}/{sym}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "high", "low", "close", "volume"])
            for t, o, h, l, c, v in rows:
                d = datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d")
                w.writerow([d, f"{o:.4f}", f"{h:.4f}", f"{l:.4f}", f"{c:.4f}", v])
        return (sym, "ok", len(rows))
    except Exception as e:  # noqa: BLE001
        return (sym, f"err-{type(e).__name__}:{str(e)[:80]}", 0)

ok = bad = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    for sym, st, n in ex.map(fetch, TICKERS):
        if st == "ok":
            ok += 1
        else:
            bad += 1
            print(f"FAIL {sym}: {st}", flush=True)
        if (ok + bad) % 50 == 0:
            print(f"... {ok+bad}/{len(TICKERS)} ok={ok} bad={bad}", flush=True)
print(f"DONE ok={ok} bad={bad}")
