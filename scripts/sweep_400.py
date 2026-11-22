# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Evolve one champion per ticker (real 20y CSVs) and build leaderboard."""
import concurrent.futures, csv, glob, os, re, subprocess, sys

BIN = "./target/release/ai-trades"
CSVS = sorted(glob.glob("data/universe/*.csv"))
os.makedirs("results", exist_ok=True)

PAT = re.compile(
    r"fitness\s+([+-]?\d[\d.]*)\s+·\s+total\s+([+-]?\d[\d. ]*)%\s+·\s+CAGR\s+([+-]?\d[\d. ]*)%"
    r"\s+·\s+Sharpe\s+([+-]?\d[\d.]*)\s+·\s+Sortino\s+([+-]?\d[\d.]*)\s+·\s+MaxDD\s+([\d.]+)%"
    r"\s+·\s+trades\s+(\d+).*?win\s+([\d.]+)%\s+·\s+PF\s+([\d.]+)"
)

def run_one(path):
    sym = os.path.splitext(os.path.basename(path))[0]
    try:
        p = subprocess.run(
            [BIN, "--csv", path, "--population", "10", "--generations", "25",
             "--seed", "42", "--quiet"],
            capture_output=True, text=True, timeout=300)
        out = p.stdout + p.stderr
        m = None
        for line in out.splitlines():
            if line.startswith("fitness "):
                mm = PAT.search(line)
                if mm:
                    m = mm
        if not m:
            return {"ticker": sym, "error": "no-parse"}
        g = m.groups()
        bars = sum(1 for _ in open(path)) - 1
        return {"ticker": sym, "fitness": float(g[0]), "total_pct": float(g[1]),
                "cagr_pct": float(g[2]), "sharpe": float(g[3]), "sortino": float(g[4]),
                "maxdd_pct": float(g[5]), "trades": int(g[6]), "win_pct": float(g[7]),
                "pf": float(g[8]), "bars": bars, "years": round(bars/252, 1)}
    except Exception as e:  # noqa: BLE001
        return {"ticker": sym, "error": str(e)[:100]}

rows = []
with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
    futs = [ex.submit(run_one, c) for c in CSVS]
    for i, f in enumerate(concurrent.futures.as_completed(futs)):
        rows.append(f.result())
        if (i+1) % 50 == 0:
            print(f"... {i+1}/{len(CSVS)}", flush=True)

good = [r for r in rows if "fitness" in r]
bad = [r for r in rows if "fitness" not in r]
good.sort(key=lambda r: r["fitness"], reverse=True)
with open("results/leaderboard_400.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ticker","fitness","total_pct","cagr_pct","sharpe",
        "sortino","maxdd_pct","trades","win_pct","pf","bars","years"])
    w.writeheader()
    w.writerows(good)
print(f"DONE ok={len(good)} bad={len(bad)}")
for r in good[:10]:
    print(f"{r['ticker']:8s} fit {r['fitness']:+.3f} tot {r['total_pct']:+7.1f}% shr {r['sharpe']:+.2f} dd {r['maxdd_pct']:.1f}% tr {r['trades']} win {r['win_pct']:.0f}%")
if bad:
    print("BAD:", bad[:5])
