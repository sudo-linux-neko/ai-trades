# EvoTrader (`ai-trades`) — Darwinian human-like trading engine in pure Rust

10 bots enter. Bottom 30% ☠ die. Top 40% ⬆ get copied with mutations. Middle 30% ─ survive.
20+ year tape. Fear of death. Love of being copied. 56 micro-eyes. Win/loss self-autopsy.

Pure Rust. Optimized hot loop (no allocs, unrolled dot-product, rayon-parallel bots).
Live CLI cinema: progress bars, price chart, equity sparkline, fitness curves, league table.

## Quickstart

```bash
cargo build --release
./target/release/ai-trades --population 10 --generations 25 --years 20 --seed 42
./target/release/ai-trades --population 30 --generations 50 --years 22 --verbose
./target/release/ai-trades --csv data/daily.csv --population 10 --generations 30
```

CSV needs header `open,high,low,close,volume`. If omitted, a deterministic
20y+ synthetic regime tape is generated (bull/bear/high-vol/crash + jumps,
252 bars/year, seeded).

Output: live dashboard + `champion_equity.csv` (bar,equity).

## How it matches your spec

- **10 copies**: default `--population 10`. Every gen the top 4 (40%) are replicated
  to replace the dead 3 (30%). Middle 3 survive untouched.
- **Fear of death**: `fear` gene 0..1 shrinks position size up to 85% at 20% drawdown.
  Cowardly genes survive crashes; reckless ones die and are not copied.
- **Love of being copied**: `was_copied` bots get +fitness love-bonus; `copy_drive`
  gene makes scores bolder so lineages stand out and get selected.
- **Win analysis**: per-bot journal splits wins, computes MFE capture, hold time,
  entry score, top-4 feature drivers (`weight × mean_feature_on_wins`).
- **Loss analysis**: mistake taxonomy — `stop-hit / whipsaw / reversal / time-stop /
  late-entry / gave-back-profit` — plus top blame exposures.
- **56 small eyes**: returns, SMA/EMA spreads, RSI×3, MACD×3, Bollinger×4, ATR/vol×7,
  momentum×5, volume×5, calendar×5, structure×10. All causal, precomputed once.
- **Fitness**: `Sharpe × min(1,trades/80) − 3×MaxDD − turnover_pen − inactivity_pen`.
  Flat-line cowards that never trade get −2 and die. No lookahead: entries at close,
  P&L from mark-to-mark, costs both sides.
- **Genetics**: uniform (60%) / arithmetic (25%) / BLX-0.5 (15%) crossover,
  Gaussian creep σ 0.30→0.05 + reset + zero-out + swap, diversity immigrants.
- **20y+**: `--years 20` = 5040 bars + 210 warmup. Tested to 40y.
- **CLI cinema**: indicatif gen/bot bars + ASCII price chart + equity/fitness
  sparklines (`▁▂▃▄▅▆▇█`) + league + autopsy, 5-gen cadence (`--verbose` every gen).

## Layout

- `src/market.rs` — 20y regime generator, CSV loader, ATR tape
- `src/features.rs` — 56 causal micro-features, O(N) caches
- `src/strategy.rs` — genome, score, mutate, crossover
- `src/backtest.rs` — honest daily L/S simulator, no lookahead
- `src/bot.rs` — bot + win/loss lessons
- `src/evolution.rs` — 40/30/30 selection, elites, diversity guard
- `src/ui.rs` — sparklines, charts, dashboard, autopsy
- `src/main.rs` — CLI + evolution loop

## Honesty note

Backtests are hypothesis generators, not profit machines. This engine enforces
costs, trade-count gates, drawdown penalties and (by default) long/short realism
so champions show realistic Sharpe 0.3–0.8 / PF 1.2–1.6, not fantasy 5000%.
Validate walk-forward before risking capital.
