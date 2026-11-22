// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

//! Backtester: single-asset daily long/short simulator.
//! Hot loop is allocation-free, cache-friendly, rayon-parallel across bots.

use crate::features::FeatureRow;
use crate::market::MarketData;
use crate::strategy::Genome;

#[derive(Clone, Debug)]
pub struct Trade {
    pub entry_idx: usize,
    pub exit_idx: usize,
    pub side: i8, // +1 long, -1 short
    pub entry_price: f64,
    pub exit_price: f64,
    pub size_frac: f64,
    pub pnl: f64,     // in equity units (fraction of equity at entry * return)
    pub pnl_pct: f64, // trade-level return on position
    pub holding: usize,
    pub entry_score: f64,
    pub exit_score: f64,
    pub mfe: f64, // max favorable excursion (fraction)
    pub mae: f64, // max adverse excursion (fraction, negative)
    pub exit_reason: ExitReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    Signal,
    Stop,
    Take,
    Time,
    End,
}

impl ExitReason {
    pub fn label(self) -> &'static str {
        match self {
            ExitReason::Signal => "signal-flip",
            ExitReason::Stop => "stop-loss",
            ExitReason::Take => "take-profit",
            ExitReason::Time => "time-stop",
            ExitReason::End => "end-of-data",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BotResult {
    pub equity: Vec<f64>,
    pub trades: Vec<Trade>,
    pub total_return: f64,
    pub cagr: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub max_dd: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub num_trades: usize,
    pub turnover: f64, // trades per year
    pub fitness: f64,
}

pub const WARMUP: usize = 210;

pub fn backtest(
    market: &MarketData,
    feats: &[FeatureRow],
    atr: &[f64],
    g: &Genome,
    fee_bps: f64,
    slip_bps: f64,
) -> BotResult {
    let n = market.len();
    let mut res = BotResult::default();
    if n < WARMUP + 30 {
        return res;
    }
    let closes: Vec<f64> = market.candles.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = market.candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = market.candles.iter().map(|c| c.low).collect();

    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut equity_curve = Vec::with_capacity(n);
    for _ in 0..WARMUP {
        equity_curve.push(1.0);
    }

    let cost_per_flip = (fee_bps + slip_bps) / 10_000.0; // e.g. 8bps -> 0.0008

    // open position state
    let mut pos_side: i8 = 0;
    let mut pos_entry: f64 = 0.0;
    let mut pos_entry_idx: usize = 0;
    let mut pos_size: f64 = 0.0;
    let mut pos_entry_score: f64 = 0.0;
    let mut pos_atr: f64 = 1.0;
    let mut pos_mfe: f64 = 0.0;
    let mut pos_mae: f64 = 0.0;

    let mut trades: Vec<Trade> = Vec::new();
    let mut daily_rets: Vec<f64> = Vec::with_capacity(n - WARMUP);

    // mark price: last valuation point of open position (entry, then each close)
    let mut mark_px: f64 = 0.0;

    for i in WARMUP..n {
        let score = g.score(&feats[i]);
        let a = atr[i].max(closes[i] * 1e-4).max(1e-9);

        // FEAR OF DEATH: live drawdown shrinks size (applies to NEW entries).
        let dd_now = if peak > 0.0 { (peak - equity) / peak } else { 0.0 };
        let fear_mult = 1.0 - g.fear * (dd_now / 0.20).clamp(0.0, 1.0) * 0.85;
        let lev = (g.risk_frac * fear_mult).clamp(0.02, 1.0);

        // desired position from score (hysteresis: hold when neutral)
        let desired: i8 = if score > g.buy_thresh {
            1
        } else if score < g.sell_thresh {
            if g.allow_short { -1 } else { 0 }
        } else {
            pos_side
        };

        let mut day_ret = 0.0_f64;
        let mut exited_this_bar = false;

        // ---- 1) manage carried position through today's bar (NO lookahead:
        // market move from mark_px is applied BEFORE any new entry today) ----
        if pos_side != 0 && pos_entry_idx < i {
            let holding = i - pos_entry_idx;
            let stop_px = if pos_side == 1 {
                pos_entry - g.stop_mult * pos_atr
            } else {
                pos_entry + g.stop_mult * pos_atr
            };
            let take_px = if pos_side == 1 {
                pos_entry + g.take_mult * pos_atr
            } else {
                pos_entry - g.take_mult * pos_atr
            };
            let stopped = if pos_side == 1 {
                lows[i] <= stop_px
            } else {
                highs[i] >= stop_px
            };
            let took = if pos_side == 1 {
                highs[i] >= take_px
            } else {
                lows[i] <= take_px
            };
            if stopped || took {
                // intrabar stop/take: value from last mark to stop/take price
                let px = if stopped { stop_px } else { take_px };
                let reason = if stopped { ExitReason::Stop } else { ExitReason::Take };
                let r = pos_side as f64 * pos_size * (px / mark_px - 1.0);
                equity *= 1.0 + r;
                // exit cost
                let c = cost_per_flip * pos_size;
                equity *= 1.0 - c;
                day_ret += r - c;
                let trade_ret = (px / pos_entry - 1.0) * pos_side as f64;
                // update excursions to exit
                if trade_ret > pos_mfe {
                    pos_mfe = trade_ret;
                }
                if trade_ret < pos_mae {
                    pos_mae = trade_ret;
                }
                trades.push(Trade {
                    entry_idx: pos_entry_idx,
                    exit_idx: i,
                    side: pos_side,
                    entry_price: pos_entry,
                    exit_price: px,
                    size_frac: pos_size,
                    pnl: pos_size * trade_ret - 2.0 * cost_per_flip * pos_size,
                    pnl_pct: trade_ret - 2.0 * cost_per_flip,
                    holding,
                    entry_score: pos_entry_score,
                    exit_score: score,
                    mfe: pos_mfe,
                    mae: pos_mae,
                    exit_reason: reason,
                });
                pos_side = 0;
                exited_this_bar = true;
            } else {
                // mark to close
                let r = pos_side as f64 * pos_size * (closes[i] / mark_px - 1.0);
                equity *= 1.0 + r;
                day_ret += r;
                mark_px = closes[i];
                let cur_ret = (closes[i] / pos_entry - 1.0) * pos_side as f64;
                if cur_ret > pos_mfe {
                    pos_mfe = cur_ret;
                }
                if cur_ret < pos_mae {
                    pos_mae = cur_ret;
                }
                // time / signal exits at the close
                let mut sig_exit: Option<ExitReason> = None;
                if holding >= g.max_hold {
                    sig_exit = Some(ExitReason::Time);
                } else if desired != pos_side && desired != 0 {
                    sig_exit = Some(ExitReason::Signal);
                } else if desired == 0 && pos_side != 0 {
                    let flat_exit =
                        (pos_side == 1 && score < 0.0) || (pos_side == -1 && score > 0.0);
                    if flat_exit {
                        sig_exit = Some(ExitReason::Signal);
                    }
                }
                if let Some(reason) = sig_exit {
                    let trade_ret = (closes[i] / pos_entry - 1.0) * pos_side as f64;
                    let c = cost_per_flip * pos_size;
                    equity *= 1.0 - c;
                    day_ret -= c;
                    trades.push(Trade {
                        entry_idx: pos_entry_idx,
                        exit_idx: i,
                        side: pos_side,
                        entry_price: pos_entry,
                        exit_price: closes[i],
                        size_frac: pos_size,
                        pnl: pos_size * trade_ret - 2.0 * cost_per_flip * pos_size,
                        pnl_pct: trade_ret - 2.0 * cost_per_flip,
                        holding,
                        entry_score: pos_entry_score,
                        exit_score: score,
                        mfe: pos_mfe,
                        mae: pos_mae,
                        exit_reason: reason,
                    });
                    pos_side = 0;
                    exited_this_bar = true;
                }
            }
        } else if pos_side != 0 && pos_entry_idx == i {
            // entered earlier today (flip case below handles); nothing to mark
        }

        // ---- 2) entries at today's close (no capture of today's move) ----
        if pos_side == 0 && desired != 0 {
            pos_side = desired;
            pos_entry = closes[i];
            mark_px = closes[i];
            pos_entry_idx = i;
            pos_size = lev;
            pos_entry_score = score;
            pos_atr = a;
            pos_mfe = 0.0;
            pos_mae = 0.0;
            let c = cost_per_flip * pos_size;
            equity *= 1.0 - c;
            // entry cost counts toward today's return
            day_ret -= c;
            let _ = exited_this_bar;
        }

        if equity < 1e-9 {
            equity = 1e-9;
        }
        // first bar has no prior day; keep lengths aligned
        if i == WARMUP {
            daily_rets.push(0.0);
        } else {
            daily_rets.push(day_ret);
        }
        if equity > peak {
            peak = equity;
        }
        equity_curve.push(equity);
    }

    // force-close open position at end (equity already marked to final close)
    if pos_side != 0 {
        let px = closes[n - 1];
        let trade_ret = (px / pos_entry - 1.0) * pos_side as f64;
        // charge exit cost on equity too for consistency
        equity *= 1.0 - cost_per_flip * pos_size;
        trades.push(Trade {
            entry_idx: pos_entry_idx,
            exit_idx: n - 1,
            side: pos_side,
            entry_price: pos_entry,
            exit_price: px,
            size_frac: pos_size,
            pnl: trade_ret * pos_size - 2.0 * cost_per_flip * pos_size,
            pnl_pct: trade_ret - 2.0 * cost_per_flip,
            holding: n - 1 - pos_entry_idx,
            entry_score: pos_entry_score,
            exit_score: 0.0,
            mfe: pos_mfe,
            mae: pos_mae,
            exit_reason: ExitReason::End,
        });
    }

    // ---- stats ----
    let total_return = equity - 1.0;
    let years = ((n - WARMUP) as f64 / 252.0).max(0.1);
    let cagr = equity.powf(1.0 / years) - 1.0;
    let sharpe = {
        if daily_rets.is_empty() {
            0.0
        } else {
            let m = daily_rets.iter().sum::<f64>() / daily_rets.len() as f64;
            let v = daily_rets.iter().map(|x| (x - m) * (x - m)).sum::<f64>()
                / daily_rets.len() as f64;
            let sd = v.sqrt();
            if sd < 1e-12 {
                0.0
            } else {
                m / sd * (252.0_f64).sqrt()
            }
        }
    };
    let sortino = {
        if daily_rets.is_empty() {
            0.0
        } else {
            let m = daily_rets.iter().sum::<f64>() / daily_rets.len() as f64;
            let dn: Vec<f64> = daily_rets.iter().cloned().filter(|x| *x < 0.0).collect();
            if dn.is_empty() {
                sharpe
            } else {
                let v = dn.iter().map(|x| x * x).sum::<f64>() / dn.len() as f64;
                let sd = v.sqrt();
                if sd < 1e-12 {
                    0.0
                } else {
                    m / sd * (252.0_f64).sqrt()
                }
            }
        }
    };
    let max_dd = {
        let mut pk = f64::NEG_INFINITY;
        let mut md = 0.0;
        for &e in &equity_curve {
            if e > pk {
                pk = e;
            }
            let dd = if pk > 0.0 { (pk - e) / pk } else { 0.0 };
            if dd > md {
                md = dd;
            }
        }
        md
    };
    let num_trades = trades.len();
    let wins = trades.iter().filter(|t| t.pnl > 0.0).count();
    let win_rate = if num_trades > 0 {
        wins as f64 / num_trades as f64
    } else {
        0.0
    };
    let gross_p: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
    let gross_l: f64 = trades.iter().filter(|t| t.pnl <= 0.0).map(|t| -t.pnl).sum();
    let profit_factor = if gross_l > 1e-12 {
        gross_p / gross_l
    } else if gross_p > 0.0 {
        5.0
    } else {
        0.0
    };
    // display averages are trade-level (per-trade % on position) to match journal;
    // PF stays equity-weighted (size-aware).
    let avg_win = if wins > 0 {
        trades
            .iter()
            .filter(|t| t.pnl > 0.0)
            .map(|t| t.pnl_pct)
            .sum::<f64>()
            / wins as f64
    } else {
        0.0
    };
    let avg_loss = if num_trades > wins {
        -trades
            .iter()
            .filter(|t| t.pnl <= 0.0)
            .map(|t| t.pnl_pct)
            .sum::<f64>()
            / (num_trades - wins) as f64
    } else {
        0.0
    };
    let turnover = num_trades as f64 / years;
    // fitness: risk-adjusted + trade-count gate + turnover + DD penalty.
    // Cowards that never trade must DIE: heavy inactivity penalty so
    // evolution cannot converge to flat-line bots.
    let mut fitness = if num_trades < 15 {
        -2.0 - (15 - num_trades) as f64 * 0.05 - 3.0 * max_dd
    } else {
        let trade_gate = ((num_trades as f64) / 80.0).min(1.0);
        let turnover_pen = (turnover / 120.0).min(1.0) * 0.4;
        let inactivity_pen = if num_trades < 40 {
            (40 - num_trades) as f64 * 0.008
        } else {
            0.0
        };
        sharpe * trade_gate - 3.0 * max_dd - turnover_pen - inactivity_pen
    };
    // sanitize: corrupt ticks must yield death, never NaN (which poisons sorts)
    if !fitness.is_finite() {
        fitness = -5.0;
    }

    res.equity = equity_curve;
    res.trades = trades;
    res.total_return = total_return;
    res.cagr = cagr;
    res.sharpe = sharpe;
    res.sortino = sortino;
    res.max_dd = max_dd;
    res.win_rate = win_rate;
    res.profit_factor = profit_factor;
    res.avg_win = avg_win;
    res.avg_loss = avg_loss;
    res.num_trades = num_trades;
    res.turnover = turnover;
    res.fitness = fitness;
    res
}
