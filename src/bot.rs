// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

//! Bot: a living trader with memory, fear, copy-love, and self-reflection.
//! Every generation each bot journals wins/losses and diagnoses its mistakes.

use crate::backtest::{BotResult, ExitReason, Trade, backtest};
use crate::features::{N_FEATURES, FeatureRow, feature_names};
use crate::market::MarketData;
use crate::strategy::Genome;

static mut NEXT_ID: u64 = 1;
fn next_id() -> u64 {
    unsafe {
        let v = NEXT_ID;
        NEXT_ID += 1;
        v
    }
}

#[derive(Clone, Debug)]
pub struct WinLesson {
    pub avg_win_pct: f64,
    pub avg_hold_win: f64,
    pub avg_entry_score: f64,
    pub capture: f64, // pnl/mfe
    pub top_drivers: Vec<(String, f64)>, // feature -> mean value on wins
    pub verdict: String,
}

#[derive(Clone, Debug)]
pub struct LossLesson {
    pub avg_loss_pct: f64,
    pub avg_hold_loss: f64,
    pub mistake_counts: Vec<(String, usize)>,
    pub top_blame: Vec<(String, f64)>,
    pub verdict: String,
}

#[derive(Clone, Debug)]
pub struct Bot {
    pub id: u64,
    pub generation: u32,
    pub genome: Genome,
    pub parents: (u64, u64),
    pub was_copied: bool,
    pub times_copied: u32, // lifetime copies (love meter)
    pub survived_gens: u32,
    pub alive: bool,
    pub result: BotResult,
    pub win_lesson: Option<WinLesson>,
    pub loss_lesson: Option<LossLesson>,
}

impl Bot {
    pub fn newborn(generation: u32, genome: Genome, parents: (u64, u64), was_copied: bool) -> Self {
        Self {
            id: next_id(),
            generation,
            genome,
            parents,
            was_copied,
            times_copied: 0,
            survived_gens: 0,
            alive: true,
            result: BotResult::default(),
            win_lesson: None,
            loss_lesson: None,
        }
    }

    /// Run 20-year backtest + journal analysis.
    pub fn trade(
        &mut self,
        market: &MarketData,
        feats: &[FeatureRow],
        atr: &[f64],
        fee_bps: f64,
        slip_bps: f64,
    ) {
        let mut r = backtest(market, feats, atr, &self.genome, fee_bps, slip_bps);
        // LOVE OF BEING COPIED: small intrinsic bonus if this bot was elite
        // (was copied last gen). Reinforces lineages that get replicated.
        if self.was_copied {
            r.fitness += 0.02 * (1.0 + self.times_copied as f64 * 0.1);
        }
        // FEAR OF DEATH shapes realized fitness already via position sizing,
        // but add tiny survival shaping: reward low-DD survivors.
        // (kept small so profit still dominates)
        if r.max_dd < 0.15 && r.num_trades > 30 {
            r.fitness += 0.01 * (1.0 - self.genome.fear * 0.3);
        }
        self.result = r;
        let (w, l) = self.analyze(feats);
        self.win_lesson = Some(w);
        self.loss_lesson = Some(l);
    }

    /// Split journal into wins/losses; attribute to features + exit reasons.
    fn analyze(&self, feats: &[FeatureRow]) -> (WinLesson, LossLesson) {
        let names = feature_names();
        let trades = &self.result.trades;
        let wins: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losses: Vec<&Trade> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        // mean feature vector on wins vs losses (attribution)
        let mut mean_win = [0.0_f64; N_FEATURES];
        let mut mean_loss = [0.0_f64; N_FEATURES];
        for t in &wins {
            let f = &feats[t.entry_idx.min(feats.len() - 1)];
            for i in 0..N_FEATURES {
                mean_win[i] += f[i];
            }
        }
        for t in &losses {
            let f = &feats[t.entry_idx.min(feats.len() - 1)];
            for i in 0..N_FEATURES {
                mean_loss[i] += f[i];
            }
        }
        if !wins.is_empty() {
            for v in mean_win.iter_mut() {
                *v /= wins.len() as f64;
            }
        }
        if !losses.is_empty() {
            for v in mean_loss.iter_mut() {
                *v /= losses.len() as f64;
            }
        }
        // weight * mean = contribution
        let mut contrib_win: Vec<(usize, f64)> = (0..N_FEATURES)
            .map(|i| (i, self.genome.weights[i] * mean_win[i]))
            .collect();
        contrib_win.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        let mut contrib_loss: Vec<(usize, f64)> = (0..N_FEATURES)
            .map(|i| (i, self.genome.weights[i] * mean_loss[i]))
            .collect();
        contrib_loss.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));

        let avg_win_pct = if wins.is_empty() {
            0.0
        } else {
            wins.iter().map(|t| t.pnl_pct).sum::<f64>() / wins.len() as f64
        };
        let avg_hold_win = if wins.is_empty() {
            0.0
        } else {
            wins.iter().map(|t| t.holding as f64).sum::<f64>() / wins.len() as f64
        };
        let avg_entry_score_win = if wins.is_empty() {
            0.0
        } else {
            wins.iter().map(|t| t.entry_score).sum::<f64>() / wins.len() as f64
        };
        let capture = if wins.is_empty() {
            0.0
        } else {
            let s: f64 = wins
                .iter()
                .map(|t| {
                    if t.mfe > 1e-9 {
                        (t.pnl_pct / t.mfe).clamp(0.0, 1.5)
                    } else {
                        0.5
                    }
                })
                .sum();
            s / wins.len() as f64
        };

        let win_verdict = if wins.is_empty() {
            "no wins — signal never pays; thresholds too tight or edge inverted".to_string()
        } else if capture > 0.55 && avg_hold_win <= self.genome.max_hold as f64 * 0.7 {
            format!(
                "winners work: captures {:.0}% of MFE, holds {:.0}d; edge = patience + trend alignment",
                capture * 100.0,
                avg_hold_win
            )
        } else if capture <= 0.35 {
            format!(
                "leaves money on table: captures only {:.0}% of MFE — take-profit too tight ({:.1}x ATR)",
                capture * 100.0,
                self.genome.take_mult
            )
        } else {
            format!(
                "steady edge: {:.2}%/win over {} wins, entry score {:.2}",
                avg_win_pct * 100.0,
                wins.len(),
                avg_entry_score_win
            )
        };

        let avg_loss_pct = if losses.is_empty() {
            0.0
        } else {
            losses.iter().map(|t| t.pnl_pct).sum::<f64>() / losses.len() as f64
        };
        let avg_hold_loss = if losses.is_empty() {
            0.0
        } else {
            losses.iter().map(|t| t.holding as f64).sum::<f64>() / losses.len() as f64
        };
        // mistake taxonomy from exit reasons + MAE/MFE shape
        let mut counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for t in &losses {
            let key = match t.exit_reason {
                ExitReason::Stop => "stop-hit (entry vs chop)",
                ExitReason::Take => "take-giveback??",
                ExitReason::Time => "time-stop (dead money)",
                ExitReason::Signal => {
                    if t.holding <= 5 {
                        "whipsaw (flip too fast)"
                    } else {
                        "reversal (trend broke)"
                    }
                }
                ExitReason::End => "held-into-end",
            };
            // adverse-excursion flavor
            let key2 = if t.mae < -0.04 && t.holding <= 4 {
                "late entry (instant heat)"
            } else if t.mfe > 0.015 && t.pnl <= 0.0 {
                "gave back profit (no trail)"
            } else {
                key
            };
            *counts.entry(key2.to_string()).or_insert(0) += 1;
        }
        let mut mistake_counts: Vec<(String, usize)> = counts.into_iter().collect();
        mistake_counts.sort_by(|a, b| b.1.cmp(&a.1));

        let loss_verdict = if losses.is_empty() {
            "no losses recorded (suspicious — probably <5 trades)".to_string()
        } else {
            let top = mistake_counts
                .first()
                .map(|(k, c)| format!("{k} x{c}"))
                .unwrap_or_default();
            if self.genome.stop_mult < 1.4 {
                format!("main mistake: {top}; stop {:.1}xATR too tight — shaken out by noise", self.genome.stop_mult)
            } else if avg_hold_loss > avg_hold_win * 1.6 {
                format!("main mistake: {top}; holds losers {avg_hold_loss:.0}d vs winners {avg_hold_win:.0}d — disposition effect")
            } else {
                format!("main mistake: {top}; avg loss {avg_loss_pct:.2}% — needs filter or wider stop", avg_loss_pct = avg_loss_pct*100.0)
            }
        };

        let w = WinLesson {
            avg_win_pct,
            avg_hold_win,
            avg_entry_score: avg_entry_score_win,
            capture,
            top_drivers: contrib_win
                .into_iter()
                .take(4)
                .map(|(i, v)| (names[i].to_string(), v))
                .collect(),
            verdict: win_verdict,
        };
        let l = LossLesson {
            avg_loss_pct,
            avg_hold_loss,
            mistake_counts,
            top_blame: contrib_loss
                .into_iter()
                .take(4)
                .map(|(i, v)| (names[i].to_string(), v))
                .collect(),
            verdict: loss_verdict,
        };
        (w, l)
    }

    pub fn fitness(&self) -> f64 {
        self.result.fitness
    }
}
