// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

//! Genome: the heritable trading brain.
//! 56 feature weights + thresholds + risk + emotion genes (fear/copy-love).

use crate::features::N_FEATURES;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Genome {
    /// feature weights in [-1,1]
    pub weights: [f64; N_FEATURES],
    /// buy/sell score thresholds
    pub buy_thresh: f64,  // 0.05..0.9
    pub sell_thresh: f64, // -0.9..-0.05
    /// ATR multiples
    pub stop_mult: f64, // 1.0..4.0
    pub take_mult: f64, // 1.0..8.0
    /// fraction of equity risked per trade
    pub risk_frac: f64, // 0.05..1.0 (fraction of equity in position)
    /// max hold days
    pub max_hold: usize, // 3..120
    /// FEAR OF DEATH: 0=bsen, 1=very scared. Shrinks size in drawdown.
    pub fear: f64, // 0..1
    /// LOVE OF BEING COPIED: 0..1. High copy-drive => slightly more
    /// turnover / bolder bets to stand out (sexual-selection pressure).
    pub copy_drive: f64, // 0..1
    /// long-only vs long/short switch
    pub allow_short: bool,
}

impl Genome {
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        let mut weights = [0.0; N_FEATURES];
        // sparse init: 60% zeros so evolution does feature selection
        for w in weights.iter_mut() {
            let r: f64 = rng.gen();
            if r < 0.55 {
                *w = 0.0;
            } else {
                *w = rng.gen_range(-1.0..1.0);
                // occasionally strong conviction
                if rng.gen::<f64>() < 0.12 {
                    *w *= 1.0;
                } else {
                    *w *= 0.45;
                }
            }
        }
        normalize_l1(&mut weights);
        Self {
            weights,
            buy_thresh: rng.gen_range(0.02..0.28),
            sell_thresh: rng.gen_range(-0.28..-0.02),
            stop_mult: rng.gen_range(1.0..4.0),
            take_mult: rng.gen_range(1.0..8.0),
            risk_frac: rng.gen_range(0.10..0.95),
            max_hold: rng.gen_range(3..120),
            fear: rng.gen_range(0.0..1.0),
            copy_drive: rng.gen_range(0.0..1.0),
            allow_short: rng.gen::<f64>() < 0.7,
        }
    }

    /// Dot-product score. Hot loop: manual unroll-friendly, no allocs.
    #[inline]
    pub fn score(&self, f: &[f64; N_FEATURES]) -> f64 {
        let mut s = 0.0;
        // unrolled by 8
        let w = &self.weights;
        let mut i = 0;
        while i + 8 <= N_FEATURES {
            s += w[i] * f[i]
                + w[i + 1] * f[i + 1]
                + w[i + 2] * f[i + 2]
                + w[i + 3] * f[i + 3]
                + w[i + 4] * f[i + 4]
                + w[i + 5] * f[i + 5]
                + w[i + 6] * f[i + 6]
                + w[i + 7] * f[i + 7];
            i += 8;
        }
        while i < N_FEATURES {
            s += w[i] * f[i];
            i += 1;
        }
        // copy-drive adds a touch of extraversion: bolder thresholds
        s * (1.0 + 0.15 * self.copy_drive)
    }

    /// Gaussian-creep + reset + zero-out + swap mutation, sigma annealed.
    pub fn mutated<R: Rng>(&self, rng: &mut R, sigma: f64) -> Self {
        let mut child = self.clone();
        // per-gene creep
        for w in child.weights.iter_mut() {
            let r: f64 = rng.gen();
            if r < 0.14 {
                // gaussian creep
                let g = (rng.gen::<f64>() + rng.gen::<f64>() + rng.gen::<f64>() - 1.5) * 2.0; // ~N(0,~0.57)
                *w += g * sigma;
            } else if r < 0.155 {
                *w = rng.gen_range(-1.0..1.0); // reset
            } else if r < 0.175 {
                *w = 0.0; // sparsity
            }
            if *w > 1.0 {
                *w = 1.0;
            } else if *w < -1.0 {
                *w = -1.0;
            }
        }
        // swap two genes (preserves L1)
        if rng.gen::<f64>() < 0.08 {
            let a = rng.gen_range(0..N_FEATURES);
            let b = rng.gen_range(0..N_FEATURES);
            child.weights.swap(a, b);
        }
        normalize_l1(&mut child.weights);

        let gp = |v: f64, lo: f64, hi: f64, s: f64, rng: &mut R| -> f64 {
            if rng.gen::<f64>() < 0.25 {
                let g = (rng.gen::<f64>() + rng.gen::<f64>() + rng.gen::<f64>() - 1.5) * 2.0;
                (v + g * s * (hi - lo)).clamp(lo, hi)
            } else {
                v
            }
        };
        child.buy_thresh = gp(child.buy_thresh, 0.015, 0.6, sigma, rng);
        child.sell_thresh = gp(child.sell_thresh, -0.6, -0.015, sigma, rng);
        // keep hysteresis valid: buy must stay above sell
        if child.buy_thresh < 0.015 {
            child.buy_thresh = 0.015;
        }
        if child.sell_thresh > -0.015 {
            child.sell_thresh = -0.015;
        }
        child.stop_mult = gp(child.stop_mult, 0.8, 5.0, sigma, rng);
        child.take_mult = gp(child.take_mult, 0.8, 10.0, sigma, rng);
        child.risk_frac = gp(child.risk_frac, 0.05, 1.0, sigma, rng);
        if rng.gen::<f64>() < 0.20 {
            let d: i32 = rng.gen_range(-12..13);
            child.max_hold = ((child.max_hold as i32 + d).clamp(2, 150)) as usize;
        }
        child.fear = gp(child.fear, 0.0, 1.0, sigma * 0.8, rng);
        child.copy_drive = gp(child.copy_drive, 0.0, 1.0, sigma * 0.8, rng);
        if rng.gen::<f64>() < 0.05 {
            child.allow_short = !child.allow_short;
        }
        child
    }

    /// Crossover: 60% uniform, 25% arithmetic, 15% BLX-0.5.
    pub fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let roll: f64 = rng.gen();
        let mut child = self.clone();
        if roll < 0.60 {
            for i in 0..N_FEATURES {
                if rng.gen::<bool>() {
                    child.weights[i] = other.weights[i];
                }
            }
            // scalar genes: pick from either parent
            if rng.gen::<bool>() {
                child.buy_thresh = other.buy_thresh;
            }
            if rng.gen::<bool>() {
                child.sell_thresh = other.sell_thresh;
            }
            if rng.gen::<bool>() {
                child.stop_mult = other.stop_mult;
            }
            if rng.gen::<bool>() {
                child.take_mult = other.take_mult;
            }
            if rng.gen::<bool>() {
                child.risk_frac = other.risk_frac;
            }
            if rng.gen::<bool>() {
                child.max_hold = other.max_hold;
            }
            if rng.gen::<bool>() {
                child.fear = other.fear;
            }
            if rng.gen::<bool>() {
                child.copy_drive = other.copy_drive;
            }
            if rng.gen::<bool>() {
                child.allow_short = other.allow_short;
            }
        } else if roll < 0.85 {
            let a: f64 = rng.gen_range(0.3..0.7);
            for i in 0..N_FEATURES {
                child.weights[i] = a * self.weights[i] + (1.0 - a) * other.weights[i];
            }
            child.buy_thresh = a * self.buy_thresh + (1.0 - a) * other.buy_thresh;
            child.sell_thresh = a * self.sell_thresh + (1.0 - a) * other.sell_thresh;
            child.stop_mult = a * self.stop_mult + (1.0 - a) * other.stop_mult;
            child.take_mult = a * self.take_mult + (1.0 - a) * other.take_mult;
            child.risk_frac = a * self.risk_frac + (1.0 - a) * other.risk_frac;
            child.fear = a * self.fear + (1.0 - a) * other.fear;
            child.copy_drive = a * self.copy_drive + (1.0 - a) * other.copy_drive;
        } else {
            for i in 0..N_FEATURES {
                let lo = self.weights[i].min(other.weights[i]);
                let hi = self.weights[i].max(other.weights[i]);
                let range = hi - lo;
                child.weights[i] =
                    rng.gen_range((lo - 0.5 * range).max(-1.0)..=(hi + 0.5 * range).min(1.0));
            }
        }
        normalize_l1(&mut child.weights);
        child
    }
}

fn normalize_l1(w: &mut [f64; N_FEATURES]) {
    let s: f64 = w.iter().map(|x| x.abs()).sum();
    if s > 1e-12 {
        let target = 1.0;
        for x in w.iter_mut() {
            *x *= target / s;
        }
    }
}
