// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

//! Micro-feature engine: 56 tiny "eyes" every bot uses to see the market.
//! All causal (no lookahead), O(N) precomputed once, reused by every bot.
//! Optimized: single pass SMA/EMA caches, no allocs in hot loop.

use crate::market::MarketData;

/// Number of micro-features ("a million small things", compressed to 56 dense eyes).
pub const N_FEATURES: usize = 56;

pub fn feature_names() -> [&'static str; N_FEATURES] {
    [
        // returns 0..6
        "ret1", "ret5", "ret10", "ret21", "ret63", "lret21", "signsum10",
        // trend spreads 7..13
        "csma10", "csma20", "csma50", "csma200", "cema12", "cema26", "cross10_20",
        // rsi 14..16
        "rsi2", "rsi14", "rsi28",
        // macd 17..19
        "macd_line", "macd_sig", "macd_hist",
        // bollinger 20..23
        "bb_pctb", "bb_width", "bb_z", "bb_squeeze",
        // vol / atr 24..30
        "atr_pct", "atr_ratio", "rv20", "rv_ratio5_20", "rv_ratio21_63", "parkinson20", "range_atr",
        // momentum 31..35
        "roc12", "roc126", "stoch_k", "stoch_d", "wpr",
        // volume 36..40
        "vol_ratio", "vol_z", "dollarvol_ratio", "obv_mom", "amihud_z",
        // calendar 41..45
        "dow_sin", "dow_cos", "mon_sin", "mon_cos", "turn_month",
        // structure 46..55
        "dd63", "du63", "days_high63", "days_low63", "gap_atr", "range_pos", "clv63",
        "dist_high20", "dist_low20", "efficiency21",
    ]
}

/// Precomputed feature matrix: features[i] aligns with candles[i].
pub type FeatureRow = [f64; N_FEATURES];

#[inline]
fn clip3(x: f64) -> f64 {
    (x / 3.0).clamp(-1.0, 1.0)
}

pub fn compute_features(m: &MarketData) -> Vec<FeatureRow> {
    let n = m.len();
    let mut out = vec![[0.0_f64; N_FEATURES]; n];
    if n == 0 {
        return out;
    }
    let o: Vec<f64> = m.candles.iter().map(|c| c.open).collect();
    let h: Vec<f64> = m.candles.iter().map(|c| c.high).collect();
    let l: Vec<f64> = m.candles.iter().map(|c| c.low).collect();
    let c: Vec<f64> = m.candles.iter().map(|c| c.close).collect();
    let v: Vec<f64> = m.candles.iter().map(|c| c.volume).collect();

    // ---- rolling SMA caches ----
    let sma = |src: &[f64], w: usize| -> Vec<f64> {
        let mut r = vec![0.0; n];
        let mut acc = 0.0;
        for i in 0..n {
            acc += src[i];
            if i >= w {
                acc -= src[i - w];
                r[i] = acc / w as f64;
            } else {
                r[i] = acc / (i + 1) as f64;
            }
        }
        r
    };
    let ema = |src: &[f64], span: usize| -> Vec<f64> {
        let mut r = vec![0.0; n];
        let k = 2.0 / (span as f64 + 1.0);
        r[0] = src[0];
        for i in 1..n {
            r[i] = src[i] * k + r[i - 1] * (1.0 - k);
        }
        r
    };

    let sma10 = sma(&c, 10);
    let sma20 = sma(&c, 20);
    let sma50 = sma(&c, 50);
    let sma200 = sma(&c, 200);
    let ema12 = ema(&c, 12);
    let ema26 = ema(&c, 26);
    let vsma20 = sma(&v, 20);

    // dollar volume + sma
    let dv: Vec<f64> = c.iter().zip(v.iter()).map(|(a, b)| a * b).collect();
    let dvsma20 = sma(&dv, 20);

    // ---- RSI (Wilder) ----
    let rsi = |period: usize| -> Vec<f64> {
        let mut r = vec![50.0; n];
        let mut g = 0.0;
        let mut lo = 0.0;
        for i in 1..n {
            let ch = c[i] - c[i - 1];
            let gain = ch.max(0.0);
            let loss = (-ch).max(0.0);
            if i <= period {
                g += gain;
                lo += loss;
                if i == period {
                    g /= period as f64;
                    lo /= period as f64;
                    let rs = g / (lo + 1e-12);
                    r[i] = 100.0 - 100.0 / (1.0 + rs);
                }
            } else {
                g = (g * (period as f64 - 1.0) + gain) / period as f64;
                lo = (lo * (period as f64 - 1.0) + loss) / period as f64;
                let rs = g / (lo + 1e-12);
                r[i] = 100.0 - 100.0 / (1.0 + rs);
            }
        }
        // fill head
        for i in 0..period.min(n) {
            r[i] = 50.0;
        }
        r
    };
    let rsi2 = rsi(2);
    let rsi14 = rsi(14);
    let rsi28 = rsi(28);

    // ---- MACD ----
    let macd_line: Vec<f64> = ema12.iter().zip(ema26.iter()).map(|(a, b)| a - b).collect();
    let macd_sig = ema(&macd_line, 9);

    // ---- Bollinger 20 ----
    let mut bb_mid = vec![0.0; n];
    let mut bb_sd = vec![0.0; n];
    {
        let mut acc = 0.0;
        let mut acc2 = 0.0;
        for i in 0..n {
            acc += c[i];
            acc2 += c[i] * c[i];
            let w = 20.min(i + 1);
            if i >= 20 {
                acc -= c[i - 20];
                acc2 -= c[i - 20] * c[i - 20];
            }
            let mean = acc / w as f64;
            let var = (acc2 / w as f64 - mean * mean).max(0.0);
            bb_mid[i] = mean;
            bb_sd[i] = var.sqrt().max(1e-9);
        }
    }
    let mut bw120_mean = vec![0.0; n];
    {
        let bw: Vec<f64> = (0..n)
            .map(|i| 4.0 * bb_sd[i] / bb_mid[i].max(1e-9))
            .collect();
        let s = sma(&bw, 120);
        bw120_mean = s;
    }

    // ---- ATR / TR ----
    let mut tr = vec![0.0; n];
    let mut atr14v = vec![0.0; n];
    let mut atr50v = vec![0.0; n];
    {
        let mut a14 = 0.0;
        let mut a50 = 0.0;
        for i in 0..n {
            let t = if i == 0 {
                h[i] - l[i]
            } else {
                (h[i] - l[i])
                    .max((h[i] - c[i - 1]).abs())
                    .max((l[i] - c[i - 1]).abs())
            }
            .max(1e-9);
            tr[i] = t;
            if i == 0 {
                a14 = t;
                a50 = t;
            } else if i < 14 {
                a14 = (a14 * i as f64 + t) / (i as f64 + 1.0);
            } else {
                a14 = (a14 * 13.0 + t) / 14.0;
            }
            if i < 50 {
                a50 = (a50 * i as f64 + t) / (i as f64 + 1.0);
            } else {
                a50 = (a50 * 49.0 + t) / 50.0;
            }
            atr14v[i] = a14;
            atr50v[i] = a50;
        }
    }

    // ---- log returns + realized vol ----
    let mut lr = vec![0.0; n];
    for i in 1..n {
        lr[i] = (c[i] / c[i - 1]).ln();
    }
    let rv = |w: usize| -> Vec<f64> {
        let mut r = vec![0.0; n];
        let mut acc = 0.0;
        let mut acc2 = 0.0;
        for i in 0..n {
            acc += lr[i];
            acc2 += lr[i] * lr[i];
            let ww = w.min(i + 1);
            if i >= w {
                acc -= lr[i - w];
                acc2 -= lr[i - w] * lr[i - w];
            }
            let mean = acc / ww as f64;
            let var = (acc2 / ww as f64 - mean * mean).max(0.0);
            r[i] = var.sqrt() * (252.0_f64).sqrt();
        }
        r
    };
    let rv5 = rv(5);
    let rv20 = rv(21);
    let rv63 = rv(63);

    // parkinson 20
    let mut park = vec![0.0; n];
    {
        let mut acc = 0.0;
        let mut buf = vec![0.0; n];
        for i in 0..n {
            let x = if h[i] > 0.0 && l[i] > 0.0 {
                (h[i] / l[i]).ln()
            } else {
                0.0
            };
            buf[i] = x * x / (4.0 * std::f64::consts::LN_2);
        }
        for i in 0..n {
            acc += buf[i];
            if i >= 20 {
                acc -= buf[i - 20];
            }
            let w = 20.min(i + 1) as f64;
            park[i] = (acc / w).sqrt() * (252.0_f64).sqrt();
        }
    }

    // ---- OBV ----
    let mut obv = vec![0.0; n];
    for i in 1..n {
        let s = if c[i] > c[i - 1] {
            1.0
        } else if c[i] < c[i - 1] {
            -1.0
        } else {
            0.0
        };
        obv[i] = obv[i - 1] + s * v[i];
    }
    let obv_sma20 = sma(&obv, 20);

    // ---- log-volume rolling z (60) ----
    let lv: Vec<f64> = v.iter().map(|x| (x + 1.0).ln()).collect();
    let lv_mean60 = sma(&lv, 60);
    let mut lv_sd60 = vec![1.0; n];
    {
        let mut acc = 0.0;
        let mut acc2 = 0.0;
        for i in 0..n {
            acc += lv[i];
            acc2 += lv[i] * lv[i];
            let w = 60.min(i + 1);
            if i >= 60 {
                acc -= lv[i - 60];
                acc2 -= lv[i - 60] * lv[i - 60];
            }
            let mean = acc / w as f64;
            let var = (acc2 / w as f64 - mean * mean).max(0.0);
            lv_sd60[i] = var.sqrt().max(1e-9);
        }
    }

    // ---- rolling z helper for generic series (252 win, min 63) ----
    let roll_z = |s: &[f64]| -> Vec<f64> {
        let mean = sma(s, 252);
        let mut sd = vec![1.0; n];
        let mut acc = 0.0;
        let mut acc2 = 0.0;
        for i in 0..n {
            acc += s[i];
            acc2 += s[i] * s[i];
            let w = 252.min(i + 1);
            if i >= 252 {
                acc -= s[i - 252];
                acc2 -= s[i - 252] * s[i - 252];
            }
            let m = acc / w as f64;
            let vv = (acc2 / w as f64 - m * m).max(0.0);
            sd[i] = vv.sqrt().max(1e-9);
        }
        (0..n)
            .map(|i| {
                if i < 63 {
                    0.0
                } else {
                    ((s[i] - mean[i]) / sd[i]).clamp(-3.0, 3.0) / 3.0
                }
            })
            .collect()
    };

    // raw series needing z-normalization
    let spread10_raw: Vec<f64> = (0..n).map(|i| (c[i] - sma10[i]) / sma10[i].max(1e-9)).collect();
    let spread20_raw: Vec<f64> = (0..n).map(|i| (c[i] - sma20[i]) / sma20[i].max(1e-9)).collect();
    let spread50_raw: Vec<f64> = (0..n).map(|i| (c[i] - sma50[i]) / sma50[i].max(1e-9)).collect();
    let spread200_raw: Vec<f64> = (0..n).map(|i| (c[i] - sma200[i]) / sma200[i].max(1e-9)).collect();
    let espread12_raw: Vec<f64> = (0..n).map(|i| (c[i] - ema12[i]) / ema12[i].max(1e-9)).collect();
    let espread26_raw: Vec<f64> = (0..n).map(|i| (c[i] - ema26[i]) / ema26[i].max(1e-9)).collect();
    let z_sp10 = roll_z(&spread10_raw);
    let z_sp20 = roll_z(&spread20_raw);
    let z_sp50 = roll_z(&spread50_raw);
    let z_sp200 = roll_z(&spread200_raw);
    let z_esp12 = roll_z(&espread12_raw);
    let z_esp26 = roll_z(&espread26_raw);

    // Amihud raw + z
    let amihud_raw: Vec<f64> = (0..n)
        .map(|i| {
            let r1 = if i > 0 { (c[i] / c[i - 1] - 1.0).abs() } else { 0.0 };
            (r1 / (c[i] * v[i] + 1e6)).ln().clamp(-30.0, 0.0)
        })
        .collect();
    let z_amihud = roll_z(&amihud_raw);

    // ---- main loop: fill rows ----
    // rolling max/min helpers via deque would be ideal; N=5k so O(N*W) for
    // small windows is fine, but for 63/200 use running scan with stride.
    for i in 0..n {
        let mut f = [0.0_f64; N_FEATURES];
        let ci = c[i].max(1e-9);

        // helper closures for lookbacks
        let ret = |k: usize| -> f64 {
            if i >= k {
                c[i] / c[i - k] - 1.0
            } else {
                0.0
            }
        };
        let hh = |k: usize| -> f64 {
            let s = i.saturating_sub(k - 1);
            let mut m = h[s];
            for j in (s + 1)..=i {
                if h[j] > m {
                    m = h[j];
                }
            }
            m
        };
        let ll = |k: usize| -> f64 {
            let s = i.saturating_sub(k - 1);
            let mut m = l[s];
            for j in (s + 1)..=i {
                if l[j] < m {
                    m = l[j];
                }
            }
            m
        };

        // 0..6 returns
        f[0] = clip3(if i > 0 { (c[i] / c[i - 1] - 1.0) * 40.0 } else { 0.0 });
        f[1] = clip3(ret(5) * 12.0);
        f[2] = clip3(ret(10) * 8.0);
        f[3] = clip3(ret(21) * 5.0);
        f[4] = clip3(ret(63) * 2.8);
        f[5] = clip3(if i >= 21 { (c[i] / c[i - 21]).ln() * 5.0 } else { 0.0 });
        // sign sum 10
        {
            let mut s = 0.0;
            for k in 1..=10 {
                if i >= k {
                    if c[i - k + 1] > c[i - k] {
                        s += 1.0;
                    } else if c[i - k + 1] < c[i - k] {
                        s -= 1.0;
                    }
                }
            }
            f[6] = s / 10.0;
        }
        // 7..13 trend
        f[7] = z_sp10[i];
        f[8] = z_sp20[i];
        f[9] = z_sp50[i];
        f[10] = z_sp200[i];
        f[11] = z_esp12[i];
        f[12] = z_esp26[i];
        f[13] = clip3(
            ((sma10[i] - sma20[i]) / sma20[i].max(1e-9)) * 60.0,
        );
        // 14..16 rsi -> [-1,1]
        f[14] = (rsi2[i] / 50.0 - 1.0).clamp(-1.0, 1.0);
        f[15] = (rsi14[i] / 50.0 - 1.0).clamp(-1.0, 1.0);
        f[16] = (rsi28[i] / 50.0 - 1.0).clamp(-1.0, 1.0);
        // 17..19 macd
        f[17] = clip3((macd_line[i] / ci) * 400.0);
        f[18] = clip3((macd_sig[i] / ci) * 400.0);
        f[19] = clip3(((macd_line[i] - macd_sig[i]) / ci) * 900.0);
        // 20..23 bollinger
        {
            let mid = bb_mid[i].max(1e-9);
            let sd = bb_sd[i];
            let up = mid + 2.0 * sd;
            let lo = (mid - 2.0 * sd).max(1e-9);
            let pctb = ((ci - lo) / (up - lo).max(1e-9)).clamp(0.0, 1.0);
            f[20] = pctb * 2.0 - 1.0;
            f[21] = clip3(((up - lo) / mid) * 14.0);
            f[22] = clip3((ci - mid) / sd);
            let bw = (up - lo) / mid;
            let avg = bw120_mean[i].max(1e-9);
            f[23] = clip3((bw / avg - 1.0) * 3.0);
        }
        // 24..30 vol
        f[24] = clip3((atr14v[i] / ci) * 30.0 - 0.6);
        f[25] = clip3((atr14v[i] / atr50v[i].max(1e-9) - 1.0) * 3.0);
        f[26] = clip3((rv20[i] - 0.20) * 4.0);
        f[27] = clip3((rv5[i] / rv20[i].max(1e-9) - 1.0) * 2.0);
        f[28] = clip3((rv20[i] / rv63[i].max(1e-9) - 1.0) * 2.0);
        f[29] = clip3((park[i] - 0.20) * 4.0);
        f[30] = clip3(((h[i] - l[i]) / atr14v[i].max(1e-9) - 1.0) * 1.2);
        // 31..35 momentum
        f[31] = clip3(ret(12) * 7.0);
        f[32] = clip3(ret(126) * 2.0);
        {
            let h14 = hh(14);
            let l14 = ll(14);
            let rng = (h14 - l14).max(1e-9);
            let k = ((ci - l14) / rng).clamp(0.0, 1.0);
            f[33] = k * 2.0 - 1.0;
            // stoch D approx via last-3 K mean
            let mut ks: f64 = 0.0;
            let mut cnt: f64 = 0.0;
            for b in 0..3 {
                if i >= b {
                    let ii = i - b;
                    let h14b = {
                        let s = ii.saturating_sub(13);
                        let mut m = h[s];
                        for j in (s + 1)..=ii {
                            if h[j] > m {
                                m = h[j];
                            }
                        }
                        m
                    };
                    let l14b = {
                        let s = ii.saturating_sub(13);
                        let mut m = l[s];
                        for j in (s + 1)..=ii {
                            if l[j] < m {
                                m = l[j];
                            }
                        }
                        m
                    };
                    ks += ((c[ii] - l14b) / (h14b - l14b).max(1e-9)).clamp(0.0, 1.0);
                    cnt += 1.0;
                }
            }
            f[34] = (ks / cnt.max(1.0_f64)) * 2.0 - 1.0;
            f[35] = -f[33]; // WPR mirrors stoch
        }
        // 36..40 volume
        f[36] = clip3((v[i] / vsma20[i].max(1e-9) - 1.0) * 1.5);
        f[37] = ((lv[i] - lv_mean60[i]) / lv_sd60[i]).clamp(-3.0, 3.0) / 3.0;
        f[38] = clip3((dv[i] / dvsma20[i].max(1e-9) - 1.0) * 1.5);
        f[39] = clip3((obv[i] / obv_sma20[i].abs().max(1e9) - 1.0) * 2.0 + (obv[i] - obv_sma20[i]).signum() * 0.15);
        f[40] = z_amihud[i];
        // 41..45 calendar (trading-day calendar approx)
        {
            let dow = (i % 5) as f64;
            f[41] = (2.0 * std::f64::consts::PI * dow / 5.0).sin();
            f[42] = (2.0 * std::f64::consts::PI * dow / 5.0).cos();
            let mon = ((i / 21) % 12) as f64;
            f[43] = (2.0 * std::f64::consts::PI * mon / 12.0).sin();
            f[44] = (2.0 * std::f64::consts::PI * mon / 12.0).cos();
            let in_month = i % 21;
            f[45] = if in_month <= 2 || in_month >= 19 { 1.0 } else { -0.2 };
        }
        // 46..55 structure
        {
            let h63 = hh(63);
            let l63 = ll(63);
            f[46] = ((ci - h63) / h63.max(1e-9)).clamp(-0.5, 0.0) * 4.0; // DD63 [-2,0]->clip
            f[47] = ((ci - l63) / l63.max(1e-9)).clamp(0.0, 2.0) * 1.2 - 0.4;
            // days since high/low
            let mut bh = 0;
            let mut mv = f64::NEG_INFINITY;
            for k in 0..63.min(i + 1) {
                if h[i - k] >= mv {
                    mv = h[i - k];
                    bh = k;
                }
            }
            let mut bl = 0;
            let mut mv2 = f64::INFINITY;
            for k in 0..63.min(i + 1) {
                if l[i - k] <= mv2 {
                    mv2 = l[i - k];
                    bl = k;
                }
            }
            f[48] = 1.0 - (bh as f64 / 63.0) * 2.0;
            f[49] = 1.0 - (bl as f64 / 63.0) * 2.0;
            let atr = atr14v[i].max(1e-9);
            let prev_c = if i > 0 { c[i - 1] } else { o[i] };
            f[50] = clip3(((o[i] - prev_c) / atr) * 1.5);
            f[51] = (((ci - l[i]) / (h[i] - l[i]).max(1e-9)).clamp(0.0, 1.0)) * 2.0 - 1.0;
            f[52] = (((ci - l63) / (h63 - l63).max(1e-9)).clamp(0.0, 1.0)) * 2.0 - 1.0;
            let h20 = hh(20);
            let l20 = ll(20);
            f[53] = ((ci - h20) / h20.max(1e-9)).clamp(-0.4, 0.0) * 4.0;
            f[54] = ((ci - l20) / l20.max(1e-9)).clamp(0.0, 0.6) * 2.4 - 0.5;
            // efficiency 21: net change / path
            let mut path = 1e-9;
            for k in 1..=21.min(i) {
                path += (c[i - k + 1] - c[i - k]).abs();
            }
            let net = (ci - c[i.saturating_sub(21)]).abs();
            f[55] = (net / path).clamp(0.0, 1.0) * 2.0 - 1.0;
        }

        // sanitize: any non-finite (bad ticks, 0-volume flatlines) -> 0
        for v in f.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        out[i] = f;
    }
    let _ = tr;
    out
}
