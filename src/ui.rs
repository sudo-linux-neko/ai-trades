// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

//! Terminal cinema: progress bars, sparklines, price charts, league tables.
//! Pure crossterm + indicatif, no heavy TUI framework, 10Hz refresh max.

use crate::bot::Bot;
use crate::evolution::Population;
use crate::market::MarketData;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

pub const TICKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Downsample + sparkline. No allocs beyond output string.
pub fn sparkline(vals: &[f64], width: usize) -> String {
    if vals.is_empty() || width == 0 {
        return String::new();
    }
    let step = (vals.len() as f64 / width as f64).max(1.0);
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut sampled = Vec::with_capacity(width);
    let mut i = 0.0;
    while (i as usize) < vals.len() && sampled.len() < width {
        sampled.push(vals[i as usize]);
        i += step;
    }
    for &v in &sampled {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    let span = (hi - lo).max(1e-9);
    let mut s = String::with_capacity(width * 3);
    for &v in &sampled {
        let k = (((v - lo) / span) * 7.0).round() as usize;
        s.push(TICKS[k.min(7)]);
    }
    s
}

/// ASCII price chart with min/max labels. height rows.
pub fn price_chart(closes: &[f64], width: usize, height: usize) -> Vec<String> {
    let mut rows = vec![String::new(); height];
    if closes.is_empty() {
        return rows;
    }
    let take = closes.len().min(width * 4);
    let data = &closes[closes.len() - take..];
    let step = (data.len() as f64 / width as f64).max(1.0);
    let mut sampled = Vec::with_capacity(width);
    let mut i = 0.0;
    while (i as usize) < data.len() && sampled.len() < width {
        sampled.push(data[i as usize]);
        i += step;
    }
    let lo = sampled.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = sampled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(1e-9);
    let mut grid = vec![vec![' '; sampled.len()]; height];
    for (x, &v) in sampled.iter().enumerate() {
        let y = (((v - lo) / span) * (height as f64 - 1.0)).round() as usize;
        let row = height - 1 - y;
        grid[row][x] = '●';
        // stem to baseline of trend
        if row + 1 < height {
            for r in (row + 1)..height {
                if grid[r][x] == ' ' {
                    grid[r][x] = '·';
                }
            }
        }
    }
    for (r, line) in grid.into_iter().enumerate() {
        let s: String = line.into_iter().collect();
        if r == 0 {
            rows[r] = format!("hi {:>8.2} {}", hi, s);
        } else if r == height - 1 {
            rows[r] = format!("lo {:>8.2} {}", lo, s);
        } else {
            rows[r] = format!("            {}", s);
        }
    }
    rows
}

pub fn gen_spark(hist: &[f64], width: usize) -> String {
    sparkline(hist, width)
}

pub struct Cinema {
    pub mp: MultiProgress,
    pub bar_gen: ProgressBar,
    pub bar_eval: ProgressBar,
}

impl Cinema {
    pub fn new(total_gens: u64, pop: u64) -> Self {
        let mp = MultiProgress::new();
        let bar_gen = mp.add(ProgressBar::new(total_gens));
        bar_gen.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} GEN {pos}/{len} [{bar:34.cyan/blue}] {msg}",
            )
            .unwrap()
            .progress_chars("█▓▒░─"),
        );
        let bar_eval = mp.add(ProgressBar::new(pop));
        bar_eval.set_style(
            ProgressStyle::with_template(
                "     └─ bots {pos}/{len} [{bar:34.magenta/red}] {msg}",
            )
            .unwrap()
            .progress_chars("█▓▒░─"),
        );
        Self { mp, bar_gen, bar_eval }
    }

    pub fn set_eval(&self, v: u64, msg: &str) {
        self.bar_eval.set_position(v);
        self.bar_eval.set_message(msg.to_string());
    }
    pub fn reset_eval(&self, len: u64) {
        self.bar_eval.set_length(len);
        self.bar_eval.set_position(0);
    }
    pub fn finish(&self) {
        self.bar_gen.finish_with_message("evolution complete");
        self.bar_eval.finish_and_clear();
    }
}

pub fn banner() -> &'static str {
    r#"
 ███████╗██╗   ██╗ ██████╗ ████████╗██████╗  █████╗ ██████╗ ███████╗██████╗
 ██╔════╝██║   ██║██╔═══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
 █████╗  ██║   ██║██║   ██║   ██║   ██████╔╝███████║██║  ██║█████╗  ██████╔╝
 ██╔══╝  ╚██╗ ██╔╝██║   ██║   ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ██╔══██╗
 ███████╗ ╚████╔╝ ╚██████╔╝   ██║   ██║  ██║██║  ██║██████╔╝███████╗██║  ██║
 ╚══════╝  ╚═══╝   ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
  Darwinian trading bots · fear of death · love of being copied · 20yr tape
"#
}

pub fn print_dashboard(
    pop: &Population,
    market: &MarketData,
    gen: u32,
    total: u32,
    best: &Bot,
    gen_time_ms: u128,
) {
    let w = term_width().min(100).max(50);
    let closes: Vec<f64> = market.candles.iter().map(|c| c.close).collect();
    let eq = &best.result.equity;
    println!(
        "\n┌─ GEN {:>3}/{:<3} ─────────────────────────────────────────────────┐",
        gen, total
    );
    println!(
        "│ best #{:<5} fit {:+.3} │ ret {:+.1}% │ Sharpe {:+.2} │ MaxDD {:.1}% │ {} │",
        best.id,
        best.result.fitness,
        best.result.total_return * 100.0,
        best.result.sharpe,
        best.result.max_dd * 100.0,
        format!("{}ms", gen_time_ms),
    );
    println!(
        "│ fear {:.2} · greed/copy {:.2} · risk {:.0}% · stop {:.1}x · take {:.1}x · hold {}d · {} │",
        best.genome.fear,
        best.genome.copy_drive,
        best.genome.risk_frac * 100.0,
        best.genome.stop_mult,
        best.genome.take_mult,
        best.genome.max_hold,
        if best.genome.allow_short { "L/S" } else { "long " },
    );
    println!("├─ PRICE (20yr tape, last ~4y) ────────────────────────────────────┤");
    for line in price_chart(&closes, (w - 16) as usize, 6) {
        println!("│ {} │", truncate(&line, (w - 4) as usize));
    }
    println!("├─ CHAMPION EQUITY ───────────────────────────────────────────────┤");
    let eq_spark = sparkline(eq, (w - 24) as usize);
    println!(
        "│ {} {:+.1}% │",
        eq_spark,
        best.result.total_return * 100.0
    );
    println!("├─ FITNESS ACROSS GENERATIONS ────────────────────────────────────┤");
    println!(
        "│ best {} │",
        gen_spark(&pop.best_hist, (w - 10) as usize)
    );
    println!(
        "│ mean {} │",
        gen_spark(&pop.mean_hist, (w - 10) as usize)
    );
    // league table top5
    let mut order: Vec<&Bot> = pop.bots.iter().collect();
    order.sort_by(|a, b| b.fitness().total_cmp(&a.fitness()));
    println!("├─ LEAGUE (top 5 / {} bots) ───────────────────────────────────────┤", pop.bots.len());
    println!("│  rank  bot    fitness   return   sharpe    dd    trades  fate     │");
    for (r, b) in order.iter().take(5).enumerate() {
        let fate = if r == 0 {
            "★ COPY ×N"
        } else if (r as f64) < pop.bots.len() as f64 * 0.4 {
            "⬆ copy  "
        } else if (r as f64) >= pop.bots.len() as f64 * 0.7 {
            "☠ die   "
        } else {
            "─ live  "
        };
        println!(
            "│  #{:<3}  {:<5}  {:+.3}   {:+.1}%   {:+.2}   {:.1}%   {:<5}  {} │",
            r + 1,
            b.id,
            b.result.fitness,
            b.result.total_return * 100.0,
            b.result.sharpe,
            b.result.max_dd * 100.0,
            b.result.num_trades,
            fate
        );
    }
    println!("└─────────────────────────────────────────────────────────────────┘");
}

fn term_width() -> usize {
    crossterm::terminal::size().map(|(w, _)| w as usize).unwrap_or(88)
}

fn truncate(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_string()
    } else {
        chars[..max].iter().collect()
    }
}

pub fn print_bot_autopsy(best: &Bot) {
    println!("\n╔═ BOT #{} AUTOPSY — what it learned about itself ═══════════════╗", best.id);
    println!("║ trades: {}  win-rate {:.1}%  PF {:.2}  avgWin {:.3}%  avgLoss {:.3}% ║",
        best.result.num_trades,
        best.result.win_rate * 100.0,
        best.result.profit_factor,
        best.result.avg_win * 100.0,
        best.result.avg_loss * 100.0,
    );
    if let Some(w) = &best.win_lesson {
        println!("╟─ WINS: {} ─────────────────────────────────────────────", w.verdict);
        for (k, v) in &w.top_drivers {
            println!("║   driver {:<14} contribution {:+.3}", k, v);
        }
    }
    if let Some(l) = &best.loss_lesson {
        println!("╟─ LOSSES: {} ───────────────────────────────────────", l.verdict);
        for (k, c) in l.mistake_counts.iter().take(5) {
            println!("║   mistake {:<28} x{}", k, c);
        }
        for (k, v) in &l.top_blame {
            println!("║   blame   {:<14} exposure {:+.3}", k, v);
        }
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
}
