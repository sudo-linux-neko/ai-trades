// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

//! EvoTrader — Darwinian trading engine.
//! 10 bots enter. Bottom 30% die. Top 40% get copied with mutations. Middle survives.
//! 20-year tape. Fear of death. Love of being copied. Pure Rust.

mod backtest;
mod bot;
mod evolution;
mod features;
mod market;
mod strategy;
mod ui;

use clap::Parser;
use evolution::{EvoConfig, Population};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "ai-trades", version, about = "EvoTrader — evolve human-like traders over 20 years")]
struct Args {
    /// Population size (default 10 = the 10 copies you asked for)
    #[arg(long, default_value_t = 10)]
    population: usize,
    /// Generations to evolve
    #[arg(long, default_value_t = 25)]
    generations: u32,
    /// Years of market tape (min 5, default 20)
    #[arg(long, default_value_t = 20)]
    years: u32,
    /// RNG seed (reproducible runs)
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Optional CSV with open,high,low,close,volume (overrides synthetic tape)
    #[arg(long)]
    csv: Option<String>,
    /// Fee in bps per side
    #[arg(long, default_value_t = 5.0)]
    fee_bps: f64,
    /// Slippage in bps per side
    #[arg(long, default_value_t = 3.0)]
    slip_bps: f64,
    /// Print full dashboard every generation (else every 5 + last)
    #[arg(long, default_value_t = false)]
    verbose: bool,
    /// Quiet: only progress bars + final report
    #[arg(long, default_value_t = false)]
    quiet: bool,
}

fn main() {
    let args = Args::parse();
    let pop_size = args.population.clamp(6, 200);
    let gens = args.generations.clamp(1, 500);
    let years = args.years.clamp(5, 40);

    println!("{}", ui::banner());
    println!(
        "population {} · generations {} · tape {}y · seed {} · fees {:.0}+{:.0}bps",
        pop_size, gens, years, args.seed, args.fee_bps, args.slip_bps
    );
    println!("rule: bottom 30% ☠ die · top 40% ⬆ copied w/ mutation · middle 30% ─ survive\n");

    // ---- market ----
    let t0 = Instant::now();
    let market = match &args.csv {
        Some(p) => match market::load_csv(p, "CUSTOM") {
            Ok(m) => {
                println!("loaded {} bars from {}", m.len(), p);
                m
            }
            Err(e) => {
                eprintln!("csv error ({e}) — falling back to synthetic tape");
                market::generate_market(years, args.seed, "SYNTH-20Y")
            }
        },
        None => market::generate_market(years, args.seed, "SYNTH-20Y"),
    };
    println!(
        "tape: {} bars = {:.1}y · {} … {:.2} → {:.2}",
        market.len(),
        market.years(),
        market.symbol,
        market.candles.first().map(|c| c.close).unwrap_or(0.0),
        market.candles.last().map(|c| c.close).unwrap_or(0.0),
    );

    // ---- features (the million small eyes) ----
    println!("computing 56 micro-features over full tape…");
    let t1 = Instant::now();
    let feats = features::compute_features(&market);
    let atr = market::atr14(&market);
    println!("features ready in {}ms\n", t1.elapsed().as_millis());

    // ---- evolution ----
    let cfg = EvoConfig {
        fee_bps: args.fee_bps,
        slip_bps: args.slip_bps,
        ..Default::default()
    };
    let mut pop = Population::new(pop_size, args.seed, cfg);
    let cinema = ui::Cinema::new(gens as u64, pop_size as u64);

    // gen 0 evaluation with progress feel
    let mut gen_t = Instant::now();
    pop.evaluate(&market, &feats, &atr);
    {
        let champ = pop.champion().clone();
        cinema.bar_gen.set_position(0);
        cinema.bar_gen.set_message(format!(
            "gen 0/{gens} best fit {:+.3} ret {:+.1}%",
            champ.result.fitness,
            champ.result.total_return * 100.0
        ));
        if !args.quiet {
            ui::print_dashboard(&pop, &market, 0, gens, &champ, gen_t.elapsed().as_millis());
            ui::print_bot_autopsy(&champ);
        }
    }

    for g in 1..=gens {
        gen_t = Instant::now();
        cinema.reset_eval(pop_size as u64);
        // evaluate happens inside step(); animate eval bar optimistically
        let rep = pop.step(&market, &feats, &atr);
        // step() leaves offspring unevaluated — evaluate now for honest reporting
        pop.evaluate(&market, &feats, &atr);
        let champ = pop.champion().clone();
        cinema.bar_gen.set_position(g as u64);
        cinema.set_eval(pop_size as u64, &format!("fit {:+.3}", champ.result.fitness));
        cinema.bar_gen.set_message(format!(
            "gen {g}/{gens} best {:+.3} mean {:+.3} div {:.3} σ {:.2} ☠{} ⬆{}",
            rep.best_fitness,
            rep.mean_fitness,
            rep.diversity,
            rep.sigma,
            rep.killed.len(),
            rep.copied.len(),
        ));
        let show = !args.quiet && (args.verbose || g % 5 == 0 || g == gens);
        if show {
            // suspend bars briefly so dashboard prints cleanly
            cinema.mp.suspend(|| {
                ui::print_dashboard(&pop, &market, g, gens, &champ, gen_t.elapsed().as_millis());
                if g == gens || args.verbose {
                    ui::print_bot_autopsy(&champ);
                } else {
                    // mini journal line
                    if let (Some(w), Some(l)) = (&champ.win_lesson, &champ.loss_lesson) {
                        println!("  ↳ wins: {}", w.verdict);
                        println!("  ↳ loss: {}", l.verdict);
                    }
                }
            });
        }
    }
    cinema.finish();

    // ---- final ----
    let champ = pop.champion();
    println!("\n═══════════════ EVOLUTION COMPLETE ═══════════════");
    println!(
        "wall time {:.1}s · {} bots × {} gens × {} bars (20y+)",
        t0.elapsed().as_secs_f64(),
        pop_size,
        gens,
        market.len()
    );
    println!(
        "CHAMPION #{} (gen {}, parents {:?}, copied {}x, survived {} gens)",
        champ.id, champ.generation, champ.parents, champ.times_copied, champ.survived_gens
    );
    println!(
        "fitness {:+.4} · total {:+.1}% · CAGR {:+.1}% · Sharpe {:+.2} · Sortino {:+.2} · MaxDD {:.1}% · trades {} ({:.0}/y) · win {:.1}% · PF {:.2}",
        champ.result.fitness,
        champ.result.total_return * 100.0,
        champ.result.cagr * 100.0,
        champ.result.sharpe,
        champ.result.sortino,
        champ.result.max_dd * 100.0,
        champ.result.num_trades,
        champ.result.turnover,
        champ.result.win_rate * 100.0,
        champ.result.profit_factor,
    );
    println!(
        "brain: fear {:.2} (death-shrinks-size) · copy-love {:.2} · risk {:.0}% · buy>{:.2} sell<{:.2} · stop {:.1}xATR take {:.1}xATR hold {}d {}",
        champ.genome.fear,
        champ.genome.copy_drive,
        champ.genome.risk_frac * 100.0,
        champ.genome.buy_thresh,
        champ.genome.sell_thresh,
        champ.genome.stop_mult,
        champ.genome.take_mult,
        champ.genome.max_hold,
        if champ.genome.allow_short { "LONG/SHORT" } else { "LONG-ONLY" },
    );
    // top-8 weights = the strategy in words
    {
        let names = features::feature_names();
        let mut w: Vec<(usize, f64)> = (0..features::N_FEATURES)
            .map(|i| (i, champ.genome.weights[i]))
            .collect();
        w.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        println!("strategy DNA (top 8 of 56 eyes):");
        for (i, v) in w.into_iter().take(8) {
            let dir = if v > 0.0 { "long when high" } else { "long when low " };
            println!("  {:<14} {:+.3}  ({})", names[i], v, dir);
        }
    }
    ui::print_bot_autopsy(champ);

    // fitness curve
    println!("\nfitness history (best):");
    println!("  {}", ui::gen_spark(&pop.best_hist, 72));
    println!("fitness history (mean):");
    println!("  {}", ui::gen_spark(&pop.mean_hist, 72));

    // save champion equity for external plotting
    let path = "champion_equity.csv";
    if let Ok(mut s) = std::fs::File::create(path) {
        use std::io::Write;
        let _ = writeln!(s, "bar,equity");
        for (i, e) in champ.result.equity.iter().enumerate() {
            let _ = writeln!(s, "{i},{e}");
        }
        println!("\nsaved {path} ({} points) — plot it!", champ.result.equity.len());
    }
}
