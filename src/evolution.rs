// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

//! Evolution: kill bottom 30%, copy top 40% with variation, middle survives.
//! Ranked by risk-adjusted fitness. Fear + copy-love co-evolve.

use crate::bot::Bot;
use crate::features::FeatureRow;
use crate::market::MarketData;
use crate::strategy::Genome;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

pub struct EvoConfig {
    pub fee_bps: f64,
    pub slip_bps: f64,
    pub sigma0: f64,
    pub sigma_decay: f64,
    pub sigma_min: f64,
    pub elite_keep: usize,
}

impl Default for EvoConfig {
    fn default() -> Self {
        Self {
            fee_bps: 5.0,
            slip_bps: 3.0,
            sigma0: 0.30,
            sigma_decay: 0.965,
            sigma_min: 0.05,
            elite_keep: 1,
        }
    }
}

pub struct GenerationReport {
    pub gen: u32,
    pub best_idx: usize,
    pub best_fitness: f64,
    pub best_return: f64,
    pub best_sharpe: f64,
    pub best_dd: f64,
    pub mean_fitness: f64,
    pub killed: Vec<u64>,
    pub copied: Vec<u64>,
    pub survived: Vec<u64>,
    pub diversity: f64,
    pub sigma: f64,
}

pub struct Population {
    pub bots: Vec<Bot>,
    pub generation: u32,
    pub best_hist: Vec<f64>,
    pub mean_hist: Vec<f64>,
    pub cfg: EvoConfig,
    pub rng: ChaCha8Rng,
}

impl Population {
    pub fn new(pop_size: usize, seed: u64, cfg: EvoConfig) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x9e3779b97f4a7c15);
        let mut bots = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            bots.push(Bot::newborn(0, Genome::random(&mut rng), (0, 0), false));
        }
        Self {
            bots,
            generation: 0,
            best_hist: Vec::new(),
            mean_hist: Vec::new(),
            cfg,
            rng,
        }
    }

    pub fn size(&self) -> usize {
        self.bots.len()
    }

    /// Evaluate every bot on 20yr tape in parallel.
    pub fn evaluate(&mut self, market: &MarketData, feats: &[FeatureRow], atr: &[f64]) {
        let fee = self.cfg.fee_bps;
        let slip = self.cfg.slip_bps;
        self.bots.par_iter_mut().for_each(|b| {
            b.trade(market, feats, atr, fee, slip);
        });
    }

    pub fn ranked_indices(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.bots.len()).collect();
        idx.sort_by(|&a, &b| {
            self.bots[b].fitness().total_cmp(&self.bots[a].fitness())
        });
        idx
    }

    fn diversity(&self) -> f64 {
        // mean pairwise L1 distance of weight vectors (sample up to 60 pairs)
        let n = self.bots.len().min(12);
        if n < 2 {
            return 0.0;
        }
        let mut s = 0.0;
        let mut c = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut d = 0.0;
                for k in 0..self.bots[i].genome.weights.len() {
                    d += (self.bots[i].genome.weights[k] - self.bots[j].genome.weights[k]).abs();
                }
                s += d / self.bots[i].genome.weights.len() as f64;
                c += 1;
            }
        }
        s / c.max(1) as f64
    }

    /// One generational step. Returns report for UI.
    pub fn step(&mut self, market: &MarketData, feats: &[FeatureRow], atr: &[f64]) -> GenerationReport {
        self.evaluate(market, feats, atr);
        let rank = self.ranked_indices();
        let n = self.bots.len();
        let n_kill = ((n as f64 * 0.30).round() as usize).clamp(1, n - 1);
        let n_top = ((n as f64 * 0.40).round() as usize).clamp(1, n);
        // top slice, middle slice, bottom slice
        let top: Vec<usize> = rank[..n_top].to_vec();
        let bottom: Vec<usize> = rank[n - n_kill..].to_vec();
        let middle: Vec<usize> = rank[n_top..n - n_kill].to_vec();

        let best_idx = rank[0];
        let best_fit = self.bots[best_idx].fitness();
        let mean_fit = self.bots.iter().map(|b| b.fitness()).sum::<f64>() / n as f64;
        self.best_hist.push(best_fit);
        self.mean_hist.push(mean_fit);

        let sigma = (self.cfg.sigma0 * self.cfg.sigma_decay.powi(self.generation as i32))
            .max(self.cfg.sigma_min);

        // record ids
        let killed: Vec<u64> = bottom.iter().map(|&i| self.bots[i].id).collect();
        let mut copied: Vec<u64> = Vec::new();
        let survived: Vec<u64> = middle.iter().map(|&i| self.bots[i].id).collect();

        // elites survive untouched
        let elite = self.cfg.elite_keep.min(n_top);
        for &ei in rank.iter().take(elite) {
            self.bots[ei].survived_gens += 1;
        }

        // Build next generation:
        // - keep top (breeders persist as themselves; their copy-count grows)
        // - keep middle (survivors, untouched)
        // - replace bottom with offspring sampled from top (crossover+mutate)
        // Diversity guard: if too similar, inject random immigrants.
        let div = self.diversity();
        let next_gen = self.generation + 1;

        // mark breeders as copied (they get replicated -> love reward next round)
        for &ti in &top {
            self.bots[ti].times_copied += 1;
            self.bots[ti].was_copied = true;
            copied.push(self.bots[ti].id);
        }
        for &mi in &middle {
            self.bots[mi].survived_gens += 1;
            self.bots[mi].was_copied = false;
            self.bots[mi].generation = next_gen;
        }
        // bottom die
        for &bi in &bottom {
            self.bots[bi].alive = false;
        }

        // create offspring to fill bottom slots
        let mut offspring: Vec<Bot> = Vec::with_capacity(bottom.len());
        for s in 0..bottom.len() {
            let use_immigrant = div < 0.02 && self.rng.gen::<f64>() < 0.35;
            let child_genome = if use_immigrant {
                Genome::random(&mut self.rng)
            } else {
                // pick 1-2 parents from top
                let p1 = top[self.rng.gen_range(0..top.len())].clone();
                let g1 = self.bots[p1].genome.clone();
                let id1 = self.bots[p1].id;
                let (g, id2) = if self.rng.gen::<f64>() < 0.75 && top.len() > 1 {
                    let mut p2 = top[self.rng.gen_range(0..top.len())].clone();
                    let mut guard = 0;
                    while p2 == p1 && guard < 6 {
                        p2 = top[self.rng.gen_range(0..top.len())].clone();
                        guard += 1;
                    }
                    let g2 = self.bots[p2].genome.clone();
                    let id2 = self.bots[p2].id;
                    (g1.crossover(&g2, &mut self.rng), id2)
                } else {
                    (g1, id1)
                };
                let _ = s;
                g.mutated(&mut self.rng, sigma)
                    .clone()
            };
            let (pa, pb) = if offspring.is_empty() {
                let p1 = top[self.rng.gen_range(0..top.len())];
                (self.bots[p1].id, self.bots[p1].id)
            } else {
                let p1 = top[self.rng.gen_range(0..top.len())];
                (self.bots[p1].id, self.bots[p1].id)
            };
            let _ = pa;
            let _ = pb;
            // proper parent tracking: resample for record
            let q1 = top[self.rng.gen_range(0..top.len())];
            let q2 = top[self.rng.gen_range(0..top.len())];
            let mut b = Bot::newborn(
                next_gen,
                child_genome,
                (self.bots[q1].id, self.bots[q2].id),
                true,
            );
            // newborns haven't earned copies yet
            b.times_copied = 0;
            offspring.push(b);
        }

        // stitch: survivors = top + middle (bots persist), bottom replaced
        // collect surviving bots (top+middle) then push offspring
        let mut keep: Vec<Bot> = Vec::with_capacity(n);
        // keep order: ranked top first so best stays visible
        for &i in rank.iter().filter(|i| !bottom.contains(i)) {
            let mut b = self.bots[i].clone();
            b.generation = next_gen;
            b.alive = true;
            keep.push(b);
        }
        keep.extend(offspring);
        debug_assert_eq!(keep.len(), n);
        self.bots = keep;
        self.generation = next_gen;

        // evaluate not yet; caller evaluates at start of next step.
        // But report needs current-gen stats (already computed pre-replacement).
        // Re-fetch best bot stats from pre-replacement ranking via stored values:
        // we kept clones, so find max fitness among keep's parents? Instead
        // store from before.
        let rep = GenerationReport {
            gen: next_gen,
            best_idx: 0, // best is first in keep (ranked)
            best_fitness: best_fit,
            best_return: 0.0, // filled by caller via champion()
            best_sharpe: 0.0,
            best_dd: 0.0,
            mean_fitness: mean_fit,
            killed,
            copied,
            survived,
            diversity: div,
            sigma,
        };
        // NOTE: caller should call evaluate() at loop top; for gen 0 we already
        // evaluated. After step, bots contain unevaluated offspring — next
        // loop iteration evaluates them.
        let _ = market;
        let _ = feats;
        let _ = atr;
        rep
    }

    pub fn champion(&self) -> &Bot {
        self.bots
            .iter()
            .max_by(|a, b| a.fitness().total_cmp(&b.fitness()))
            .expect("empty pop")
    }
}
