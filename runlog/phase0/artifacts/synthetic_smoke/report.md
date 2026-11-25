# Phase 0 — Corrected v1 baseline

Generated 2026-11-25T16:43:55+00:00 · pfo 2.0.0a0 · source `synthetic`

> **Synthetic smoke run.** Prices come from a random factor model with no exploitable signal. These numbers only prove the pipeline runs end-to-end; they say nothing about real markets.

### Universe: `synthetic` (USD, 20 assets)

- Out-of-sample: **2018-09-14 → 2023-07-12**, 1259 unique trading days, 10 non-overlapping folds (train 504d, test 126d)
- Rebalance every 21d · execution lag 1d · costs 10.0 bps per unit turnover · data source `synthetic`

| Strategy | Sharpe (95% CI) | CAGR | Vol | Max DD | CVaR95 (daily) | Turnover/yr | Cost drag/yr |
|---|---|---|---|---|---|---|---|
| equal_weight | 0.84 (-0.06, 1.75) | 7.1% | 8.6% | -12.9% | -1.2% | 1.41 | 0.1% |
| min_variance | 0.82 (-0.17, 1.74) | 7.3% | 9.0% | -15.7% | -1.2% | 1.85 | 0.2% |
| markowitz_v1 | 0.83 (-0.14, 1.75) | 7.3% | 9.0% | -15.8% | -1.2% | 1.82 | 0.2% |
| rl_ppo | -0.14 (-1.14, 0.84) | -3.3% | 15.6% | -54.7% | -2.4% | 10.33 | 1.0% |

| vs equal weight | Δ Sharpe (95% CI) | p (bootstrap) | mean monthly excess | NW t | p (NW) | folds won |
|---|---|---|---|---|---|---|
| min_variance | -0.02 (-0.25, 0.22) | 0.826 | 0.02% | 0.25 | 0.807 | 4/10 |
| markowitz_v1 | -0.01 (-0.25, 0.23) | 0.918 | 0.03% | 0.32 | 0.754 | 4/10 |
| rl_ppo | -0.98 (-1.85, -0.06) | 0.030 | -0.76% | -1.17 | 0.248 | 3/10 |

## What v1 claimed (not comparable — produced with look-ahead and overlapping windows)

| Strategy | v1 Sharpe | v1 CAGR | v1 Max DD |
|---|---|---|---|
| rl_ppo | 1.47 | 20.0% | -32.8% |
| markowitz_v1 | 0.77 | 8.7% | -42.3% |
| equal_weight | 1.24 | 16.9% | -41.5% |

v1 universe: 40 assets, NSE and NYSE dates intersected, no FX. v1 windows: 30 overlapping 126-day windows stepped by 21 days (735 unique days counted as 3,780). v1 RL p-value vs EW: 4.3e-22.
