# Phase 0 — Corrected v1 baseline

Generated 2026-11-25T17:59:36+00:00 · pfo 2.0.0a0 · source `v1`

### Universe: `nifty` (INR, 20 assets)

- Out-of-sample: **2012-11-19 → 2026-09-21**, 3401 unique trading days, 27 non-overlapping folds (train 504d, test 126d)
- Rebalance every 21d · execution lag 1d · costs 10.0 bps per unit turnover · data source `v1`

| Strategy | Sharpe (95% CI) | CAGR | Vol | Max DD | CVaR95 (daily) | Turnover/yr | Cost drag/yr |
|---|---|---|---|---|---|---|---|
| equal_weight | 1.15 (0.48, 1.80) | 18.2% | 15.6% | -34.9% | -2.2% | 0.63 | 0.1% |
| min_variance | 1.14 (0.52, 1.77) | 15.9% | 13.8% | -28.0% | -1.9% | 2.30 | 0.2% |
| markowitz_v1 | 1.15 (0.55, 1.78) | 16.1% | 13.9% | -27.5% | -1.9% | 2.67 | 0.3% |
| rl_ppo | 1.07 (0.48, 1.67) | 18.8% | 17.6% | -30.2% | -2.4% | 12.33 | 1.2% |

| vs equal weight | Δ Sharpe (95% CI) | p (bootstrap) | mean monthly excess | NW t | p (NW) | folds won |
|---|---|---|---|---|---|---|
| min_variance | -0.01 (-0.23, 0.19) | 0.938 | -0.19% | -1.61 | 0.109 | 15/27 |
| markowitz_v1 | -0.00 (-0.23, 0.19) | 0.990 | -0.17% | -1.41 | 0.162 | 16/27 |
| rl_ppo | -0.08 (-0.37, 0.18) | 0.580 | 0.07% | 0.37 | 0.711 | 8/27 |

### Universe: `sp500` (USD, 18 assets)

- Out-of-sample: **2013-04-18 → 2026-10-23**, 3401 unique trading days, 27 non-overlapping folds (train 504d, test 126d)
- Excluded (history covers < 95% of the calendar): ABBV, META
- Rebalance every 21d · execution lag 1d · costs 10.0 bps per unit turnover · data source `v1`

| Strategy | Sharpe (95% CI) | CAGR | Vol | Max DD | CVaR95 (daily) | Turnover/yr | Cost drag/yr |
|---|---|---|---|---|---|---|---|
| equal_weight | 1.42 (0.91, 2.06) | 26.9% | 17.9% | -31.4% | -2.7% | 0.65 | 0.1% |
| min_variance | 1.13 (0.63, 1.73) | 16.0% | 14.0% | -25.0% | -2.0% | 2.05 | 0.2% |
| markowitz_v1 | 1.22 (0.71, 1.83) | 18.6% | 14.9% | -24.4% | -2.2% | 3.43 | 0.3% |
| rl_ppo | 1.19 (0.72, 1.73) | 28.6% | 23.5% | -28.9% | -3.4% | 13.15 | 1.3% |

| vs equal weight | Δ Sharpe (95% CI) | p (bootstrap) | mean monthly excess | NW t | p (NW) | folds won |
|---|---|---|---|---|---|---|
| min_variance | -0.29 (-0.62, 0.03) | 0.077 | -0.78% | -3.97 | 0.000 | 10/27 |
| markowitz_v1 | -0.20 (-0.50, 0.10) | 0.194 | -0.59% | -3.11 | 0.002 | 10/27 |
| rl_ppo | -0.23 (-0.57, 0.06) | 0.139 | 0.18% | 0.71 | 0.479 | 9/27 |

## What v1 claimed (not comparable — produced with look-ahead and overlapping windows)

| Strategy | v1 Sharpe | v1 CAGR | v1 Max DD |
|---|---|---|---|
| rl_ppo | 1.47 | 20.0% | -32.8% |
| markowitz_v1 | 0.77 | 8.7% | -42.3% |
| equal_weight | 1.24 | 16.9% | -41.5% |

v1 universe: 40 assets, NSE and NYSE dates intersected, no FX. v1 windows: 30 overlapping 126-day windows stepped by 21 days (735 unique days counted as 3,780). v1 RL p-value vs EW: 4.3e-22.
