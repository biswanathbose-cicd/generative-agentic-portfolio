# Business-impact memo (auto-generated)

**Data:** synthetic - 24 store x department series, 1095 days. Findings here validate the *method*, not any retailer's reality.

## What was built
A 7-day-ahead unit-demand forecaster. Features use only information available 7 days before the target day
(lags >= 7, shifted rolling stats) plus planned promotions/prices. Evaluation is rolling-origin: 4 folds x 28 days,
each trained only on data known when the first forecast of the fold would have been issued. Hyper-parameters were tuned
on a time-ordered validation block that ends before the first test fold.

## Results (WAPE, lower is better)
| model | WAPE mean | WAPE std across folds | bias |
|---|---|---|---|
| seasonal naive (same weekday last week) | 0.195 | 0.016 | +0.005 |
| ridge (log target, log lags) | 0.113 | 0.011 | -0.001 |
| gradient boosting | 0.100 | 0.005 | -0.006 |
| *oracle that knows the true Poisson rate* | *0.055* | - | - |

Gradient boosting cuts WAPE by **49%** versus seasonal naive. The oracle row is the
irreducible count-noise floor of this synthetic data: on real data no such floor is knowable, so an equivalent
sanity check would be a held-out comparison against the planners' current forecast.

The 10th-90th percentile interval covers **73.7%** of actuals (nominal 80%), median width 0.32x actual units.

## Illustrative cost translation (ASSUMED parameters - replace with real ones)
Safety stock proxy = z(95%) x std of 7-day forecast error per series. Unit costs (USD): {'apparel': 18.0, 'electronics': 45.0, 'grocery': 3.0, 'home': 22.0}. Annual holding rate 20%.

| | annual holding cost of safety stock (USD, 24 series) |
|---|---|
| seasonal naive | 6,013 |
| gradient boosting | 3,302 |
| difference | 2,712 (45% lower) |

This ignores lead-time aggregation, demand correlation across series and stock-out cost, so treat it as an order-of-magnitude
argument for *why forecast error matters*, not as a savings forecast.

## Top drivers (permutation importance, final fold; MAE increase when shuffled)
- `roll_mean_28`: +30.33
- `lag_7`: +19.65
- `lag_21`: +17.00
- `lag_14`: +15.33
- `lag_28`: +14.04
- `roll_mean_7`: +8.01

## Risks / next steps
- Promotions are known in advance here; in practice promo calendars change late - evaluate with noisy promo flags.
- No new-item or cold-start handling; no hierarchy reconciliation (store -> region -> total).
- Monitor WAPE and bias per department weekly; alert on drift of `lag`/`roll` feature distributions.
