# NFL Game Prediction of Spread, Total Points & Passing Yards

Predictive modeling of three NFL game outcomes for Weeks 7–11 of the 2025 season, using play-by-play data from the nflfastR database and custom-engineered team performance features.

---

## Overview

This project builds three separate models to predict:

- **Spread** — Home points minus away points (my primary contribution)
- **Total** — Combined points scored by both teams
- **Pass** — Combined passing yards for both teams

Each outcome required its own feature engineering approach and model selection process, evaluated using Mean Absolute Error (MAE) via cross-validation. All predictions were generated using only information available before game day to prevent data leakage.

---

## Data

- **Source:** nflfastR R package — play-by-play data for the full 2024 season and Weeks 1–6 of the 2025 season; additional 2020–2023 seasons used for Pass model training
- **Scope:** Regular-season games only (`season_type == "REG"`); 372 raw variables reduced to 56 relevant features
- **Outcome variables:** Spread, Total, and Pass were computed from aggregated game-level statistics

### Custom Engineered Features

| Feature | Description |
|---|---|
| **Offensive Score** | Composite z-score index: EPA/play, success rate, completion rate, TDs, yards per rush/play |
| **Defensive Score** | Composite z-score index (inverted): EPA allowed, yards allowed, sacks |
| **Passer Rating** | Computed using the official NFL formula from completions, yards, TDs, and interceptions |
| **Rolling averages** | 3-game lagged and season-to-date averages for points for/against, per team |
| **Team differentials** | Home minus away for all performance metrics, capturing relative team strength |

---

## Models

### Spread (primary contribution)
Multiple linear regression with **dampened coefficient weights** to reduce the outsized influence of extreme team performance differentials. Dampening factors were applied after examining the correlation structure of predictors:

```r
lm(actual_spread ~ spread_line +
   0.6 * points_for_diff + 0.5 * points_against_diff +
   0.35 * total_yards_diff + 0.35 * pass_yards_diff +
   0.35 * rush_yards_diff + 0.4 * plays_diff +
   0.5 * yards_per_play_diff, data = train)
```

The Vegas spread line and points differential were the strongest predictors. Dampening constrained unrealistic 14+ point predictions while preserving the model's ability to separate close games from blowouts.

### Total
Random Forest regression (via `tidymodels` and `ranger`) using rolling 3-game and season-to-date averages for points scored and allowed. Hyperparameters tuned via 3-fold cross-validation minimizing MAE. Recent offensive averages (`pf_l3`, `pf_season`) were the most important features by permutation importance.

### Pass
Gradient Boosting Machine (GBM) with 5,000 trees, interaction depth of 3, and shrinkage of 0.01. Predictors selected via stepwise regression (`stepAIC`, bidirectional). Optimal tree count determined by 5-fold cross-validation (`gbm.perf`). Final MAE: **18.04**, RMSE: **22.50**.

---

## Results Summary

| Model | Method | Primary Metric |
|---|---|---|
| Spread | Linear regression w/ dampening | MAE (cross-validated on 2024 season) |
| Total | Random Forest | MAE (3-fold CV) |
| Pass | Gradient Boosting (GBM) | MAE = 18.04, RMSE = 22.50 |

---

## Files

| File | Description |
|---|---|
| `SpreadRCode.Rmd` | R Markdown source — data cleaning, feature engineering, and spread model |
| `NFLProject.pdf` | Full written report covering all three models |

---

## Tools & Libraries

R · nflfastR · nflreadr · tidymodels · ranger · gbm · MASS (stepAIC) · tidyverse · slider

---

## Authors

Sarah Cao, Nick Copland, Sophia Draghici, Stephen Bridges, Daniel Larson  
*STOR 538 — October 2025*
