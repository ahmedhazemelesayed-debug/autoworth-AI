# 🚗 AutoWorth AI — Used Car Price & Deal Advisor

An individual final Machine Learning project. AutoWorth AI predicts the **fair market price** of a used UK car and tells you whether the **seller's asking price is a good deal**.

```
Estimated Market Price: £18,500
Seller Price:           £17,000
Difference:             £1,500 cheaper
Deal Rating:            🟢 Good Deal
```

## Results

| | |
|---|---|
| Final model | XGBoost (tuned) |
| **Test R²** | **0.967** |
| **Test MAE** | **£1,114** |
| Test RMSE | £1,807 |
| Typical error | ≈ 5% of the car's price (median % error in every price band) |

Validation comparison of the five models (all trained on `log(1+price)`, metrics in £):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.924 | £1,599 | £2,740 |
| Decision Tree | 0.930 | £1,468 | £2,622 |
| Random Forest | 0.957 | £1,139 | £2,045 |
| Gradient Boosting | 0.937 | £1,490 | £2,493 |
| XGBoost | 0.961 | £1,122 | £1,964 |

Tuning (RandomizedSearchCV, 8 candidates × 3-fold CV): validation R² 0.9607 → 0.9608, MAE £1,122 → £1,117. The gain is small: the default XGBoost was already well configured. Best parameters: subsample: 1.0, n_estimators: 800, min_child_weight: 3, max_depth: 10, learning_rate: 0.03, colsample_bytree: 0.6.

Overfitting check: the Decision Tree overfits (train R² 0.9996 vs validation 0.930); XGBoost generalises well (gap 0.015).

Top features (permutation importance):

| Feature | Importance (drop in R²) |
|---|---|
| `car_age` | 0.188 |
| `engine_mpg_ratio` | 0.126 |
| `engineSize` | 0.119 |
| `mileage` | 0.102 |
| `transmission` | 0.087 |
| `is_premium_brand` | 0.077 |
| `model` | 0.066 |
| `mpg` | 0.059 |

## Pipeline

1. **Load** the 9 manufacturer files, add a `Make` column to each *before* combining (99,187 rows).
2. **Combine** — fix Hyundai's `tax(£)` column and leading spaces in model names.
3. **Clean** (each decision is explained in the notebook): 1,475 duplicates removed; 3 impossible years (2060, 1970) dropped; 265 engine sizes of 0 on non-electric cars → missing → imputed from the training set; mpg clipped to [10, 150]; price and mileage extremes investigated and kept → 97,709 rows.
4. **EDA** — 7 visualisations, each with a written conclusion.
5. **Split** 70 / 15 / 15 (train / validation / test); the test set is used once.
6. **Feature engineering** — `car_age` (reference year 2020), `mileage_per_year` (safe for age 0), `is_premium_brand`, `engine_mpg_ratio`, `engine_missing`.
7. **Preprocessing** — median imputation + scaling for numbers, One-Hot Encoding for categories, **fitted on the training set only**.
8. **Models** — Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
9. **Evaluation** — R², MAE, RMSE, plus the overfitting check (train vs validation).
10. **Tuning** — RandomizedSearchCV (preprocessing inside the pipeline, so there is no leakage in CV).
11. **Final model** chosen on validation, evaluated **once** on the test set.
12. **Model understanding** — actual vs predicted, feature importance, error analysis, what-if on age and mileage.
13. **Smart Deal Advisor** — 🟢 Great (>10% cheaper), 🟢 Good (5–10%), 🟡 Fair (±5%), 🟠 Slightly overpriced (5–10%), 🔴 Overpriced (>10%).
14. **Streamlit app**.

On the real test listings the advisor rates 48% as fair, 26% as good/great deals and 26% as overpriced.

## Repository structure

```
├── AutoWorth_AI.ipynb          # full notebook (open in Google Colab)
├── app.py                      # Streamlit app
├── autoworth_core.py           # feature engineering + deal advisor + predict helper
├── models/autoworth_model.joblib
├── data/                       # the 9 manufacturer CSV files
├── figures/                    # saved plots
├── results.json                # all reported numbers
├── AutoWorth_AI_Presentation.pptx
└── requirements.txt
```

## How to run

**Google Colab:** open `AutoWorth_AI.ipynb` in Colab, set `GITHUB_USER` in the first code cell, then *Runtime → Run all* (about 10–15 minutes). If you did not upload the repo, the notebook asks you to upload the dataset zip instead.

**Locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Re-run the notebook to regenerate `models/autoworth_model.joblib`.

## Limitations

* The data was scraped in 2020 — today's prices differ; UK market, nine makes only.
* No trim level, condition, options, service history or location in the data; rare expensive cars (G-Class, R8, Mustang) have the largest £ errors.
* The app fills `tax` and `mpg` with typical values when the user does not enter them.
* Cars with an imputed engine size can be badly mispredicted (e.g. one BMW X5 at −49%).

## Dataset

[100,000 UK Used Car Dataset](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) (Kaggle).
