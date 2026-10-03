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
| Final model | HistGradientBoosting (XGBoost stand-in) (tuned) |
| **Test R²** | **0.968** |
| **Test MAE** | **£1,094** |
| Test RMSE | £1,773 |
| Typical error | median ≈ £713 (≈ 5% of the price) |

Validation comparison of the five models (all trained on `log(1+price)`, metrics in £):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.924 | £1,599 | £2,740 |
| Decision Tree | 0.920 | £1,472 | £2,799 |
| Random Forest | 0.957 | £1,139 | £2,046 |
| Gradient Boosting | 0.937 | £1,489 | £2,484 |
| HistGradientBoosting (XGBoost stand-in) | 0.958 | £1,149 | £2,025 |

> The 5th model in the saved results is scikit-learn's `HistGradientBoostingRegressor`, used automatically when `xgboost` is not installed. With `pip install xgboost` (pre-installed on Google Colab) the notebook uses **XGBoost** instead and its numbers will differ slightly.

Top features (permutation importance): 

| Feature | Importance (drop in R²) |
|---|---|
| `car_age` | 0.251 |
| `engine_mpg_ratio` | 0.138 |
| `engineSize` | 0.125 |
| `is_premium_brand` | 0.110 |
| `transmission` | 0.106 |
| `model` | 0.073 |
| `mpg` | 0.065 |
| `mileage` | 0.047 |

## Pipeline

1. **Load** 9 manufacturer files, add a `Make` column to each *before* combining (99,187 rows).
2. **Combine** — fix Hyundai's `tax(£)` column and leading spaces in model names.
3. **Clean** (each decision explained in the notebook): 1,475 duplicates removed; 3 impossible years (2060, 1970) dropped; 265 engine sizes of 0 on non-electric cars → missing → imputed from the training set; mpg clipped to [10, 150]; price and mileage extremes investigated and kept. → 97,709 rows.
4. **EDA** — 7 visualisations, each with a written conclusion.
5. **Split** 70 / 15 / 15 (train / validation / test); test set used once.
6. **Feature engineering** — `car_age` (reference year 2020), `mileage_per_year` (safe for age 0), `is_premium_brand`, `engine_mpg_ratio`, `engine_missing`.
7. **Preprocessing** — median imputation + scaling for numbers, One-Hot Encoding for categories, **fitted on the training set only**.
8. **Models** — Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost (or HistGradientBoosting).
9. **Evaluation** — R², MAE, RMSE; **overfitting check** (train vs validation).
10. **Tuning** — RandomizedSearchCV (3-fold CV, preprocessing inside the pipeline).
11. **Final model** chosen on validation, evaluated **once** on the test set.
12. **Model understanding** — actual vs predicted, feature importance, error analysis, what-if on age and mileage.
13. **Smart Deal Advisor** — 🟢 Great (>10% cheaper), 🟢 Good (5–10%), 🟡 Fair (±5%), 🟠 Slightly overpriced (5–10%), 🔴 Overpriced (>10%).
14. **Streamlit app**.

On the real test listings the advisor rates 49% as fair, 25% as good/great deals and 26% as overpriced.

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

**Google Colab:** upload `AutoWorth_AI.ipynb` (and the `data/` folder or the dataset zip when asked) → *Runtime → Run all*. Training and tuning take about 10–15 minutes.

**Locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the notebook with Jupyter to re-train and regenerate `models/autoworth_model.joblib`.

## Limitations

* Data was scraped in 2020 — today's prices differ; UK market, nine makes only.
* No trim level, condition, options, service history or location in the data; rare expensive cars (G-Class, R8, Mustang) have the largest £ errors.
* The app fills `tax` and `mpg` with typical values when the user does not enter them.

## Dataset

[100,000 UK Used Car Dataset](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) (Kaggle).
