"""Helper functions shared by the notebook and the Streamlit app (copied verbatim from the notebook)."""
import numpy as np
import pandas as pd

REFERENCE_YEAR = 2020          # the dataset was scraped in 2020 -> newest car is 2020
PREMIUM_MAKES = ["Audi", "BMW", "Mercedes"]

def add_features(df):
    """Row-wise feature engineering (nothing is learned from data -> no leakage)."""
    df = df.copy()
    # 1) Car age (years) - never negative
    df["car_age"] = (REFERENCE_YEAR - df["year"]).clip(lower=0)
    # 2) Mileage per year - car_age is clipped to >= 1 so brand-new cars never divide by zero
    df["mileage_per_year"] = df["mileage"] / df["car_age"].clip(lower=1)
    # 3) Premium brand flag
    df["is_premium_brand"] = df["Make"].isin(PREMIUM_MAKES).astype(int)
    # 4) Engine size relative to fuel economy (power-vs-economy proxy)
    df["engine_mpg_ratio"] = df["engineSize"] / df["mpg"]
    # 5) Flag: engine size was missing/invalid in the raw data (imputed later, on train only)
    df["engine_missing"] = df["engineSize"].isna().astype(int)
    return df


def rate_deal(predicted_price, seller_price):
    """Compare the model's predicted market price with the seller's asking price."""
    difference = predicted_price - seller_price          # > 0  -> seller is cheaper than market
    pct = difference / predicted_price * 100             # % cheaper (+) / more expensive (-)
    if pct > 10:
        label, emoji = "Great Deal", "🟢"
    elif pct > 5:
        label, emoji = "Good Deal", "🟢"
    elif pct >= -5:
        label, emoji = "Fair Price", "🟡"
    elif pct >= -10:
        label, emoji = "Slightly Overpriced", "🟠"
    else:
        label, emoji = "Overpriced", "🔴"
    if difference > 0:
        text = f"£{abs(difference):,.0f} cheaper"
    elif difference < 0:
        text = f"£{abs(difference):,.0f} more expensive"
    else:
        text = "same as market"
    return {"predicted_price": float(predicted_price), "seller_price": float(seller_price),
            "difference": float(difference), "difference_pct": float(pct),
            "difference_text": text, "rating": label, "emoji": emoji}


def predict_price(bundle, make, model, year, mileage, transmission, fuel_type, engine_size,
                  tax=None, mpg=None):
    """Predict the market price for ONE car. Missing tax / mpg are filled with typical
    values for the same make+model+fuel (computed from the TRAINING data only)."""
    d = bundle["defaults"]
    eng = f"{float(engine_size):.1f}"
    ref = (d["make_model_fuel_engine_trans"].get(f"{make}|{model}|{fuel_type}|{eng}|{transmission}")
           or d["make_model_fuel_engine"].get(f"{make}|{model}|{fuel_type}|{eng}")
           or d["make_model_fuel"].get(f"{make}|{model}|{fuel_type}")
           or d["make_model"].get(f"{make}|{model}")
           or d["global"])
    tax = ref["tax"] if tax is None else tax
    mpg = ref["mpg"] if mpg is None else mpg
    row = pd.DataFrame([{"Make": make, "model": model, "year": year, "mileage": mileage,
                         "transmission": transmission, "fuelType": fuel_type,
                         "tax": tax, "mpg": mpg, "engineSize": engine_size}])
    row = add_features(row)
    X = bundle["preprocessor"].transform(row[bundle["features"]])
    return float(bundle["model"].predict(X)[0])
