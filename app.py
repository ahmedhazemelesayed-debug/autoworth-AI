# -*- coding: utf-8 -*-
"""AutoWorth AI - Streamlit app.  Run:  streamlit run app.py"""
import os
import joblib
import streamlit as st
from autoworth_core import predict_price, rate_deal

st.set_page_config(page_title="AutoWorth AI", page_icon="🚗", layout="centered")


@st.cache_resource
def load_bundle():
    # works whether the model sits in a models/ folder or next to app.py
    for path in ("models/autoworth_model.joblib", "autoworth_model.joblib"):
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("autoworth_model.joblib not found - upload it to the repository.")


bundle = load_bundle()
opt = bundle["options"]

st.title("🚗 AutoWorth AI")
st.caption("Used-car price estimator & Smart Deal Advisor · UK market data (2020)")

col1, col2 = st.columns(2)
make = col1.selectbox("Make", sorted(opt["models_by_make"]))
model = col2.selectbox("Model", opt["models_by_make"][make])
key = f"{make}|{model}"

col3, col4 = st.columns(2)
y0, y1 = opt["year_range"]
year = col3.number_input("Year", min_value=y0, max_value=y1, value=min(2017, y1), step=1)
mileage = col4.number_input("Mileage", min_value=0, max_value=opt["max_mileage"], value=30000, step=1000)

col5, col6, col7 = st.columns(3)
transmission = col5.selectbox("Transmission", opt["transmissions_by_model"][key])
fuel = col6.selectbox("Fuel type", opt["fuels_by_model"][key])
engines = opt["engines_by_model"][key] or [0.0]
engine = col7.selectbox("Engine size (L)", engines, index=min(1, len(engines) - 1) if len(engines) > 1 else 0)

seller_price = st.number_input("Seller asking price (£)", min_value=100, max_value=300000, value=15000, step=100)

with st.expander("Advanced (optional): annual tax and MPG"):
    st.write("Leave unticked to use typical values for this make / model / engine.")
    use_manual = st.checkbox("Enter tax and MPG manually")
    tax = st.number_input("Road tax (£/year)", 0, 700, 145) if use_manual else None
    mpg = st.number_input("MPG", 10.0, 150.0, 55.0) if use_manual else None

if st.button("Estimate price & rate the deal", type="primary"):
    predicted = predict_price(bundle, make, model, int(year), float(mileage), transmission, fuel,
                              float(engine), tax=tax, mpg=mpg)
    r = rate_deal(predicted, seller_price)

    st.divider()
    a, b = st.columns(2)
    a.metric("💰 Estimated Market Price", f"£{r['predicted_price']:,.0f}")
    b.metric("🏷️ Seller Price", f"£{r['seller_price']:,.0f}")
    c, d = st.columns(2)
    c.metric("💵 Difference", r["difference_text"], f"{r['difference_pct']:+.1f}% vs market", delta_color="normal")
    d.metric("🚦 Deal Rating", f"{r['emoji']} {r['rating'].upper()}")

    banner = {"Great Deal": st.success, "Good Deal": st.success, "Fair Price": st.warning,
              "Slightly Overpriced": st.warning, "Overpriced": st.error}[r["rating"]]
    banner(f"{r['emoji']} {r['rating']}: the asking price is {r['difference_text']} than the estimated market price "
           f"({abs(r['difference_pct']):.1f}%).")
    if r["difference_pct"] > 30:
        st.info("⚠️ A price this far below the estimate can mean damage, a typo, or a scam - inspect the car carefully.")
    st.caption(f"Model: {bundle['model_name']} · test R² {bundle['test_metrics']['R2']:.3f} · "
               f"typical error ≈ £{bundle['test_metrics']['MAE']:,.0f}. The estimate ignores trim, condition and options.")
