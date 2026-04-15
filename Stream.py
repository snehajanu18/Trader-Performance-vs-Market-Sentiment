import streamlit as st
import pandas as pd
import joblib
import os

# ---------- LOAD ----------
@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)
    clf = joblib.load(os.path.join(base_path, "gb_classifier.pkl"))
    reg = joblib.load(os.path.join(base_path, "gb_regressor.pkl"))
    kmeans = joblib.load(os.path.join(base_path, "kmeans_model.pkl"))
    features = joblib.load(os.path.join(base_path, "features.pkl"))
    return clf, reg, kmeans, features

clf, reg, kmeans, features = load_models()

# FIX: define kmeans features manually (remove one feature)
kmeans_features = ['num_trades', 'avg_size', 'win_rate', 'long_ratio']   # assumes last feature not used in kmeans

# ---------- UI ----------
st.title("Trader Performance Predictor")

num_trades = st.number_input("Number of Trades", min_value=0.0)
avg_size = st.number_input("Average Trade Size", min_value=0.0)
win_rate = st.slider("Win Rate", 0.0, 1.0, 0.5)
long_ratio = st.slider("Long Ratio", 0.0, 1.0, 0.5)
value = st.number_input("Fear & Greed Index Value", min_value=0.0)

# ---------- INPUT ----------
input_data = pd.DataFrame([{
    "num_trades": num_trades,
    "avg_size": avg_size,
    "win_rate": win_rate,
    "long_ratio": long_ratio,
    "value": value
}])

# correct order
input_data = input_data[features]

# ---------- PREDICT ----------
if st.button("Predict"):
    profit_class = clf.predict(input_data)[0]
    pnl_pred = reg.predict(input_data)[0]

    # FIX: use only kmeans features
    cluster = kmeans.predict(input_data[kmeans_features])[0]

    st.subheader("Results")
    st.write(f"Profitable: {'Yes' if profit_class == 1 else 'No'}")
    st.write(f"Predicted PnL: {pnl_pred:.2f}")
    st.write(f"Trader Cluster: {cluster}")

# ---------- BATCH ----------
st.sidebar.header("Batch Prediction")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df = df[features]

    df["Predicted_Profit"] = clf.predict(df)
    df["Predicted_PnL"] = reg.predict(df)
    df["Cluster"] = kmeans.predict(df[kmeans_features])

    st.write(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv, "predictions.csv", "text/csv")
