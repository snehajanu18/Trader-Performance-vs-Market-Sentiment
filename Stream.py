import streamlit as st
import pandas as pd
import pickle

# Load saved artifacts
@st.cache_resource
def load_models():
    with open("gb_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("gb_regressor.pkl", "rb") as f:
        reg = pickle.load(f)
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return clf, reg, kmeans, features

clf, reg, kmeans, features = load_models()

st.title("Trader Performance Predictor")

# Input fields
num_trades = st.number_input("Number of Trades", min_value=0.0)
avg_size = st.number_input("Average Trade Size", min_value=0.0)
win_rate = st.slider("Win Rate", 0.0, 1.0, 0.5)
long_ratio = st.slider("Long Ratio", 0.0, 1.0, 0.5)
value = st.number_input("Fear & Greed Index Value", min_value=0.0)

# Create input dataframe
input_data = pd.DataFrame([{
    "num_trades": num_trades,
    "avg_size": avg_size,
    "win_rate": win_rate,
    "long_ratio": long_ratio,
    "value": value
}])

# Ensure correct feature order
input_data = input_data[features]

# Predictions
if st.button("Predict"):
    profit_class = clf.predict(input_data)[0]
    pnl_pred = reg.predict(input_data)[0]
    cluster = kmeans.predict(input_data)[0]

    st.subheader("Results")
    st.write(f"Profitable: {'Yes' if profit_class == 1 else 'No'}")
    st.write(f"Predicted PnL: {pnl_pred:.2f}")
    st.write(f"Trader Cluster: {cluster}")

# Optional: batch upload
st.sidebar.header("Batch Prediction")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df = df[features]

    df["Predicted_Profit"] = clf.predict(df)
    df["Predicted_PnL"] = reg.predict(df)
    df["Cluster"] = kmeans.predict(df)

    st.write(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv, "predictions.csv", "text/csv")
