# Trader Performance vs Market Sentiment

## Overview
This project predicts trader performance using behavioral metrics and market sentiment. It combines classification, regression, and clustering to evaluate profitability, estimate PnL, and segment traders.

The Streamlit app provides an interactive interface to input trader behavior and get real-time predictions.

---

## Project Structure
trader-performance-vs-market-sentiment/
│
├── Stream.py
├── Trader Performance vs Market Sentiment.ipynb
├── gb_classifier.pkl
├── gb_regressor.pkl
├── kmeans_model.pkl
├── features.pkl
├── requirements.txt
└── outputs/

## Setup Instructions

### 1. Clone Repository
git clone <https://github.com/snehajanu18/Trader-Performance-vs-Market-Sentiment>
cd trader-performance-vs-market-sentiment

2. Install Dependencies
pip install -r requirements.txt

3. Run Streamlit App
streamlit run Stream.py

Features Used
num_trades → trading activity
avg_size → trade size (risk exposure)
win_rate → consistency
long_ratio → directional bias
value → Fear & Greed Index (market sentiment)

Models
1. Classification Model
Algorithm: Gradient Boosting Classifier
Output: Predicts whether a trader is profitable (Yes/No)

3. Regression Model
Algorithm: Gradient Boosting Regressor
Output: Predicts expected PnL

5. Clustering Model
Algorithm: KMeans
Output: Segments traders into behavioral clusters

Methodology

Data aggregated at trader level
Feature engineering focused on activity, consistency, and risk
Models trained separately for:
Profit classification
PnL prediction
Behavioral clustering
Feature alignment enforced during inference to avoid mismatch errors

Insights

Win rate is the strongest predictor of profitability
High trade size increases both gains and losses (volatility effect)
Extreme long/short bias leads to unstable performance
Market sentiment influences outcomes but is secondary to trader behavior
Low win rate (<0.4) consistently results in negative PnL

Strategy Recommendations

Maintain win rate above 0.55
Avoid extreme directional bias (>0.8 long/short)
Scale trade size gradually based on consistency
Reduce exposure during extreme fear/greed conditions
Focus on quality trades instead of high frequency

Outputs

The app provides:

Profitability prediction (Yes/No)
Expected PnL value
Trader cluster label

Batch prediction via CSV upload is also supported.

Conclusion

Trader performance is primarily driven by consistency (win rate) and controlled risk exposure. Sentiment acts as a secondary signal. The system enables quick evaluation and segmentation of trading behavior using machine learning.

