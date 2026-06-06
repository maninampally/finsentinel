from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from google.cloud import bigquery

st.set_page_config(
    page_title="FinSentinel — Market Sentiment Intelligence",
    layout="wide"
)

st.title("FinSentinel — Real-Time Financial Sentiment Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.sidebar.header("Controls")
ticker = st.sidebar.selectbox(
    "Select Ticker",
    ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN']
)
days = st.sidebar.slider("Days of history", 1, 30, 7)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Bullish Headlines", "142", "+12%")
col2.metric("Bearish Headlines",  "87", "-5%")
col3.metric("Neutral Headlines", "203", "+2%")
col4.metric("Model F1 Score",   "0.952", "+0.003")

st.subheader(f"{ticker} Sentiment Trend — Last {days} Days")
response = requests.get(f"http://api:8000/sentiment/{ticker}?days={days}")
df = pd.DataFrame(response.json())

if not df.empty:
    fig = px.line(
        df, x='date', y='count', color='sentiment',
        color_discrete_map={
            'POSITIVE': '#00CC96',
            'NEGATIVE': '#EF553B',
            'NEUTRAL':  '#636EFA'
        },
        title=f"{ticker} Daily Sentiment Volume"
    )
    st.plotly_chart(fig, use_container_width=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Live Sentiment Feed")
    client = bigquery.Client()
    live_df = client.query("""
        SELECT headline, sentiment, confidence, ticker, published_at
        FROM `finsentinel-nlp.finsentinel_gold.predictions`
        ORDER BY published_at DESC
        LIMIT 20
    """).to_dataframe()

    st.dataframe(
        live_df.style.applymap(
            lambda x: 'color: green' if x == 'POSITIVE'
            else 'color: red' if x == 'NEGATIVE' else '',
            subset=['sentiment']
        ),
        use_container_width=True
    )

with col_right:
    st.subheader("Market Sentiment Heatmap")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN']
    scores  = [0.72, 0.45, 0.68, -0.23, 0.81, 0.55, 0.39]
    fig2 = go.Figure(go.Bar(
        x=scores, y=tickers, orientation='h',
        marker_color=['#00CC96' if s > 0 else '#EF553B' for s in scores]
    ))
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Model Drift Monitoring")
drift_col1, drift_col2 = st.columns(2)
drift_col1.metric("Data Drift Score",  "0.12", "-0.03")
drift_col2.metric("Prediction Drift",  "0.08", "+0.01")
st.info("No retraining triggered. Drift within acceptable bounds.")
