import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

st.set_page_config(page_title="Stock Market Trend Analysis", page_icon="📈", layout="wide")

st.title("📈 Stock Market Trend Analysis")
st.caption("Python-based stock analysis dashboard using historical market data.")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Symbol", "AAPL").upper().strip()
    period = st.selectbox("Historical Period", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
    interval = st.selectbox("Interval", ["1d","1wk","1mo"], index=0)
    show_ma = st.checkbox("Show Moving Averages", True)

@st.cache_data(ttl=300)
def load_data(symbol, p, i):
    df = yf.download(symbol, period=p, interval=i, auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["Daily Return %"] = df["Close"].pct_change() * 100
    df["Volatility %"] = df["Daily Return %"].rolling(20).std()
    return df

try:
    data = load_data(ticker, period, interval)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

if data.empty:
    st.error("No data found. Check the stock symbol, e.g. AAPL, MSFT, TSLA, RELIANCE.NS, TCS.NS.")
    st.stop()

close = data["Close"].astype(float)
latest = float(close.iloc[-1])
first = float(close.iloc[0])
change = latest - first
change_pct = (change / first) * 100
avg_volume = float(data["Volume"].mean()) if "Volume" in data else 0
volatility = float(data["Volatility %"].iloc[-1]) if pd.notna(data["Volatility %"].iloc[-1]) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Price", f"{latest:.2f}")
c2.metric("Period Change", f"{change:.2f}", f"{change_pct:.2f}%")
c3.metric("Avg Volume", f"{avg_volume:,.0f}")
c4.metric("20-Day Volatility", f"{volatility:.2f}%")

st.subheader(f"Price Trend — {ticker}")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index, open=data["Open"], high=data["High"],
    low=data["Low"], close=data["Close"], name="Price"
))
if show_ma:
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], mode="lines", name="SMA 20"))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], mode="lines", name="SMA 50"))
fig.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("📊 Volume")
    vf = go.Figure(go.Bar(x=data.index, y=data["Volume"], name="Volume"))
    vf.update_layout(height=350, template="plotly_white")
    st.plotly_chart(vf, use_container_width=True)

with right:
    st.subheader("📉 Daily Returns")
    rf = go.Figure(go.Scatter(x=data.index, y=data["Daily Return %"], mode="lines", name="Return %"))
    rf.update_layout(height=350, template="plotly_white", yaxis_title="%")
    st.plotly_chart(rf, use_container_width=True)

st.subheader("🔍 Trend Analysis")
sma20 = data["SMA20"].iloc[-1]
sma50 = data["SMA50"].iloc[-1]

if pd.isna(sma50):
    trend = "Insufficient data for 50-period moving average."
elif latest > sma20 > sma50:
    trend = "Uptrend: price is above SMA20 and SMA50."
elif latest < sma20 < sma50:
    trend = "Downtrend: price is below SMA20 and SMA50."
else:
    trend = "Sideways/Mixed trend: moving-average signals are mixed."

st.info(trend)

st.subheader("🤖 Simple Price Trend Projection")
forecast_days = st.slider("Projection periods", 5, 30, 10)

n = min(60, len(data))
recent = close.tail(n).values
X = np.arange(n).reshape(-1, 1)
model = LinearRegression().fit(X, recent)
future_X = np.arange(n, n + forecast_days).reshape(-1, 1)
forecast = model.predict(future_X)

proj = pd.DataFrame({
    "Period": np.arange(1, forecast_days + 1),
    "Projected Price": forecast
})
st.line_chart(proj.set_index("Period"))

st.caption("Projection is a simple linear-regression demonstration for an academic project; it is not investment advice.")

st.subheader("📋 Historical Data")
st.dataframe(data.tail(50).round(2), use_container_width=True)

csv = data.to_csv().encode("utf-8")
st.download_button("⬇️ Download Historical Data CSV", csv, f"{ticker}_historical_data.csv", "text/csv")

st.markdown("---")
st.caption("Stock Market Trend Analysis | Academic Python Project")
