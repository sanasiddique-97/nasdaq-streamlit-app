# stock_vs_nasdaq_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import finnhub

st.set_page_config(layout="wide", page_title="Stock vs NASDAQ (^IXIC) — Weekly Comparison")

# ---- Sidebar controls ----
st.sidebar.header("Controls")

# Ticker selection dropdown (default options)
available_tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOG", "META", "NFLX"]
tickers = st.sidebar.multiselect("Select tickers", available_tickers, default=["AAPL", "MSFT", "TSLA"])

# Market cap range
st.sidebar.subheader("Market Cap Filter (USD Billions)")
min_mcap = st.sidebar.number_input("Min market cap", value=5.0, min_value=0.0, step=0.5)
max_mcap = st.sidebar.number_input("Max market cap (0 = no limit)", value=0.0, min_value=0.0, step=0.5)

# Weeks and thresholds
weeks = st.sidebar.slider("Number of weeks to compare (weekly granularity)", min_value=1, max_value=52, value=12)
show_news = st.sidebar.checkbox("Fetch latest news for overperformers", value=True)
compare_range = st.sidebar.selectbox("Quick range", options=["Past week", "Past 3 months", "Custom weeks"], index=2)
if compare_range == "Past week":
    weeks = 1
elif compare_range == "Past 3 months":
    weeks = 12

st.sidebar.markdown("---")
st.sidebar.markdown("Defaults: market cap = 5B, out/underperform threshold = ±5%")
threshold_pct = st.sidebar.number_input("Out/Underperform threshold (%)", value=5.0, min_value=0.1, step=0.1)

# ---- Main ----
st.title("Compare selected stocks vs NASDAQ Composite (^IXIC) — Weekly basis")
st.markdown("Weekly percent changes; difference = stock_weekly% - index_weekly%.")

# ---- Data fetching ----
index_ticker = "^IXIC"
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(weeks=weeks + 4)  # fetch extra weeks for alignment

@st.cache_data(show_spinner=False)
def fetch_tickers_info(tlist):
    info = {}
    for t in tlist:
        tk = yf.Ticker(t)
        try:
            data = tk.history(period="5y", auto_adjust=False)
        except Exception:
            data = pd.DataFrame()
        try:
            mc = tk.info.get("marketCap", None)
        except Exception:
            mc = None
        info[t] = {
            "history": data,
            "marketCap": mc
        }
    return info

# Fetch data
all_symbols = tickers + [index_ticker]
with st.spinner("Fetching data..."):
    info = fetch_tickers_info(all_symbols)

# Filter tickers by market cap and data availability
filtered = []
filtered_reasons = {}
for t in tickers:
    mcap = info[t]["marketCap"]
    if mcap is None:
        filtered_reasons[t] = "market cap unknown"
        continue
    if mcap < min_mcap * 1e9:
        filtered_reasons[t] = f"market cap {mcap:.0f} < {min_mcap*1e9:.0f}"
        continue
    if max_mcap > 0 and mcap > max_mcap * 1e9:
        filtered_reasons[t] = f"market cap {mcap:.0f} > {max_mcap*1e9:.0f}"
        continue
    if info[t]["history"].empty:
        filtered_reasons[t] = "no price history"
        continue
    filtered.append(t)

if len(filtered) == 0:
    st.error("No tickers passed the market-cap filter or had usable price data. See reasons below.")
    st.write(pd.DataFrame.from_dict(filtered_reasons, orient="index", columns=["reason"]))
    st.stop()

# ---- Weekly % changes ----
def weekly_close_pct(df):
    if df.empty:
        return pd.Series(dtype=float)
    w = df["Close"].resample("W-FRI").last().dropna()
    pct = w.pct_change().dropna() * 100.0
    pct.name = "weekly_pct"
    return pct

def cumulative_return(pct_series):
    factor = (1 + pct_series/100).cumprod()
    return (factor - 1) * 100

# Index weekly
index_weekly = weekly_close_pct(info[index_ticker]["history"])
if index_weekly.empty:
    st.error(f"Could not fetch weekly data for index {index_ticker}.")
    st.stop()
index_weekly = index_weekly.sort_index().tail(weeks)
index_weekly.index = index_weekly.index.date

# Compare tickers
comp_rows = []
for t in filtered:
    s_weekly = weekly_close_pct(info[t]["history"]).sort_index()
    s_weekly.index = s_weekly.index.date
    aligned = pd.DataFrame({
        "stock_weekly_pct": s_weekly,
        "index_weekly_pct": index_weekly
    }).dropna().tail(weeks)
    if aligned.empty:
        continue
    aligned["diff_pct"] = aligned["stock_weekly_pct"] - aligned["index_weekly_pct"]
    aligned["stock_cum_pct"] = cumulative_return(aligned["stock_weekly_pct"])
    aligned["index_cum_pct"] = cumulative_return(aligned["index_weekly_pct"])
    aligned["diff_cum_pct"] = aligned["stock_cum_pct"] - aligned["index_cum_pct"]
    
    avg_diff = aligned["diff_pct"].mean()
    latest_diff = aligned["diff_pct"].iloc[-1]
    status = "neutral"
    if latest_diff > threshold_pct:
        status = "outperform"
    elif latest_diff < -threshold_pct:
        status = "underperform"
    comp_rows.append({
        "ticker": t,
        "marketCap": info[t]["marketCap"],
        "weeks_available": len(aligned),
        "avg_diff_pct": avg_diff,
        "latest_diff_pct": latest_diff,
        "status": status,
        "aligned": aligned
    })

if len(comp_rows) == 0:
    st.write("No aligned weekly data for selected tickers and index in the requested range.")
    st.stop()

# ---- Summary Table ----
summary_df = pd.DataFrame([{
    "Ticker": r["ticker"],
    "MarketCap (USD)": r["marketCap"],
    "Weeks": r["weeks_available"],
    "Avg diff (%)": round(r["avg_diff_pct"], 3),
    "Latest diff (%)": round(r["latest_diff_pct"], 3),
    "Status": r["status"]
} for r in comp_rows]).set_index("Ticker")

def highlight_status(row):
    if row["Status"] == "outperform":
        return ["background-color: #d4edda"]*len(row)
    elif row["Status"] == "underperform":
        return ["background-color: #f8d7da"]*len(row)
    else:
        return [""]*len(row)

st.subheader("Weekly comparison summary")
st.dataframe(summary_df.style.apply(highlight_status, axis=1))

# ---- Weekly % difference chart ----
st.subheader("Interactive weekly % difference chart")
fig = go.Figure()
date_axis = None
for r in comp_rows:
    df = r["aligned"].reset_index().rename(columns={"index":"date"})
    date_axis = df["date"] if date_axis is None else date_axis
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["diff_pct"],
        mode="lines+markers",
        name=r["ticker"],
        hovertemplate="<b>%{text}</b><br>Date: %{x}<br>% diff: %{y:.2f}%%<extra></extra>",
        text=[r["ticker"]]*len(df)
    ))
    out_mask = df["diff_pct"] > threshold_pct
    under_mask = df["diff_pct"] < -threshold_pct
    fig.add_trace(go.Scatter(
        x=df.loc[out_mask, "date"], y=df.loc[out_mask, "diff_pct"],
        mode="markers", marker=dict(size=10, symbol="triangle-up"),
        name=f"{r['ticker']} > +{threshold_pct}%", text=[r["ticker"]]*out_mask.sum(),
        hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Outperform %{y:.2f}%%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df.loc[under_mask, "date"], y=df.loc[under_mask, "diff_pct"],
        mode="markers", marker=dict(size=10, symbol="triangle-down"),
        name=f"{r['ticker']} < -{threshold_pct}%", text=[r["ticker"]]*under_mask.sum(),
        hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Underperform %{y:.2f}%%<extra></extra>"
    ))

fig.update_layout(
    xaxis_title="Date (Weekly, Fri close)",
    yaxis_title="Stock % - NASDAQ % (weekly, % points)",
    legend_title="Series",
    hovermode="closest", height=600
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("**Client graph options**: use the legend to toggle series on/off. Markers show ±threshold breaches.")

# ---- Cumulative % difference chart ----
st.subheader("Cumulative % Change vs NASDAQ")
fig_cum = go.Figure()
for r in comp_rows:
    df = r["aligned"].reset_index().rename(columns={"index":"date"})
    fig_cum.add_trace(go.Scatter(
        x=df["date"], y=df["diff_cum_pct"],
        mode="lines+markers",
        name=r["ticker"],
        hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Cumulative diff: %{y:.2f}%%<extra></extra>",
        text=[r["ticker"]]*len(df)
    ))
fig_cum.update_layout(
    xaxis_title="Date (Weekly, Fri close)",
    yaxis_title="Cumulative Stock % - NASDAQ %",
    legend_title="Tickers",
    height=600
)
st.plotly_chart(fig_cum, use_container_width=True)

# ---- Per-ticker weekly details ----
st.subheader("Per-ticker weekly details")
for r in comp_rows:
    st.markdown(f"**{r['ticker']}** — status: **{r['status']}**, market cap: {r['marketCap']:,}")
    df = r["aligned"].copy().reset_index().rename(columns={"index":"Week_End_Date"})
    df = df.round(3)
    st.dataframe(df, height=250)

# ---- Out/Underperformers Table ----
st.subheader(f"Tickers vs NASDAQ ±{threshold_pct}% (latest week)")
out_table = summary_df[(summary_df["Latest diff (%)"] > threshold_pct) | 
                       (summary_df["Latest diff (%)"] < -threshold_pct)]
if out_table.empty:
    st.write("No tickers breached threshold this week.")
else:
    st.dataframe(out_table.style.apply(highlight_status, axis=1))

# ---- Latest news for current outperformers ----
if show_news:
    st.subheader(f"Latest news for current outperformers (> +{threshold_pct}%)")
    outperf = [r for r in comp_rows if r["latest_diff_pct"] > threshold_pct]
    if not outperf:
        st.write("No current outperformers above threshold.")
    else:
        API_KEY = "d38l8ahr01qthpo0bo10d38l8ahr01qthpo0bo1g"  # replace with your key
        finnhub_client = finnhub.Client(api_key=API_KEY)
        days_back = 30
        for r in outperf:
            t = r["ticker"]
            st.markdown(f"**{t}**")
            _from = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            _to = datetime.now().strftime('%Y-%m-%d')
            try:
                news = finnhub_client.company_news(symbol=t, _from=_from, to=_to)
            except Exception as e:
                st.write(f"Error fetching news for {t}: {e}")
                news = []
            if not news:
                st.write("No news found for this ticker.")
            else:
                for a in news[:5]:
                    headline = a.get("headline", "No title")
                    url = a.get("url")
                    summary = a.get("summary", "No summary available")
                    date = pd.to_datetime(a.get("datetime", None), unit='s').strftime('%Y-%m-%d') if a.get("datetime") else ""
                    if url:
                        st.markdown(f"- [{headline}]({url}) ({date})")
                    else:
                        st.markdown(f"- {headline} ({date})")
                    st.markdown(f"  > {summary}")

st.markdown("---")
st.caption("Data source: Yahoo Finance via yfinance. Index: Nasdaq Composite (^IXIC).")
