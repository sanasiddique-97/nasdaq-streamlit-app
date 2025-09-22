# # stock_vs_nasdaq_app.py
# import finnhub
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import plotly.graph_objects as go
# from datetime import datetime, timedelta
#   # for fetching news


# st.set_page_config(layout="wide", page_title="Stock vs NASDAQ (^IXIC) — Weekly Comparison")

# # ---- Sidebar controls ----
# st.sidebar.header("Controls")
# tickers_input = st.sidebar.text_input("Enter tickers (comma separated)", value="AAPL,MSFT,TSLA")
# min_mcap = st.sidebar.number_input("Minimum market cap (USD, billions)", value=5.0, min_value=0.0, step=0.5)
# weeks = st.sidebar.slider("Number of weeks to compare (weekly granularity)", min_value=1, max_value=52, value=12)
# show_news = st.sidebar.checkbox("Fetch latest news for overperformers", value=True)
# compare_range = st.sidebar.selectbox("Quick range", options=["Past week", "Past 3 months", "Custom weeks"], index=2)
# if compare_range == "Past week":
#     weeks = 1
# elif compare_range == "Past 3 months":
#     weeks = 12

# st.sidebar.markdown("---")
# st.sidebar.markdown("Defaults: market cap = 5B, out/underperform threshold = ±5%")
# threshold_pct = st.sidebar.number_input("Out/Underperform threshold (%)", value=5.0, min_value=0.1, step=0.1)

# # ---- Main ----
# st.title("Compare selected stocks vs NASDAQ Composite (^IXIC) — Weekly basis")
# st.markdown("Weekly percent changes; difference = stock_weekly% - index_weekly%.")

# # parse tickers
# raw = tickers_input.strip().upper()
# if not raw:
#     st.warning("Enter at least one ticker (e.g., AAPL).")
#     st.stop()
# tickers = [t.strip() for t in raw.split(",") if t.strip()]
# index_ticker = "^IXIC"

# # compute date range
# end_date = datetime.utcnow().date()
# start_date = end_date - timedelta(weeks=weeks + 4)  # fetch a few extra weeks to ensure alignment



# @st.cache_data(show_spinner=False)
# def fetch_tickers_info(tlist):
#     info = {}
#     for t in tlist:
#         tk = yf.Ticker(t)
#         try:
#             data = tk.history(period="5y", auto_adjust=False)
#         except Exception:
#             data = pd.DataFrame()
#         try:
#             mc = tk.info.get("marketCap", None)
#         except Exception:
#             mc = None
#         # ⚠️ store only serializable parts (drop the raw yfinance object)
#         info[t] = {
#             "history": data,
#             "marketCap": mc
#         }
#     return info


# # fetch data (stocks + index)
# all_symbols = tickers + [index_ticker]
# with st.spinner("Fetching data..."):
#     info = fetch_tickers_info(all_symbols)

# # filter out tickers lacking data or below market cap
# filtered = []
# filtered_reasons = {}
# for t in tickers:
#     mcap = info[t]["marketCap"]
#     if mcap is None:
#         # attempt to infer via latest market cap from last close * sharesOutstanding if available
#         filtered_reasons[t] = "market cap unknown"
#         continue
#     if mcap < min_mcap * 1e9:
#         filtered_reasons[t] = f"market cap {mcap:.0f} < {min_mcap*1e9:.0f}"
#         continue
#     if info[t]["history"].empty:
#         filtered_reasons[t] = "no price history"
#         continue
#     filtered.append(t)

# if len(filtered) == 0:
#     st.error("No tickers passed the market-cap filter or had usable price data. See reasons below.")
#     st.write(pd.DataFrame.from_dict(filtered_reasons, orient="index", columns=["reason"]))
#     st.stop()

# # Prepare weekly resampled returns
# def weekly_close_pct(df):
#     # df expected to have DatetimeIndex and columns including 'Close'
#     if df.empty:
#         return pd.Series(dtype=float)
#     w = df["Close"].resample("W-FRI").last().dropna()
#     pct = w.pct_change().dropna() * 100.0
#     pct.name = "weekly_pct"
#     return pct

# index_weekly = weekly_close_pct(info[index_ticker]["history"])
# if index_weekly.empty:
#     st.error(f"Could not fetch weekly data for index {index_ticker}.")
#     st.stop()

# # Align index weeks and keep most recent `weeks`
# index_weekly = index_weekly.sort_index()
# index_weekly = index_weekly.tail(weeks)
# index_weekly.index = index_weekly.index.date  # use simple dates for display

# # DataFrame to store comparisons
# comp_rows = []

# for t in filtered:
#     s_weekly = weekly_close_pct(info[t]["history"])
#     s_weekly = s_weekly.sort_index()
#     # align on same weekly index (dates)
#     s_weekly.index = s_weekly.index.date
#     aligned = pd.DataFrame({
#         "stock_weekly_pct": s_weekly,
#         "index_weekly_pct": index_weekly
#     }).dropna()
#     # keep last `weeks` rows
#     aligned = aligned.tail(weeks)
#     if aligned.empty:
#         continue
#     aligned["diff_pct"] = aligned["stock_weekly_pct"] - aligned["index_weekly_pct"]
#     # summarize
#     avg_diff = aligned["diff_pct"].mean()
#     latest_diff = aligned["diff_pct"].iloc[-1]
#     # mark if latest over/under out of threshold
#     status = "neutral"
#     if latest_diff > threshold_pct:
#         status = "outperform"
#     elif latest_diff < -threshold_pct:
#         status = "underperform"
#     comp_rows.append({
#         "ticker": t,
#         "marketCap": info[t]["marketCap"],
#         "weeks_available": len(aligned),
#         "avg_diff_pct": avg_diff,
#         "latest_diff_pct": latest_diff,
#         "status": status,
#         "aligned": aligned
#     })

# if len(comp_rows) == 0:
#     st.write("No aligned weekly data for selected tickers and index in the requested range.")
#     st.stop()

# # # Build comparison table (weekly summary)
# # summary_df = pd.DataFrame([{
# #     "Ticker": r["ticker"],
# #     "MarketCap (USD)": r["marketCap"],
# #     "Weeks": r["weeks_available"],
# #     "Avg diff (%)": round(r["avg_diff_pct"], 3),
# #     "Latest diff (%)": round(r["latest_diff_pct"], 3),
# #     "Status": r["status"]
# # } for r in comp_rows]).set_index("Ticker")

# # st.subheader("Weekly comparison summary")
# # st.dataframe(summary_df)

# # Build comparison table (weekly summary)
# summary_df = pd.DataFrame([{
#     "Ticker": r["ticker"],
#     "MarketCap (USD)": r["marketCap"],
#     "Weeks": r["weeks_available"],
#     "Avg diff (%)": round(r["avg_diff_pct"], 3),
#     "Latest diff (%)": round(r["latest_diff_pct"], 3),
#     "Status": r["status"]
# } for r in comp_rows]).set_index("Ticker")

# # Styling function
# def highlight_status(row):
#     if row["Status"] == "outperform":
#         return ["background-color: #d4edda"]*len(row)  # light green
#     elif row["Status"] == "underperform":
#         return ["background-color: #f8d7da"]*len(row)  # light red
#     else:
#         return [""]*len(row)  # neutral

# st.subheader("Weekly comparison summary")
# st.dataframe(summary_df.style.apply(highlight_status, axis=1))


# # Interactive Plotly weekly chart
# st.subheader("Interactive weekly % difference chart")
# fig = go.Figure()
# date_axis = None

# for r in comp_rows:
#     name = r["ticker"]
#     df = r["aligned"].copy()
#     df = df.reset_index().rename(columns={"index":"date"})
#     date_axis = df["date"] if date_axis is None else date_axis
#     # line trace for diff
#     fig.add_trace(go.Scatter(
#         x=df["date"], y=df["diff_pct"],
#         mode="lines+markers",
#         name=name,
#         hovertemplate="<b>%{text}</b><br>Date: %{x}<br>% diff: %{y:.2f}%%<extra></extra>",
#         text=[name]*len(df),
#         visible=True
#     ))
#     # markers for > threshold and < -threshold
#     out_mask = df["diff_pct"] > threshold_pct
#     under_mask = df["diff_pct"] < -threshold_pct
#     # outperform markers (green)
#     fig.add_trace(go.Scatter(
#         x=df.loc[out_mask, "date"], y=df.loc[out_mask, "diff_pct"],
#         mode="markers",
#         marker=dict(size=10, symbol="triangle-up"),
#         name=f"{name} > +{threshold_pct}%",
#         hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Outperform %{y:.2f}%%<extra></extra>",
#         text=[name]*out_mask.sum()
#     ))
#     # underperform markers (red)
#     fig.add_trace(go.Scatter(
#         x=df.loc[under_mask, "date"], y=df.loc[under_mask, "diff_pct"],
#         mode="markers",
#         marker=dict(size=10, symbol="triangle-down"),
#         name=f"{name} < -{threshold_pct}%",
#         hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Underperform %{y:.2f}%%<extra></extra>",
#         text=[name]*under_mask.sum()
#     ))

# # layout
# fig.update_layout(
#     xaxis_title="Date (Weekly, Fri close)",
#     yaxis_title="Stock % - NASDAQ % (weekly, % points)",
#     legend_title="Series (click to toggle)",
#     hovermode="closest",
#     height=600
# )
# st.plotly_chart(fig, use_container_width=True)

# # Provide selectable toggles for highlighting (client options)
# st.markdown("**Client graph options**: use the Plotly legend to toggle series on/off. Click a legend item to isolate or hide a series. Markers show +/- threshold breaches.")

# # Weekly table per ticker (expandable)
# st.subheader("Per-ticker weekly details")
# for r in comp_rows:
#     st.markdown(f"**{r['ticker']}** — status: **{r['status']}**, market cap: {r['marketCap']:,}")
#     df = r["aligned"].copy().reset_index().rename(columns={"index":"Week_End_Date"})
#     df["stock_weekly_pct"] = df["stock_weekly_pct"].round(3)
#     df["index_weekly_pct"] = df["index_weekly_pct"].round(3)
#     df["diff_pct"] = df["diff_pct"].round(3)
#     st.dataframe(df, height=250)

# # Fetch and show news for current overperformers (latest diff > threshold)
# # Fetch and show news for current outperformers (latest diff > threshold)
# # if show_news:
# #     st.subheader("Latest news for current outperformers (> +{:.1f}%)".format(threshold_pct))
# #     outperf = [r for r in comp_rows if r["latest_diff_pct"] > threshold_pct]
# #     if not outperf:
# #         st.write("No current outperformers above threshold.")
# #     else:
# #         for r in outperf:
# #             t = r["ticker"]
# #             st.markdown(f"**{t}**")

# #             # create a fresh Ticker object here (not cached)
# #             tk = yf.Ticker(t)
# #             try:
# #                 articles = tk.get_news() if hasattr(tk, "get_news") else getattr(tk, "news", [])
# #             except Exception:
# #                 articles = []

# #             if not articles:
# #                 st.write("No news found for this ticker.")
# #             else:
# #                 for a in articles[:5]:
# #                     title = a.get("title") or str(a)[:80]
# #                     link = a.get("link") or a.get("url")
# #                     if link:
# #                         st.markdown(f"- [{title}]({link})")
# #                     else:
# #                         st.markdown(f"- {title}")

# # Fetch and show news for current overperformers (latest diff > threshold)
# if show_news:
#     st.subheader(f"Latest news for current outperformers (> +{threshold_pct}%)")
#     outperf = [r for r in comp_rows if r["latest_diff_pct"] > threshold_pct]
    
#     if not outperf:
#         st.write("No current outperformers above threshold.")
#     else:
#         import finnhub
#         # ⚡ Setup Finnhub client
#         API_KEY = "YOUR_FINNHUB_API_KEY"  # replace with your key
#         finnhub_client = finnhub.Client(api_key=API_KEY)
        
#         days_back = 30  # fetch news from last 30 days
        
#         for r in outperf:
#             t = r["ticker"]
#             st.markdown(f"**{t}**")
            
#             _from = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
#             _to = datetime.now().strftime('%Y-%m-%d')
            
#             try:
#                 news = finnhub_client.company_news(symbol=t, _from=_from, to=_to)
#             except Exception as e:
#                 st.write(f"Error fetching news for {t}: {e}")
#                 news = []
            
#             if not news:
#                 st.write("No news found for this ticker.")
#             else:
#                 # display top 5 news with clickable links and summary
#                 for a in news[:5]:
#                     headline = a.get("headline", "No title")
#                     url = a.get("url")
#                     summary = a.get("summary", "No summary available")
#                     date = pd.to_datetime(a.get("datetime", None), unit='s').strftime('%Y-%m-%d') if a.get("datetime") else ""
#                     if url:
#                         st.markdown(f"- [{headline}]({url}) ({date})")
#                     else:
#                         st.markdown(f"- {headline} ({date})")
#                     st.markdown(f"  > {summary}")



# st.markdown("---")
# st.caption("Data source: Yahoo Finance via yfinance. Index: Nasdaq Composite (^IXIC).")

# stock_vs_nasdaq_app_updated.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Stock vs NASDAQ (^IXIC) — Weekly Comparison")

# ---- Sidebar controls ----
st.sidebar.header("Controls")
tickers_input = st.sidebar.text_input("Enter tickers (comma separated)", value="AAPL,MSFT,TSLA")
# Market cap range selector (user asked to start from 5 billion and be able to increase)
market_cap_range = st.sidebar.slider("Market cap range (USD, billions)", min_value=5.0, max_value=2000.0, value=(5.0, 200.0), step=1.0)
min_mcap, max_mcap = market_cap_range
weeks = st.sidebar.slider("Number of weeks to compare (weekly granularity)", min_value=1, max_value=156, value=12)
show_news = st.sidebar.checkbox("Fetch latest news for overperformers", value=False)
compare_range = st.sidebar.selectbox("Quick range", options=["Past week", "Past 3 months", "Custom weeks"], index=2)
if compare_range == "Past week":
    weeks = 1
elif compare_range == "Past 3 months":
    weeks = 12

st.sidebar.markdown("---")
st.sidebar.markdown("Defaults: market cap lower bound = 5B, out/underperform threshold = ±5%")
threshold_pct = st.sidebar.number_input("Out/Underperform threshold (%)", value=5.0, min_value=0.1, step=0.1)

# ---- Main ----
st.title("Compare selected stocks vs NASDAQ Composite (^IXIC) — Weekly basis")
st.markdown("Weekly percent changes; difference = stock_weekly% - index_weekly%.")

# parse tickers
raw = tickers_input.strip().upper()
if not raw:
    st.warning("Enter at least one ticker (e.g., AAPL).")
    st.stop()
tickers = [t.strip() for t in raw.split(",") if t.strip()]
index_ticker = "^IXIC"

# compute date range
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(weeks=weeks + 8)  # fetch extras to ensure alignment


@st.cache_data(show_spinner=False)
def fetch_tickers_info(tlist):
    info = {}
    for t in tlist:
        tk = yf.Ticker(t)
        try:
            # fetch 5y to be safe; we'll resample down to weekly
            data = tk.history(period="5y", auto_adjust=False)
        except Exception:
            data = pd.DataFrame()
        try:
            mc = None
            # yfinance's info may sometimes be slow/fail; guard it
            if hasattr(tk, "info") and isinstance(tk.info, dict):
                mc = tk.info.get("marketCap", None)
        except Exception:
            mc = None
        info[t] = {
            "history": data,
            "marketCap": mc
        }
    return info


# fetch data (stocks + index)
all_symbols = tickers + [index_ticker]
with st.spinner("Fetching data..."):
    info = fetch_tickers_info(all_symbols)

# filter out tickers lacking data or outside market cap range
filtered = []
filtered_reasons = {}
for t in tickers:
    mcap = info.get(t, {}).get("marketCap")
    hist = info.get(t, {}).get("history", pd.DataFrame())
    if hist is None or hist.empty:
        filtered_reasons[t] = "no price history"
        continue
    if mcap is None:
        filtered_reasons[t] = "market cap unknown"
        continue
    if (mcap < min_mcap * 1e9) or (mcap > max_mcap * 1e9):
        filtered_reasons[t] = f"market cap {mcap:.0f} outside [{min_mcap*1e9:.0f}, {max_mcap*1e9:.0f}]"
        continue
    filtered.append(t)

if len(filtered) == 0:
    st.error("No tickers passed the market-cap filter or had usable price data. See reasons below.")
    st.write(pd.DataFrame.from_dict(filtered_reasons, orient="index", columns=["reason"]))
    st.stop()


# Prepare weekly resampled closes and percent returns
def weekly_close_and_pct(df):
    """Return DataFrame with weekly close (Fri) and percent change relative to previous week.
    Returns empty DataFrame if input is empty."""
    if df is None or df.empty:
        return pd.DataFrame()
    w_close = df['Close'].resample('W-FRI').last().dropna()
    if len(w_close) < 2:
        return pd.DataFrame()
    pct = w_close.pct_change().dropna() * 100.0
    result = pd.DataFrame({'close': w_close, 'weekly_pct': pct})
    return result


# Index weekly
index_weekly_df = weekly_close_and_pct(info[index_ticker]['history'])
if index_weekly_df.empty:
    st.error(f"Could not fetch weekly data for index {index_ticker}.")
    st.stop()

# keep last `weeks` rows of index
index_weekly_df = index_weekly_df.sort_index().tail(weeks)
index_weekly_df.index = index_weekly_df.index.date

# DataFrame to store comparisons
comp_rows = []

for t in filtered:
    s_weekly_df = weekly_close_and_pct(info[t]['history'])
    if s_weekly_df.empty:
        continue
    s_weekly_df = s_weekly_df.sort_index()
    s_weekly_df.index = s_weekly_df.index.date
    # align on same weekly index
    aligned = pd.DataFrame({
        'stock_close': s_weekly_df['close'],
        'stock_weekly_pct': s_weekly_df['weekly_pct'],
        'index_close': index_weekly_df['close'],
        'index_weekly_pct': index_weekly_df['weekly_pct']
    }).dropna()
    aligned = aligned.tail(weeks)
    if aligned.empty:
        continue
    aligned['diff_pct'] = aligned['stock_weekly_pct'] - aligned['index_weekly_pct']
    # summarize
    avg_diff = aligned['diff_pct'].mean()
    latest_diff = aligned['diff_pct'].iloc[-1]
    # mark if latest over/under out of threshold
    status = 'neutral'
    if latest_diff > threshold_pct:
        status = 'outperform'
    elif latest_diff < -threshold_pct:
        status = 'underperform'

    # Cumulative calculations (correct methods):
    # 1) Compounded method (product of (1 + weekly_return))
    stock_factors = (1 + aligned['stock_weekly_pct'] / 100.0)
    index_factors = (1 + aligned['index_weekly_pct'] / 100.0)
    stock_cum_compounded = stock_factors.prod() - 1.0
    index_cum_compounded = index_factors.prod() - 1.0
    # as percent
    stock_cum_compounded_pct = stock_cum_compounded * 100.0
    index_cum_compounded_pct = index_cum_compounded * 100.0

    # 2) Start-End method using weekly closes
    stock_start = aligned['stock_close'].iloc[0]
    stock_end = aligned['stock_close'].iloc[-1]
    index_start = aligned['index_close'].iloc[0]
    index_end = aligned['index_close'].iloc[-1]
    stock_cum_startend_pct = (stock_end - stock_start) / stock_start * 100.0
    index_cum_startend_pct = (index_end - index_start) / index_start * 100.0

    # store
    comp_rows.append({
        'ticker': t,
        'marketCap': info[t]['marketCap'],
        'weeks_available': len(aligned),
        'avg_diff_pct': avg_diff,
        'latest_diff_pct': latest_diff,
        'status': status,
        'aligned': aligned,
        'stock_cum_compounded_pct': stock_cum_compounded_pct,
        'index_cum_compounded_pct': index_cum_compounded_pct,
        'stock_cum_startend_pct': stock_cum_startend_pct,
        'index_cum_startend_pct': index_cum_startend_pct
    })

if len(comp_rows) == 0:
    st.write("No aligned weekly data for selected tickers and index in the requested range.")
    st.stop()

# Build comparison table (weekly summary)
summary_df = pd.DataFrame([{ 
    'Ticker': r['ticker'],
    'MarketCap (USD)': r['marketCap'],
    'Weeks': r['weeks_available'],
    'Avg diff (%)': round(r['avg_diff_pct'], 3),
    'Latest diff (%)': round(r['latest_diff_pct'], 3),
    'Status': r['status']
} for r in comp_rows]).set_index('Ticker')

# Styling function
def highlight_status(row):
    if row['Status'] == 'outperform':
        return ['background-color: #d4edda']*len(row)  # light green
    elif row['Status'] == 'underperform':
        return ['background-color: #f8d7da']*len(row)  # light red
    else:
        return ['']*len(row)  # neutral

st.subheader("Weekly comparison summary")
st.dataframe(summary_df.style.apply(highlight_status, axis=1))

# Interactive Plotly weekly chart (difference)
st.subheader("Interactive weekly % difference chart")
fig = go.Figure()
date_axis = None
for r in comp_rows:
    name = r['ticker']
    df = r['aligned'].copy().reset_index().rename(columns={'index':'date'})
    date_axis = df['date'] if date_axis is None else date_axis
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['diff_pct'],
        mode='lines+markers',
        name=name,
        hovertemplate="<b>%{text}</b><br>Date: %{x}<br>% diff: %{y:.2f}%%<extra></extra>",
        text=[name]*len(df),
        visible=True
    ))
    out_mask = df['diff_pct'] > threshold_pct
    under_mask = df['diff_pct'] < -threshold_pct
    fig.add_trace(go.Scatter(
        x=df.loc[out_mask, 'date'], y=df.loc[out_mask, 'diff_pct'],
        mode='markers',
        marker=dict(size=10, symbol='triangle-up'),
        name=f"{name} > +{threshold_pct}%",
        hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Outperform %{y:.2f}%%<extra></extra>",
        text=[name]*out_mask.sum()
    ))
    fig.add_trace(go.Scatter(
        x=df.loc[under_mask, 'date'], y=df.loc[under_mask, 'diff_pct'],
        mode='markers',
        marker=dict(size=10, symbol='triangle-down'),
        name=f"{name} < -{threshold_pct}%",
        hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Underperform %{y:.2f}%%<extra></extra>",
        text=[name]*under_mask.sum()
    ))

fig.update_layout(
    xaxis_title='Date (Weekly, Fri close)',
    yaxis_title='Stock % - NASDAQ % (weekly, % points)',
    legend_title='Series (click to toggle)',
    hovermode='closest',
    height=600
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Client graph options**: use the Plotly legend to toggle series on/off. Click a legend item to isolate or hide a series. Markers show +/- threshold breaches.")

# Weekly table per ticker (expandable)
st.subheader("Per-ticker weekly details")
for r in comp_rows:
    st.markdown(f"**{r['ticker']}** — status: **{r['status']}**, market cap: {r['marketCap']:,}")
    df = r['aligned'].copy().reset_index().rename(columns={'index':'Week_End_Date'})
    df['stock_weekly_pct'] = df['stock_weekly_pct'].round(3)
    df['index_weekly_pct'] = df['index_weekly_pct'].round(3)
    df['diff_pct'] = df['diff_pct'].round(3)
    df['stock_close'] = df['stock_close'].round(3)
    df['index_close'] = df['index_close'].round(3)
    st.dataframe(df, height=250)

# ------------------ New: Cumulative calculations and displays ------------------
st.markdown("---")
st.subheader("Cumulative percentage change (correct methods)")
st.markdown("This section shows cumulative returns calculated using two correct methods:\n1) Compounding weekly returns (product of 1 + weekly_return)\n2) Start-to-end percent change using weekly closes")


# Build cumulative summary table
cum_rows = []
for r in comp_rows:
    cum_rows.append({
        'Ticker': r['ticker'],
        'MarketCap (USD)': r['marketCap'],
        'Weeks': r['weeks_available'],
        'Stock cum (compounded %)': round(r['stock_cum_compounded_pct'], 3),
        'Index cum (compounded %)': round(r['index_cum_compounded_pct'], 3),
        'Stock cum (start-end %)': round(r['stock_cum_startend_pct'], 3),
        'Index cum (start-end %)': round(r['index_cum_startend_pct'], 3),
        'Cum diff (compounded)': round(r['stock_cum_compounded_pct'] - r['index_cum_compounded_pct'], 3),
    })

cum_df = pd.DataFrame(cum_rows).set_index('Ticker')

# highlight outperform/underperform by cumulative compounded difference
def highlight_cum(row):
    diff = row['Cum diff (compounded)']
    if diff > threshold_pct:
        return ['background-color: #d4edda']*len(row)
    if diff < -threshold_pct:
        return ['background-color: #f8d7da']*len(row)
    return ['']*len(row)

st.dataframe(cum_df.style.apply(highlight_cum, axis=1))

# Separate graph for cumulative compounded returns (stock vs index) per ticker
st.subheader('Cumulative compounded returns — separate charts')
for r in comp_rows:
    st.markdown(f"**{r['ticker']}** — Compounded cumulative over last {r['weeks_available']} weeks")
    df = r['aligned'].copy()
    # cumulative series by multiplying factors cumulatively
    df['stock_cum_factor'] = (1 + df['stock_weekly_pct']/100.0).cumprod()
    df['index_cum_factor'] = (1 + df['index_weekly_pct']/100.0).cumprod()
    # convert to pct relative to 1
    df['stock_cum_pct'] = (df['stock_cum_factor'] - 1.0) * 100.0
    df['index_cum_pct'] = (df['index_cum_factor'] - 1.0) * 100.0
    # plot
    figc = go.Figure()
    dates = df.reset_index().rename(columns={'index':'date'})['date']
    figc.add_trace(go.Scatter(x=dates, y=df['stock_cum_pct'], mode='lines+markers', name=f"{r['ticker']} cumulative %"))
    figc.add_trace(go.Scatter(x=dates, y=df['index_cum_pct'], mode='lines+markers', name='NASDAQ cumulative %'))
    figc.update_layout(xaxis_title='Date (Weekly, Fri close)', yaxis_title='Cumulative % (compounded)', height=450)
    st.plotly_chart(figc, use_container_width=True)

# Combined cumulative chart — compare all tickers vs index
st.subheader('Combined cumulative compounded chart')
fig_comb = go.Figure()
for r in comp_rows:
    df = r['aligned'].copy()
    df['stock_cum_pct'] = (1 + df['stock_weekly_pct']/100.0).cumprod() - 1.0
    df['stock_cum_pct'] = df['stock_cum_pct'] * 100.0
    df = df.reset_index().rename(columns={'index':'date'})
    fig_comb.add_trace(go.Scatter(x=df['date'], y=df['stock_cum_pct'], mode='lines', name=r['ticker']))
# index baseline — use index from first comp_rows aligned (they all share same index set by construction)
# pick index series from the first
first_index_df = comp_rows[0]['aligned'].copy()
first_index_df['index_cum_pct'] = (1 + first_index_df['index_weekly_pct']/100.0).cumprod() - 1.0
first_index_df['index_cum_pct'] = first_index_df['index_cum_pct'] * 100.0
first_index_df = first_index_df.reset_index().rename(columns={'index':'date'})
fig_comb.add_trace(go.Scatter(x=first_index_df['date'], y=first_index_df['index_cum_pct'], mode='lines', name='NASDAQ', line=dict(dash='dash')))
fig_comb.update_layout(xaxis_title='Date (Weekly, Fri close)', yaxis_title='Cumulative % (compounded)', height=600)
st.plotly_chart(fig_comb, use_container_width=True)

# Table that highlights which tick outperformed and underperformed with nasdaq (using cumulative compounded)
st.subheader('Out/Underperform vs NASDAQ (Cumulative, compounded)')
out_rows = []
for r in comp_rows:
    diff = r['stock_cum_compounded_pct'] - r['index_cum_compounded_pct']
    status = 'neutral'
    if diff > threshold_pct:
        status = 'outperform'
    elif diff < -threshold_pct:
        status = 'underperform'
    out_rows.append({
        'Ticker': r['ticker'],
        'MarketCap (USD)': r['marketCap'],
        'Stock cum (compounded %)': round(r['stock_cum_compounded_pct'], 3),
        'Index cum (compounded %)': round(r['index_cum_compounded_pct'], 3),
        'Cum diff (pct points)': round(diff, 3),
        'Status': status
    })

out_df = pd.DataFrame(out_rows).set_index('Ticker')

# style
st.dataframe(out_df.style.apply(lambda row: ['background-color: #d4edda']*len(row) if row['Status']=='outperform' else (['background-color: #f8d7da']*len(row) if row['Status']=='underperform' else ['']*len(row)), axis=1))

# Optional: fetch news for outperformers
if show_news:
    st.markdown('---')
    st.subheader(f'Latest news for current outperformers (> +{threshold_pct}%)')
    outperf = [r for r in comp_rows if (r['stock_cum_compounded_pct'] - r['index_cum_compounded_pct']) > threshold_pct]
    if not outperf:
        st.write('No current cumulative outperformers above threshold.')
    else:
        import finnhub
        API_KEY = "YOUR_FINNHUB_API_KEY"  # replace with your key
        finnhub_client = finnhub.Client(api_key=API_KEY)
        days_back = 30
        for r in outperf:
            t = r['ticker']
            st.markdown(f"**{t}**")
            _from = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            _to = datetime.now().strftime('%Y-%m-%d')
            try:
                news = finnhub_client.company_news(symbol=t, _from=_from, to=_to)
            except Exception as e:
                st.write(f"Error fetching news for {t}: {e}")
                news = []
            if not news:
                st.write('No news found for this ticker.')
            else:
                for a in news[:5]:
                    headline = a.get('headline', 'No title')
                    url = a.get('url')
                    summary = a.get('summary', 'No summary available')
                    date = pd.to_datetime(a.get('datetime', None), unit='s').strftime('%Y-%m-%d') if a.get('datetime') else ''
                    if url:
                        st.markdown(f"- [{headline}]({url}) ({date})")
                    else:
                        st.markdown(f"- {headline} ({date})")
                    st.markdown(f"  > {summary}")

st.markdown("---")
st.caption("Data source: Yahoo Finance via yfinance. Index: Nasdaq Composite (^IXIC).")
