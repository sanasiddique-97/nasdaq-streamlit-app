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

# stock_vs_nasdaq_app.py
import finnhub
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---- Streamlit config ----
st.set_page_config(layout="wide", page_title="Stock vs NASDAQ (^IXIC) — Weekly Comparison")

# ---- Sidebar controls ----
st.sidebar.header("Controls")
mode = st.sidebar.radio("Choose mode", ["Select tickers", "Auto-show last week outperformers"])

weeks = st.sidebar.slider("Number of weeks to compare", min_value=1, max_value=49, value=12)
threshold_pct = st.sidebar.number_input("Out/Underperform threshold (%)", value=5.0, min_value=0.1, step=0.1)
show_news = st.sidebar.checkbox("Fetch latest news for overperformers", value=True)

# Static universe (NASDAQ-100 tickers)
nasdaq_100 = [
    "AAPL","MSFT","AMZN","GOOGL","GOOG","META","TSLA","NVDA","NFLX","PEP","COST","AVGO","CSCO","ADBE","TXN","CMCSA","AMD","INTC","QCOM","AMGN",
    "CHTR","SBUX","HON","MDLZ","INTU","AMAT","PYPL","BKNG","ADI","LRCX","GILD","MU","ISRG","CSX","PDD","REGN","VRTX","MRNA","KDP","MNST","MELI",
    "ADP","ADI","PANW","IDXX","FTNT","KHC","NXPI","MAR","ORLY","ODFL","CTAS","WDAY","XEL","CDNS","SNPS","TEAM","ROST","AZN","ABNB","PAYX","PCAR",
    "DLTR","KDP","MRVL","ALGN","CPRT","CHTR","VRSK","CRWD","CEG","ZS","DDOG","SPLK","LULU","BKR","LCID","ZBRA"
]

# ---- Data fetch ----
@st.cache_data(show_spinner=False)
def fetch_tickers_info(tlist):
    info = {}
    for t in tlist:
        tk = yf.Ticker(t)
        try:
            data = tk.history(period="2y", auto_adjust=False)
        except Exception:
            data = pd.DataFrame()
        try:
            mc = tk.info.get("marketCap", None)
        except Exception:
            mc = None
        info[t] = {"history": data, "marketCap": mc}
    return info

# ---- Helper for weekly returns ----
def weekly_close_pct(df):
    if df.empty:
        return pd.Series(dtype=float)
    w = df["Close"].resample("W-FRI").last().dropna()
    pct = w.pct_change().dropna() * 100.0
    pct.name = "weekly_pct"
    return pct

# ---- Fetch data for index ----
index_ticker = "^IXIC"
index_info = fetch_tickers_info([index_ticker])
index_weekly = weekly_close_pct(index_info[index_ticker]["history"])
if index_weekly.empty:
    st.error("Could not fetch NASDAQ data.")
    st.stop()

index_weekly = index_weekly.tail(weeks)
index_weekly.index = index_weekly.index.date

# ---- Mode logic ----
if mode == "Select tickers":
    tickers = st.sidebar.multiselect("Select tickers (market cap >5B)", nasdaq_100, default=["AAPL","MSFT","TSLA"])
    if not tickers:
        st.warning("Select at least one ticker")
        st.stop()
else:
    tickers = nasdaq_100

# ---- Fetch data ----
with st.spinner("Fetching stock data..."):
    info = fetch_tickers_info(tickers + [index_ticker])

# ---- Filter ----
filtered = []
filtered_reasons = {}
for t in tickers:
    mcap = info[t]["marketCap"]
    if mcap is None or mcap < 5e9:
        filtered_reasons[t] = "market cap <5B or unknown"
        continue
    if info[t]["history"].empty:
        filtered_reasons[t] = "no price history"
        continue
    filtered.append(t)

tickers = filtered
if not tickers:
    st.error("No tickers passed the filter")
    st.stop()

# ---- Build comparisons ----
comp_rows = []
for t in tickers:
    s_weekly = weekly_close_pct(info[t]["history"])
    s_weekly.index = s_weekly.index.date
    aligned = pd.DataFrame({
        "stock_weekly_pct": s_weekly,
        "index_weekly_pct": index_weekly
    }).dropna().tail(weeks)
    if aligned.empty:
        continue
    aligned["diff_pct"] = aligned["stock_weekly_pct"] - aligned["index_weekly_pct"]
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

if not comp_rows:
    st.write("No data to display.")
    st.stop()

# ---- Summary table ----
summary_df = pd.DataFrame([{
    "Ticker": r["ticker"],
    "MarketCap (USD)": r["marketCap"],
    "Weeks": r["weeks_available"],
    "Avg diff (%)": round(r["avg_diff_pct"], 3),
    "Latest diff (%)": round(r["latest_diff_pct"], 3),
    "Status": r["status"]
} for r in comp_rows]).set_index("Ticker")

# Highlight styling
def highlight_status(row):
    if row["Status"] == "outperform":
        return ["background-color: #d4edda"]*len(row)
    elif row["Status"] == "underperform":
        return ["background-color: #f8d7da"]*len(row)
    return [""]*len(row)

st.subheader("Weekly comparison summary")
st.dataframe(summary_df.style.apply(highlight_status, axis=1))

# ---- Interactive Plotly weekly chart ----
fig = go.Figure()
for r in comp_rows:
    df = r["aligned"].reset_index().rename(columns={"index":"date"})
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["diff_pct"], mode="lines+markers", name=r["ticker"]
    ))

fig.update_layout(
    xaxis_title="Date (Weekly, Fri close)",
    yaxis_title="Stock % - NASDAQ % (weekly)",
    legend_title="Ticker",
    hovermode="closest", height=600
)
st.subheader("Weekly % difference chart")
st.plotly_chart(fig, use_container_width=True)

# ---- Cumulative returns ----
cum_fig = go.Figure()
index_cum = (1 + index_weekly/100).cumprod() - 1
cum_fig.add_trace(go.Scatter(x=index_weekly.index, y=index_cum*100, mode="lines", name="NASDAQ"))

for r in comp_rows:
    sw = r["aligned"]["stock_weekly_pct"] / 100
    cum = (1 + sw).cumprod() - 1
    cum_fig.add_trace(go.Scatter(x=r["aligned"].index, y=cum*100, mode="lines", name=r["ticker"]))

cum_fig.update_layout(
    xaxis_title="Date (Weekly, Fri close)",
    yaxis_title="Cumulative return (%)",
    legend_title="Ticker", height=600
)
st.subheader("Cumulative return comparison")
st.plotly_chart(cum_fig, use_container_width=True)

# ---- News ----
if show_news:
    st.subheader(f"Latest news for current outperformers (> +{threshold_pct}%)")
    outperf = [r for r in comp_rows if r["latest_diff_pct"] > threshold_pct]
    if not outperf:
        st.write("No current outperformers above threshold.")
    else:
        API_KEY = "YOUR_FINNHUB_API_KEY"
        finnhub_client = finnhub.Client(api_key=API_KEY)
        _from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        _to = datetime.now().strftime('%Y-%m-%d')
        for r in outperf:
            st.markdown(f"**{r['ticker']}**")
            try:
                news = finnhub_client.company_news(symbol=r['ticker'], _from=_from, to=_to)
            except Exception as e:
                news = []
                st.write(f"Error fetching news for {r['ticker']}: {e}")
            if not news:
                st.write("No news found.")
            else:
                for a in news[:5]:
                    headline = a.get("headline", "No title")
                    url = a.get("url")
                    date = pd.to_datetime(a.get("datetime", None), unit='s').strftime('%Y-%m-%d') if a.get("datetime") else ""
                    if url:
                        st.markdown(f"- [{headline}]({url}) ({date})")
                    else:
                        st.markdown(f"- {headline} ({date})")

st.markdown("---")
st.caption("Data source: Yahoo Finance via yfinance. Index: Nasdaq Composite (^IXIC).")

