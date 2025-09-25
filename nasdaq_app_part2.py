import finnhub
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_plotly_events import plotly_events


# Must be first Streamlit command
st.set_page_config(layout="wide", page_title="Stock vs NASDAQ (^IXIC) — Weekly Comparison")

# Now you can create tabs
tab1, tab2 = st.tabs(["Weekly Comparison", "Overall Market Summary"])




######################
# --- TAB 1: Weekly Comparison ---
######################
with tab1:

    

    # ---- Sidebar controls ----
    st.sidebar.header("Controls")
    
    # --- Fetch tickers from Google Sheet ---
    sheet_url = "https://docs.google.com/spreadsheets/d/1FWMLpoN3EMDZBL-XQVgMo5wyEQKncwmS3FlZce1CENU/export?format=csv&gid=0"
    try:
        sheet_df = pd.read_csv(sheet_url)
        available_tickers = sheet_df.iloc[:, 0].dropna().tolist()
    except Exception as e:
        st.warning(f"Could not load tickers from Google Sheet: {e}")
        available_tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOG", "META", "NFLX"]
    
    tickers = st.sidebar.multiselect("Select tickers 💹", available_tickers, default=available_tickers[:3])
    
    # # Market cap range
    # st.sidebar.subheader("Market Cap Filter (USD Billions)")
    # min_mcap = st.sidebar.number_input("Min market cap", value=5.0, min_value=0.0, step=0.5)
    # max_mcap = st.sidebar.number_input("Max market cap (0 = no limit)", value=0.0, min_value=0.0, step=0.5)
    
    # Weeks and thresholds
    weeks = st.sidebar.slider("Select Weeks 🗓️", min_value=1, max_value=52, value=12)
    show_news = st.sidebar.checkbox("Fetch latest news for overperformers 🚀", value=True)
    compare_range = st.sidebar.selectbox("Quick range", options=["Past week", "Past 3 months", "Custom weeks"], index=2)
    if compare_range == "Past week":
        weeks = 1
    elif compare_range == "Past 3 months":
        weeks = 12
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Select Threshold 💯")
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
            # try:
            #     mc = tk.info.get("marketCap", None)
            # except Exception:
            #     mc = None
            info[t] = {
                "history": data,
                # "marketCap": mc
            }
        return info
    
    # Fetch data
    all_symbols = tickers + [index_ticker]
    with st.spinner("Fetching data..."):
        info = fetch_tickers_info(all_symbols)
    
    # Filter tickers by market cap and data availability
    filtered = []
    filtered = [t for t in tickers if not info[t]["history"].empty]
    if len(filtered) == 0:
        st.error("No tickers had usable price data.")
        st.stop()

    # filtered_reasons = {}
    # for t in tickers:
    #     mcap = info[t]["marketCap"]
    #     if mcap is None:
    #         filtered_reasons[t] = "market cap unknown"
    #         continue
    #     if mcap < min_mcap * 1e9:
    #         filtered_reasons[t] = f"market cap {mcap:.0f} < {min_mcap*1e9:.0f}"
    #         continue
    #     if max_mcap > 0 and mcap > max_mcap * 1e9:
    #         filtered_reasons[t] = f"market cap {mcap:.0f} > {max_mcap*1e9:.0f}"
    #         continue
    #     if info[t]["history"].empty:
    #         filtered_reasons[t] = "no price history"
    #         continue
    #     filtered.append(t)
    
    # if len(filtered) == 0:
    #     st.error("No tickers passed the market-cap filter or had usable price data. See reasons below.")
    #     st.write(pd.DataFrame.from_dict(filtered_reasons, orient="index", columns=["reason"]))
    #     st.stop()
    
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
            # "marketCap": info[t]["marketCap"],
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
        #"MarketCap (USD)": r["marketCap"],
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
        st.markdown(f"**{r['ticker']}** — status: **{r['status']}**") ### market cap yaha change kya tha ---- market cap: {r['marketCap']:,}
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
        outperf = [r for r in comp_rows if r["aligned"]["diff_cum_pct"].iloc[-1] > threshold_pct]
    
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


    


######################
# --- TAB 2: Overall Market Summary vs NASDAQ ---
######################
# with tab2:

    
#     st.header("Overall Market Summary vs NASDAQ")

#     # Sidebar inputs
#     threshold = st.sidebar.slider("Out/Underperform threshold (%)", 1, 20, 5)
    
#     # Fetch tickers from Google Sheet
#     sheet_url = "https://docs.google.com/spreadsheets/d/1FWMLpoN3EMDZBL-XQVgMo5wyEQKncwmS3FlZce1CENU/export?format=csv&gid=0"
#     try:
#         tickers_df = pd.read_csv(sheet_url)
#         tickers_gs = tickers_df.iloc[:, 0].dropna().tolist()
#     except:
#         st.warning("Could not load tickers from Google Sheet. Using defaults.")
#         tickers_gs = ["AAPL","MSFT","TSLA","AMZN","GOOG","META"]
    
#     index_ticker = "^IXIC"
#     period = "2y"   # fetch enough history
    
#     @st.cache_data
#     def get_weekly_prices(tickers, period="2y"):
#         """Fetch weekly closing prices for tickers and index."""
#         all_prices = pd.DataFrame()
#         for t in tickers:
#             try:
#                 data = yf.Ticker(t).history(period=period)["Close"]
#                 weekly = data.resample("W-FRI").last()
#                 all_prices[t] = weekly
#             except:
#                 continue
#         return all_prices
    
#     with st.spinner("Fetching weekly prices..."):
#         prices_df = get_weekly_prices(tickers_gs + [index_ticker], period=period)
    
#     if prices_df.empty:
#         st.info("No data available for selected tickers.")
#         st.stop()
    
#     # --- Week mapping ---
#     week_labels = [f"Week {i+1} — {date.strftime('%Y-%m-%d')}" for i, date in enumerate(prices_df.index)]
#     total_weeks = len(week_labels)
    
#     if total_weeks < 2:
#         st.info("Not enough weeks of data available.")
#         st.stop()
    
#     # Sidebar range selector (choose start → end week with dates)
#     week_start, week_end = st.sidebar.select_slider(
#         "Select Week Range 🗓️",
#         options=week_labels,
#         value=(week_labels[0], week_labels[min(11, total_weeks-1)])  # default: Week 1 → Week 12
#     )
    
#     # Convert selected labels back to indices
#     start_idx = week_labels.index(week_start)
#     end_idx = week_labels.index(week_end)
    
#     # Slice dataframe
#     prices_df = prices_df.iloc[start_idx:end_idx+1]
    
#     # --- Compute cumulative % change (compounded) vs NASDAQ ---
#     cum_df = pd.DataFrame()
#     for t in tickers_gs:
#         if t in prices_df.columns:
#             stock_cum = (prices_df[t] / prices_df[t].iloc[0] - 1) * 100
#             index_cum = (prices_df[index_ticker] / prices_df[index_ticker].iloc[0] - 1) * 100
#             cum_df[t] = stock_cum - index_cum
    
#     if cum_df.empty:
#         st.info("No cumulative data could be calculated.")
#         st.stop()
    
#     # --- Cumulative moving average (CMA) ---
#     cma_df = cum_df.expanding().mean().round(2)
    
#     # Show CMA table
#     st.subheader("Cumulative Moving Average (CMA) vs NASDAQ")
#     st.dataframe(cma_df)
    
#     # --- Latest cumulative difference ---
#     latest_diff = cum_df.iloc[-1]
#     summary_df = pd.DataFrame({
#         "Cumulative % vs NASDAQ": latest_diff.round(2)
#     })
#     summary_df["Status"] = summary_df["Cumulative % vs NASDAQ"].apply(
#         lambda x: "Outperformer ✅" if x > threshold else ("Underperformer ❌" if x < -threshold else "Neutral")
#     )
    
#     st.subheader(f"Cumulative Performance Summary (Threshold ±{threshold}%)")
#     st.dataframe(summary_df)
    
#     # --- Filter tickers breaching threshold ---
#     perf_table = summary_df[
#         (summary_df["Cumulative % vs NASDAQ"] >= threshold) |
#         (summary_df["Cumulative % vs NASDAQ"] <= -threshold)
#     ]
#     if perf_table.empty:
#         st.info(f"No tickers breached ±{threshold}% vs NASDAQ in the selected range.")
#     else:
#         st.subheader(f"Tickers breaching ±{threshold}%")
#         st.dataframe(perf_table)
    
#     # --- Interactive Plot ---
#     st.subheader("Cumulative % vs NASDAQ — Compounded with Moving Average")
#     fig = go.Figure()
    
#     for t in cum_df.columns:
#         # Raw cumulative
#         fig.add_trace(go.Scatter(
#             x=cum_df.index,
#             y=cum_df[t],
#             mode="lines+markers",
#             name=f"{t} Cumulative"
#         ))
#         # Moving average
#         fig.add_trace(go.Scatter(
#             x=cma_df.index,
#             y=cma_df[t],
#             mode="lines",
#             name=f"{t} CMA",
#             line=dict(dash="dot")
#         ))
    
#     # Threshold lines
#     fig.add_hline(y=threshold, line=dict(color="green", dash="dot"), annotation_text=f"+{threshold}%")
#     fig.add_hline(y=-threshold, line=dict(color="red", dash="dot"), annotation_text=f"-{threshold}%")
    
#     fig.update_layout(
#         yaxis_title="Cumulative % vs NASDAQ",
#         xaxis_title="Week End Date",
#         legend_title="Tickers",
#         hovermode="closest",
#         height=600
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # --- Exact Cumulative % Difference Chart from old code ---
#     st.subheader("Cumulative % vs NASDAQ — Compounded (Original Graph)")
    
#     fig_old = go.Figure()
#     for t in cum_df.columns:
#         fig_old.add_trace(go.Scatter(
#             x=cum_df.index,
#             y=cum_df[t],
#             mode="lines+markers",
#             name=t
#         ))
    
#     # Threshold lines
#     fig_old.add_hline(y=threshold, line=dict(color="green", dash="dot"), annotation_text=f"+{threshold}%")
#     fig_old.add_hline(y=-threshold, line=dict(color="red", dash="dot"), annotation_text=f"-{threshold}%")
    
#     fig_old.update_layout(
#         yaxis_title="Cumulative % vs NASDAQ",
#         xaxis_title="Week End Date",
#         legend_title="Tickers",
#         height=600
#     )
#     st.plotly_chart(fig_old, use_container_width=True)

with tab2:
    st.header("Overall Market Summary vs NASDAQ")

    # Sidebar inputs
    threshold = st.sidebar.slider("Out/Underperform threshold (%)", 1, 20, 5)

    # Fetch tickers from Google Sheet
    sheet_url = "https://docs.google.com/spreadsheets/d/1FWMLpoN3EMDZBL-XQVgMo5wyEQKncwmS3FlZce1CENU/export?format=csv&gid=0"
    try:
        tickers_df = pd.read_csv(sheet_url)
        tickers_gs = tickers_df.iloc[:, 0].dropna().tolist()
    except:
        st.warning("Could not load tickers from Google Sheet. Using defaults.")
        tickers_gs = ["AAPL","MSFT","TSLA","AMZN","GOOG","META"]

    index_ticker = "^IXIC"
    period = "2y"  # fetch enough history

    @st.cache_data
    def get_weekly_prices(tickers, period="2y"):
        """Fetch weekly closing prices for tickers and index."""
        all_prices = pd.DataFrame()
        for t in tickers:
            try:
                data = yf.Ticker(t).history(period=period)["Close"]
                weekly = data.resample("W-FRI").last()
                all_prices[t] = weekly
            except:
                continue
        return all_prices

    with st.spinner("Fetching weekly prices..."):
        prices_df = get_weekly_prices(tickers_gs + [index_ticker], period=period)

    if prices_df.empty:
        st.info("No data available for selected tickers.")
        st.stop()

    # --- Week mapping & range selection ---
    week_labels = [f"Week {i+1} — {date.strftime('%Y-%m-%d')}" for i, date in enumerate(prices_df.index)]
    total_weeks = len(week_labels)

    if total_weeks < 2:
        st.info("Not enough weeks of data available.")
        st.stop()

    week_start, week_end = st.sidebar.select_slider(
        "Select Week Range 🗓️",
        options=week_labels,
        value=(week_labels[0], week_labels[min(11, total_weeks-1)])
    )
    start_idx = week_labels.index(week_start)
    end_idx = week_labels.index(week_end)
    prices_df = prices_df.iloc[start_idx:end_idx+1]

    # --- Compute cumulative % vs NASDAQ ---
    cum_df = pd.DataFrame()
    for t in tickers_gs:
        if t in prices_df.columns:
            stock_cum = (prices_df[t] / prices_df[t].iloc[0] - 1) * 100
            index_cum = (prices_df[index_ticker] / prices_df[index_ticker].iloc[0] - 1) * 100
            cum_df[t] = stock_cum - index_cum

    if cum_df.empty:
        st.info("No cumulative data could be calculated.")
        st.stop()

    # --- Cumulative moving average ---
    cma_df = cum_df.expanding().mean().round(2)
    st.subheader("Cumulative Moving Average (CMA) vs NASDAQ")
    st.dataframe(cma_df)

    # --- Latest cumulative difference ---
    latest_diff = cum_df.iloc[-1]
    summary_df = pd.DataFrame({
        "Cumulative % vs NASDAQ": latest_diff.round(2)
    })
    summary_df["Status"] = summary_df["Cumulative % vs NASDAQ"].apply(
        lambda x: "Outperformer ✅" if x > threshold else ("Underperformer ❌" if x < -threshold else "Neutral")
    )
    st.subheader(f"Cumulative Performance Summary (Threshold ±{threshold}%)")
    st.dataframe(summary_df)

    # --- Interactive Plot with CMA ---
    st.subheader("Cumulative % vs NASDAQ — Compounded with Moving Average")
    fig = go.Figure()
    for t in cum_df.columns:
        fig.add_trace(go.Scatter(
            x=cum_df.index,
            y=cum_df[t],
            mode="lines+markers",
            name=f"{t} Cumulative",
            customdata=[t]*len(cum_df)  # store ticker for click events
        ))
        fig.add_trace(go.Scatter(
            x=cma_df.index,
            y=cma_df[t],
            mode="lines",
            name=f"{t} CMA",
            line=dict(dash="dot"),
            customdata=[t]*len(cum_df)
        ))
    fig.add_hline(y=threshold, line=dict(color="green", dash="dot"), annotation_text=f"+{threshold}%")
    fig.add_hline(y=-threshold, line=dict(color="red", dash="dot"), annotation_text=f"-{threshold}%")
    fig.update_layout(
        yaxis_title="Cumulative % vs NASDAQ",
        xaxis_title="Week End Date",
        legend_title="Tickers",
        hovermode="closest",
        height=600
    )

    # --- Capture click events ---
    selected_points = plotly_events(fig, click_event=True, hover_event=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Mini Drill-Down Page ---
    if selected_points:
        clicked_ticker = selected_points[0]["customdata"]
        st.markdown(f"## 🔍 Detailed View: {clicked_ticker}")

        # Ticker cumulative & CMA
        ticker_cum = cum_df[clicked_ticker]
        ticker_cma = cma_df[clicked_ticker]

        # Performance status
        latest_val = ticker_cum.iloc[-1]
        status = "Outperformer ✅" if latest_val > threshold else ("Underperformer ❌" if latest_val < -threshold else "Neutral")

        # Display key stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Cum % vs NASDAQ", f"{latest_val:.2f}%")
        col2.metric("Average CMA", f"{ticker_cma.mean():.2f}%")
        col3.metric("Status", status)

        # Plot ticker chart
        fig_ticker = go.Figure()
        fig_ticker.add_trace(go.Scatter(
            x=ticker_cum.index,
            y=ticker_cum,
            mode="lines+markers",
            name=f"{clicked_ticker} Cumulative"
        ))
        fig_ticker.add_trace(go.Scatter(
            x=ticker_cma.index,
            y=ticker_cma,
            mode="lines",
            name=f"{clicked_ticker} CMA",
            line=dict(dash="dot")
        ))
        fig_ticker.add_hline(y=threshold, line=dict(color="green", dash="dot"), annotation_text=f"+{threshold}%")
        fig_ticker.add_hline(y=-threshold, line=dict(color="red", dash="dot"), annotation_text=f"-{threshold}%")
        fig_ticker.update_layout(
            yaxis_title="Cumulative % vs NASDAQ",
            xaxis_title="Week End Date",
            height=400
        )
        st.plotly_chart(fig_ticker, use_container_width=True)



