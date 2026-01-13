# Stock Dashboard for Streamlit in Snowflake
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# No st.set_page_config() for SiS

# Cache stock data
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch stock data from yfinance."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.reset_index(inplace=True)
    return df

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_info(ticker: str) -> dict:
    """Fetch stock info from yfinance."""
    stock = yf.Ticker(ticker)
    return stock.info

# App UI
st.title(":material/trending_up: Stock Dashboard")

# Sidebar
with st.sidebar:
    st.header(":material/settings: Settings")
    
    ticker = st.text_input("Stock Ticker:", value="AAPL", max_chars=10).upper()
    
    period = st.selectbox(
        "Time Period:",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    st.divider()
    st.caption("Data from Yahoo Finance")

# Main content
if ticker:
    try:
        with st.spinner(f"Loading {ticker} data..."):
            df = get_stock_data(ticker, period)
            info = get_stock_info(ticker)
        
        if df.empty:
            st.error(f"No data found for {ticker}")
            st.stop()
        
        # Company info
        company_name = info.get("longName", ticker)
        st.subheader(f"{company_name} ({ticker})")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = df["Close"].iloc[-1]
        prev_price = df["Close"].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_pct = (price_change / prev_price) * 100
        
        with col1:
            st.metric(
                "Price",
                f"${current_price:.2f}",
                f"{price_change:+.2f} ({price_pct:+.1f}%)"
            )
        
        with col2:
            st.metric("High", f"${df['High'].max():.2f}")
        
        with col3:
            st.metric("Low", f"${df['Low'].min():.2f}")
        
        with col4:
            avg_volume = df["Volume"].mean()
            st.metric("Avg Volume", f"{avg_volume/1e6:.1f}M")
        
        # Price chart
        st.subheader(":material/show_chart: Price History")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC"
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price ($)",
            xaxis_rangeslider_visible=False,
            height=400,
            margin=dict(t=10, l=0, r=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        st.subheader(":material/bar_chart: Volume")
        
        fig_vol = px.bar(
            df, x="Date", y="Volume",
            color_discrete_sequence=["#00BFC4"]
        )
        fig_vol.update_layout(
            xaxis_title="Date",
            yaxis_title="Volume",
            height=200,
            margin=dict(t=10, l=0, r=0, b=0)
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Data table
        with st.expander(":material/table_chart: View Raw Data"):
            st.dataframe(
                df[["Date", "Open", "High", "Low", "Close", "Volume"]].tail(20),
                use_container_width=True,
                hide_index=True
            )
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
