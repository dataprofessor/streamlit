# Stock Dashboard

> **TLDR:** A real-time stock dashboard with price charts, volume analysis, and key metrics using Yahoo Finance data.

## Features

- Candlestick price charts
- Volume bar charts
- Key metrics (price, high, low, avg volume)
- Adjustable time periods (1mo to 5y)
- Data caching for performance

## Run Locally

```bash
# Clone the repo
git clone <your-repo-url>
cd stock_dashboard

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy!

## Deploy to Streamlit in Snowflake

1. Create a Streamlit app in Snowflake
2. Upload app.py and requirements.txt
3. Add `yfinance` to external access (if required by your Snowflake setup)
4. Run from Snowflake UI

---

Built with [Streamlit](https://streamlit.io) and [yfinance](https://github.com/ranaroussi/yfinance)
