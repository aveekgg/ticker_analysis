# NSE Market Data

Streamlit app for browsing historical NSE price data (1149 stocks with
market cap > ₹2000cr, ~2M daily OHLCV rows, 2016-2026) with Stan Weinstein
stage-analysis (Stage 1-4) annotations, plus a SQL query tab against the
underlying DuckDB file.

## Run locally

```
pip install -r requirements.txt
streamlit run market_data_app.py
```

## Data

`nse_market_data.duckdb` (tracked via Git LFS) contains:

- `instruments` — tradingsymbol, Kite instrument_token, exchange
- `daily_prices` — tradingsymbol, date, open, high, low, close, volume

Data was sourced via the Kite Connect API and a screener.in market-cap
screen. The data-collection scripts used to build this file are not part
of this repo — this app only reads the pre-built DuckDB file.

## Stage classifier

`stage_classifier.py` labels each stock's weekly regime (Stage 1: base,
Stage 2: advance, Stage 3: top, Stage 4: decline) using a mix of 30-week
and daily (200d/50d) moving averages. Thresholds are tunable from the app's
"Stage classifier settings" panel; click "Help" in the app for a plain-
language explanation of the framework and current settings.
