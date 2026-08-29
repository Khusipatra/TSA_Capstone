# Stock Price Forecasting & Portfolio Allocation

ARIMA, SARIMAX, GARCH, LSTM, GRU and Transformer compared on 5 NSE stocks: `TCS.NS`, `HDFCBANK.NS`, `SUNPHARMA.NS`, `NESTLEIND.NS`, `MARUTI.NS` (2015–present).

## Pipeline
- Download & clean data
- Engineer features: log returns, RSI, MACD, volume ratio, rolling volatility
- EDA & ADF stationarity check
- Fit ARIMA / SARIMAX
- Forecast volatility with GARCH
- Train LSTM / GRU / Transformer on windowed sequences
- Compare all models by RMSE
- Backtest a volatility-scaled long/cash strategy vs. buy-and-hold
- Build an inverse-volatility portfolio

## How to run
- Open in Jupyter/Colab, run top to bottom
- First cell installs `pmdarima` and `arch`
- Also needs: `numpy`, `pandas`, `matplotlib`, `seaborn`, `yfinance`, `statsmodels`, `tensorflow`, `scikit-learn`, `scipy`
- Price data caches to `<TICKER>.csv` after first download
- Slowest step: deep learning training (80 epochs × 3 models × 5 stocks)

## Key config (top of notebook)
- `STOCKS` — tickers included
- `START_DATE` / `END_DATE` — data range
- `TEST_SIZE` — 60-day holdout
- `WINDOW` — 60-day lookback
- `EPOCHS` — training epochs
- `CAPITAL` — ₹10,00,000
- `TRANSACTION_COST_BPS` — backtest cost assumption

## Notes
- Scalers fit on train only (no leakage)
- Test split is strictly time-ordered
- Models predict next-day log return, not price
- No short-selling in the backtest
- No results are hard-coded — all numbers regenerate on each run

## Outputs
- `final_portfolio.csv`
- `model_comparison.csv`
- `backtest_results.csv`
- Inline plots: forecasts, training loss, RMSE comparison, equity curves, allocation pie chart, correlation heatmap
