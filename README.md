# Stock Price Forecasting & Portfolio Allocation

ARIMA, SARIMAX, GARCH, LSTM, GRU and Transformer compared on 5 NSE stocks (`TCS.NS`, `HDFCBANK.NS`, `SUNPHARMA.NS`, `NESTLEIND.NS`, `MARUTI.NS`), 2015–present.

**Pipeline:** download & clean data → engineer features (log returns, RSI, MACD, volume ratio, rolling volatility) → EDA & ADF stationarity check → fit ARIMA/SARIMAX → forecast volatility with GARCH → train LSTM/GRU/Transformer on windowed sequences → compare all models by RMSE → backtest a volatility-scaled long/cash strategy vs. buy-and-hold → build an inverse-volatility portfolio.

**Run:** open in Jupyter/Colab, run top to bottom. First cell installs `pmdarima` and `arch`; also needs `numpy`, `pandas`, `matplotlib`, `seaborn`, `yfinance`, `statsmodels`, `tensorflow`, `scikit-learn`, `scipy`. Price data caches to `<TICKER>.csv` after first download. Deep learning training (80 epochs × 3 models × 5 stocks) is the slowest step.

**Key config** (top of notebook): `STOCKS`, `START_DATE`/`END_DATE`, `TEST_SIZE` (60-day holdout), `WINDOW` (60-day lookback), `EPOCHS`, `CAPITAL` (₹10,00,000), `TRANSACTION_COST_BPS`.

**Notes:** scalers fit on train only (no leakage); test split is strictly time-ordered; models predict next-day log return, not price; no short-selling in the backtest.

**Outputs:** `final_portfolio.csv`, `model_comparison.csv`, `backtest_results.csv`, plus inline plots (forecasts, training loss, RMSE comparison, equity curves, allocation pie chart, correlation heatmap).

No results are hard-coded — all numbers regenerate on each run.

*Educational/research use only — not investment advice.*
