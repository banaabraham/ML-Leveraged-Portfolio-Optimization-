**Concept Notebook Overview**

- File: `concept.ipynb`
- Goal: Prototype a simple end‑to‑end pipeline that forecasts next‑day prices for a basket of equities and runs a daily portfolio optimization based on the forecasts, then evaluates results versus a historical‑return benchmark.

**What It Does**

- Load tickers: Reads `sp500_companies.csv` and takes the top 50 symbols from the `Symbol` column.
- Download data: Uses `yfinance` to fetch daily OHLCV for those tickers (example start: `2024‑01‑01`).
- Prep features: Builds a multivariate time series of daily closing prices and fills gaps.
- Forecast: Trains rolling 1‑step‑ahead models using `darts`:
  - `LightGBMModel(lags=12)`
  - `LinearRegressionModel(lags=10)`
- Metrics: Computes `MAE`, `MAPE`, and `RMSE` on the aligned actuals vs. predictions.
- Portfolio step: For each day, solves a minimum‑variance optimization in `cvxpy` with a minimum expected return constraint using the model’s predicted returns; long‑only, weights sum to 1.
- Benchmark: Repeats the optimization but with expected returns from historical averages (no model), as a baseline.
- Simulation: Compounds realized portfolio returns, computes wealth curve, cumulative return, and annualized Sharpe.
- Visualization: Plots selected series and the forecast overlay for quick inspection.

**Key Objects/Variables**

- `top50_sample`: list of tickers selected from the S&P 500 list.
- `data`: pandas DataFrame of prices indexed by `Date` (daily).
- `value_cols`: list of price columns used to build the multivariate series (one per ticker).
- `series_multi`: `darts.TimeSeries` built from `data` (daily frequency, missing values filled).
- `pred`: concatenated predictions from rolling forecasts; `pred_df` is the pandas view.
- `trades_df`: simulation output per day with `port_ret`, `wealth`, `cum_return`, and per‑asset `w_...` (weights) and `ret_...` (realized returns) columns.
- `simulate(model)`: wraps the full rolling‑forecast → optimization → evaluation loop for a given model.

**Dependencies**

- Python 3.11 
- pandas, numpy, matplotlib, tqdm
- yfinance
- darts (TimeSeries + forecasting models)
- lightgbm (backend used by `LightGBMModel`)
- cvxpy + a solver (the notebook uses `ECOS`)

Example install (adjust versions as needed):

- `pip install pandas numpy matplotlib tqdm yfinance` 
- `pip install darts==0.29.0`
- `pip install lightgbm`
- `pip install cvxpy==1.7.1 ecos==2.0.14`

**Data Input**

- Place `sp500_companies.csv` alongside the notebook.
- Expected column: `Symbol` (used to build the ticker list).
- Prices are downloaded on the fly via `yfinance` using those symbols.

**How To Run**

- Open `concept.ipynb` in Jupyter or VS Code and run cells top‑to‑bottom.
- Ensure `sp500_companies.csv` is present and dependencies are installed.
- The notebook prints forecast metrics and the portfolio’s annualized Sharpe, and produces plots inline.

**Notable Parameters**

- Rolling window: `initial_window = 30`, `train_length = 30`.
- Forecast horizon: `h = 1` (next‑day).
- Optimization: long‑only, weights sum to 1, minimum expected daily return `req_return = 0.005`.
- Missing data: handled with `darts.utils.missing_values.fill_missing_values`.

**Limitations & Notes**

- This is a concept/prototype for exploration; it does not handle transaction costs, slippage, corporate actions, or survivorship bias.
- `yfinance` can return multi‑index columns; the notebook selects/reshapes closing prices into `value_cols` before building `TimeSeries`.
- `cvxpy` requires a working solver (`ECOS` is called). Install and verify the solver for your platform.
- Results depend heavily on date range, lags, training length, and the required return constraint.

**File**

- Notebook: `concept.ipynb`
- Input list: `sp500_companies.csv`