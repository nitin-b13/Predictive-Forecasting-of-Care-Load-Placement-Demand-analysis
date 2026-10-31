# UAC Care-Load Forecast Desk

Forecasting children in HHS care, discharge demand and capacity stress for the Unaccompanied Children (UAC) Program.
Data: `data/HHS_Unaccompanied_Alien_Children_Program.csv` (daily reports, 12 Jan 2023 to 21 Dec 2025).

## What is in the box
| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (forecast chart, discharge demand, model lab, confidence intervals, scenarios) |
| `pipeline/prepare.py` | Time-series preparation: datetime index, gap handling, STL decomposition |
| `pipeline/models.py` | Feature engineering and the seven forecasting models |
| `pipeline/backtest.py` | Strict time-based walk-forward validation (52 weekly origins, horizons 1-28 days) |
| `pipeline/evaluate.py` | MAE / RMSE / MAPE / WAPE, horizon error, interval coverage, surge warning, KPIs |
| `pipeline/finalize.py` | Final 28-day forecasts with 80% / 95% intervals and the KPI summary |
| `pipeline/figures.py` | Charts used in the paper |
| `pipeline/dashboard_html.py` | Builds `dashboard.html`, the standalone interactive dashboard |
| `dashboard.html` | Open in any browser (double-click); same views and controls as the Streamlit app, works offline |
| `outputs/` | All generated tables, forecasts and figures (the dashboard reads these) |
| `docs/` | Research paper and executive summary (Word) |

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py                 # dashboard (uses the saved outputs/)
python run_pipeline.py               # rebuild everything (~25-30 min, mostly the backtest)
python run_pipeline.py --skip-backtest   # refresh metrics, forecasts and figures only
```

## Dashboard controls
Forecast horizon selector (7/14/21/28 days), primary model toggle plus model overlay, 80% / 95% uncertainty band,
capacity headroom slider, and a scenario comparison tab (custom transfer and discharge changes).

## Deploy on Streamlit Community Cloud
Push the whole folder (including `outputs/`) to GitHub, then create an app with main file `app.py`.
The app only reads the saved outputs, so it starts quickly and needs no retraining.

## Method notes
* Missing days (33%, mostly Fridays and Saturdays) are interpolated for a continuous index and excluded from scoring.
* Walk-forward origins are real reporting days, so no interpolated value leaks future information.
* Prediction intervals come from relative backtest errors per model and horizon; coverage is checked out of sample.
* The 5% capacity headroom and the "stress event" definition are assumptions; change them in the dashboard.
* Scenario deltas use a fitted flow-to-care-load factor and are a planning aid, not a causal model.
