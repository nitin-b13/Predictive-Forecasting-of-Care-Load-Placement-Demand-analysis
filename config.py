"""Shared configuration for the UAC forecasting project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "HHS_Unaccompanied_Alien_Children_Program.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

COLS = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "intake",
    "Children in CBP custody": "cbp",
    "Children transferred out of CBP custody": "transfers",
    "Children in HHS Care": "hhs",
    "Children discharged from HHS Care": "discharges",
}

# Forecast targets: care load (stock), discharges and transfers-in (flows)
TARGETS = ["hhs", "discharges", "transfers"]
TARGET_LABEL = {
    "hhs": "Children in HHS care",
    "discharges": "Discharges (sponsor placements)",
    "transfers": "Transfers into HHS",
}

# Backtest design (strict time-based, expanding-window walk-forward)
BACKTEST_DAYS = 364          # test period = last 52 weeks (origins land on Sundays so every horizon hits a reporting day)
ORIGIN_STEP = 7              # a new forecast origin every 7 days
HORIZONS = [1, 3, 7, 14, 21, 28]
MAX_HORIZON = 28
STAT_WINDOW = {"hhs": 180, "discharges": 90, "transfers": 90}   # recent-regime fitting window (days) for statistical models

MODELS = ["Naive", "Moving Avg", "ARIMA", "SARIMA", "Exp. Smoothing",
          "Random Forest", "Gradient Boosting"]
STAT_MODELS = ["ARIMA", "SARIMA", "Exp. Smoothing"]
ML_MODELS = ["Random Forest", "Gradient Boosting"]
BASELINES = ["Naive", "Moving Avg"]

SEED = 42
