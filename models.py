"""Module 2/3 - Feature engineering and forecasting models (shared by backtest and final forecast)."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from config import STAT_WINDOW, SEED

warnings.filterwarnings("ignore")
SERIES = ["hhs", "discharges", "transfers", "intake", "cbp"]
_hol = USFederalHolidayCalendar()


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    """Features known at the end of day t (only information up to t is used)."""
    F = pd.DataFrame(index=d.index)
    for c in SERIES:
        F[f"{c}_t0"] = d[c]
        for k in (1, 7, 14):                                 # lag features t-1, t-7, t-14
            F[f"{c}_lag{k}"] = d[c].shift(k)
    for c in ("hhs", "discharges", "transfers"):             # rolling mean & variance
        for w in (7, 14):
            F[f"{c}_rm{w}"] = d[c].rolling(w).mean()
            F[f"{c}_rv{w}"] = d[c].rolling(w).var()
    net = d["transfers"] - d["discharges"]                   # net pressure indicator
    F["net_t0"] = net
    F["net_rm7"] = net.rolling(7).mean()
    F["net_rm14"] = net.rolling(14).mean()
    F["hhs_chg7"] = d["hhs"] - d["hhs"].shift(7)
    F["hhs_chg14"] = d["hhs"] - d["hhs"].shift(14)
    F["origin_dow"] = d.index.dayofweek
    return F


def target_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    hol = _hol.holidays(start=dates.min() - pd.Timedelta(days=5), end=dates.max() + pd.Timedelta(days=40))
    return pd.DataFrame({"tgt_dow": dates.dayofweek, "tgt_month": dates.month,
                         "tgt_holiday": dates.isin(hol).astype(int)}, index=dates)


def base_level(d, target):
    """Level the ML models forecast around (last value for the stock, 7-day mean for flows)."""
    return d[target] if target == "hhs" else d[target].rolling(7).mean()


def make_ml(name):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_leaf=5,
                                     max_features=0.6, random_state=SEED, n_jobs=1)
    return GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=3,
                                     subsample=0.8, random_state=SEED)


def ml_forecast(d, F, target, origin, horizons, name):
    """Direct multi-horizon forecast: one model per horizon, trained only on targets known at origin."""
    base = base_level(d, target)
    preds = {}
    x0 = F.loc[[origin]]
    for h in horizons:
        y = (d[target].shift(-h) - base)                      # target_{t+h} - level_t
        idx = F.index[(F.index <= origin - pd.Timedelta(days=h))]
        X = F.loc[idx]
        yy = y.loc[idx]
        ok = X.notna().all(axis=1) & yy.notna()
        X, yy = X[ok], yy[ok]
        cal_tr = target_calendar(X.index + pd.Timedelta(days=h))
        cal_tr.index = X.index
        Xtr = pd.concat([X, cal_tr], axis=1)
        tdate = origin + pd.Timedelta(days=h)
        cal0 = target_calendar(pd.DatetimeIndex([tdate]))
        cal0.index = x0.index
        xte = pd.concat([x0, cal0], axis=1)[Xtr.columns]
        m = make_ml(name).fit(Xtr.values, yy.values)
        preds[h] = float(base.loc[origin] + m.predict(xte.values)[0])
    return preds


def stat_forecast(d, target, origin, steps, name):
    s = d.loc[:origin, target].astype(float).iloc[-STAT_WINDOW[target]:]
    s = s.asfreq("D")
    try:
        if name == "ARIMA":
            order, trend = ((2, 1, 1), "n") if target == "hhs" else ((2, 0, 1), "c")
            fc = ARIMA(s, order=order, trend=trend).fit().forecast(steps)
        elif name == "SARIMA":
            order = (1, 1, 1) if target == "hhs" else (1, 0, 1)
            trend = "n" if target == "hhs" else "c"
            fc = SARIMAX(s, order=order, seasonal_order=(1, 0, 1, 7), trend=trend,
                         enforce_stationarity=False, enforce_invertibility=False
                         ).fit(disp=False, maxiter=60).forecast(steps)
        else:  # Exponential smoothing
            if target == "hhs":
                fit = ExponentialSmoothing(s, trend="add", damped_trend=True, initialization_method="estimated").fit()
            else:
                fit = ExponentialSmoothing(s, trend=None, seasonal="add", seasonal_periods=7,
                                           initialization_method="estimated").fit()
            fc = fit.forecast(steps)
        out = np.asarray(fc, dtype=float)
        if not np.isfinite(out).all():
            raise ValueError("non-finite forecast")
    except Exception:
        out = np.repeat(float(s.iloc[-1]), steps)               # safe fallback = persistence
    return out


def baseline_forecast(d, target, origin, steps, name):
    if name == "Naive":
        return np.repeat(float(d.loc[origin, target]), steps)
    return np.repeat(float(d.loc[:origin, target].iloc[-7:].mean()), steps)


def forecast_all(d, F, target, origin, horizons, models):
    """Return {model: {h: forecast}} for one origin."""
    steps = max(horizons)
    out = {}
    for m in models:
        if m in ("Naive", "Moving Avg"):
            arr = baseline_forecast(d, target, origin, steps, m)
            out[m] = {h: float(arr[h - 1]) for h in horizons}
        elif m in ("ARIMA", "SARIMA", "Exp. Smoothing"):
            arr = stat_forecast(d, target, origin, steps, m)
            out[m] = {h: float(arr[h - 1]) for h in horizons}
        else:
            out[m] = ml_forecast(d, F, target, origin, horizons, m)
    return out
