"""Module 1 - Time-series preparation.
Date -> datetime index, continuity check, interpolation / masking of missing days,
trend-seasonality-residual decomposition."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import STL
from config import RAW_CSV, OUT, COLS


def load_raw():
    df = pd.read_csv(RAW_CSV, thousands=",")
    df = df.rename(columns=COLS)
    n_raw = len(df)
    df = df.dropna(how="all")                       # file ships ~450 empty trailing rows
    df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y")
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df, n_raw


def build_daily(df):
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    d = df.reindex(full)
    d.index.name = "date"
    d["is_observed"] = d["hhs"].notna()
    # interpolate: stock by linear interpolation; flows by linear interpolation as well, but every
    # interpolated row is flagged so evaluation can mask it out (masking = is_observed)
    for c in ["intake", "cbp", "transfers", "hhs", "discharges"]:
        d[c] = d[c].interpolate(method="linear", limit_direction="both")
    return d


def decompose(d):
    res = {}
    for c in ["hhs", "discharges", "transfers"]:
        s = d[c].astype(float)
        stl = STL(s, period=7, robust=True).fit()
        res[c] = pd.DataFrame({"observed": s, "trend": stl.trend,
                               "seasonal": stl.seasonal, "resid": stl.resid})
    out = pd.concat(res, axis=1)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out


def main():
    raw, n_raw = load_raw()
    d = build_daily(raw)
    dec = decompose(d)
    d.to_csv(OUT / "daily_clean.csv")
    dec.to_csv(OUT / "decomposition.csv")
    full_days = len(d)
    miss = d.loc[~d["is_observed"]]
    summary = {
        "raw_rows": int(n_raw), "empty_rows_dropped": int(n_raw - len(raw)),
        "observed_days": int(d["is_observed"].sum()), "calendar_days": int(full_days),
        "missing_days_interpolated": int((~d["is_observed"]).sum()),
        "missing_share_pct": round(100 * (~d["is_observed"]).mean(), 1),
        "missing_by_weekday": {k: int(v) for k, v in
                               miss.index.day_name().value_counts().items()},
        "start": str(d.index.min().date()), "end": str(d.index.max().date()),
        "longest_gap_days": int((raw.index.to_series().diff().dt.days.max()) - 1),
        "peak_hhs": int(raw["hhs"].max()), "peak_hhs_date": str(raw["hhs"].idxmax().date()),
        "min_hhs": int(raw["hhs"].min()), "min_hhs_date": str(raw["hhs"].idxmin().date()),
        "latest_hhs": int(raw["hhs"].iloc[-1]),
        "weekly_seasonal_strength_hhs": float(max(0, 1 - dec["hhs_resid"].var() /
                                    (dec["hhs_seasonal"] + dec["hhs_resid"]).var())),
        "weekly_seasonal_strength_discharges": float(max(0, 1 - dec["discharges_resid"].var() /
                                    (dec["discharges_seasonal"] + dec["discharges_resid"]).var())),
    }
    json.dump(summary, open(OUT / "data_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
