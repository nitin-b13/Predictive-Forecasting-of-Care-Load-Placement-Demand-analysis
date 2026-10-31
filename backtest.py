"""Module 3 - Walk-forward validation (strict time split, expanding window, multi-horizon)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import *
from models import build_features, forecast_all


def pick_origins(d):
    last = d.index.max()
    start = last - pd.Timedelta(days=BACKTEST_DAYS)
    origins = []
    cur = start
    while cur <= last - pd.Timedelta(days=1):
        o = cur
        while not d.loc[o, "is_observed"]:        # origin must be a real (observed) reporting day
            o -= pd.Timedelta(days=1)
        origins.append(o)
        cur += pd.Timedelta(days=ORIGIN_STEP)
    return sorted(set(origins))


def main():
    d = pd.read_csv(OUT / "daily_clean.csv", parse_dates=["date"], index_col="date")
    F = build_features(d)
    origins = pick_origins(d)
    print(f"{len(origins)} origins from {origins[0].date()} to {origins[-1].date()}", flush=True)
    rows = []
    done = set()
    f = OUT / "backtest_forecasts.csv"
    if f.exists():                                   # resume support
        prev = pd.read_csv(f, parse_dates=["origin", "target_date"])
        rows = [tuple(r) for r in prev.itertuples(index=False, name=None)]
        done = set(prev["origin"].unique())
        print(f"resuming with {len(done)} origins done", flush=True)
    t0 = time.time()
    for i, o in enumerate(origins):
        if o in done:
            continue
        for target in TARGETS:
            res = forecast_all(d, F, target, o, HORIZONS, MODELS)
            lvl = d.loc[o, target]
            scale = lvl if target == "hhs" else d[target].iloc[max(0, d.index.get_loc(o) - 27):d.index.get_loc(o) + 1].mean() + 1.0
            for m, hv in res.items():
                for h, f in hv.items():
                    td = o + pd.Timedelta(days=h)
                    if td > d.index.max():
                        continue
                    rows.append((target, m, o, h, td, float(d.loc[td, target]), f, float(scale),
                                 float(lvl), bool(d.loc[td, "is_observed"])))
        print(f"origin {i+1}/{len(origins)} {o.date()}  elapsed {time.time()-t0:.0f}s", flush=True)
        if True:
            pd.DataFrame(rows, columns=["target", "model", "origin", "horizon", "target_date", "actual",
                                        "forecast", "scale", "origin_level", "observed"]
                         ).to_csv(OUT / "backtest_forecasts.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
