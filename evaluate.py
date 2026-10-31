"""Module 4 - Model evaluation, uncertainty quantification and KPIs (reads backtest_forecasts.csv)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import *

HEADROOM = 0.05          # default capacity-stress threshold: care load 5% above the origin level
Q = {"80": (0.10, 0.90), "95": (0.025, 0.975)}


def metrics(g):
    e = g["forecast"] - g["actual"]
    nz = g["actual"] >= 1
    mape = float(np.mean(np.abs(e[nz]) / g["actual"][nz]) * 100) if nz.any() else np.nan
    return pd.Series({"MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)), "MAPE": mape,
                      "WAPE": float(np.abs(e).sum() / max(g["actual"].sum(), 1e-9) * 100),
                      "n": len(g)})


def add_rel_resid(bt):
    bt["resid"] = bt["actual"] - bt["forecast"]
    bt["rel_resid"] = bt["resid"] / bt["scale"]
    return bt


def residual_quantiles(bt):
    """Empirical relative-residual quantiles per target/model/horizon (observed target days only)."""
    rows = []
    obs = bt[bt["observed"]]
    for (t, m, h), g in obs.groupby(["target", "model", "horizon"]):
        r = g["rel_resid"].values
        rows.append({"target": t, "model": m, "horizon": h, "n": len(r),
                     **{f"q{int(p*1000):04d}": np.quantile(r, p) for p in (0.025, 0.1, 0.5, 0.9, 0.975)}})
    return pd.DataFrame(rows)


def rolling_coverage(bt):
    """Honest out-of-sample interval coverage: quantiles for an origin use only residuals that were
    already observable (origin_j + h <= origin_i)."""
    rows = []
    obs = bt[bt["observed"]].copy()
    for (t, m, h), g in obs.groupby(["target", "model", "horizon"]):
        g = g.sort_values("origin")
        for _, r in g.iterrows():
            past = g[g["target_date"] <= r["origin"]]
            if len(past) < 8:
                continue
            q = np.quantile(past["rel_resid"], [0.025, 0.1, 0.9, 0.975])
            x = r["rel_resid"]
            rows.append({"target": t, "model": m, "horizon": h,
                         "in80": q[1] <= x <= q[2], "in95": q[0] <= x <= q[3],
                         "w80": (q[2] - q[1]) * r["scale"], "w95": (q[3] - q[0]) * r["scale"]})
    c = pd.DataFrame(rows)
    return c.groupby(["target", "model", "horizon"]).agg(cover80=("in80", "mean"), cover95=("in95", "mean"),
                                                        width80=("w80", "mean"), width95=("w95", "mean"),
                                                        n=("in80", "size")).reset_index()


def surge_events(bt, target="hhs"):
    """Capacity-stress warning performance. Stress threshold at origin = origin level * (1+HEADROOM).
    Event: any observed actual in (origin, origin+28d] reaches the threshold.
    Warning: the model's forecast reaches the threshold at some evaluated horizon <= 28.
    Lead time = first horizon (days) at which the warning crossed, counted for correctly warned events."""
    b = bt[(bt["target"] == target)]
    out = []
    for (m, o), g in b.groupby(["model", "origin"]):
        g = g.sort_values("horizon")
        thr = g["origin_level"].iloc[0] * (1 + HEADROOM)
        ev = bool((g["actual"][g["observed"]] >= thr).any())
        warn_h = g.loc[g["forecast"] >= thr, "horizon"]
        out.append({"model": m, "origin": o, "event": ev, "warned": len(warn_h) > 0,
                    "lead": float(warn_h.min()) if len(warn_h) else np.nan})
    e = pd.DataFrame(out)
    res = []
    for m, g in e.groupby("model"):
        tp = (g.event & g.warned).sum(); fn = (g.event & ~g.warned).sum(); fp = (~g.event & g.warned).sum()
        res.append({"model": m, "events": int(g.event.sum()), "hits": int(tp), "missed": int(fn),
                    "false_alarms": int(fp), "recall": tp / max(tp + fn, 1) * 100,
                    "precision": tp / max(tp + fp, 1) * 100,
                    "avg_lead_days": float(g.loc[g.event & g.warned, "lead"].mean()) if tp else np.nan})
    return pd.DataFrame(res)


def stability(bt, target="hhs"):
    """Forecast Stability Index: 100*(1 - mean |revision| / forecast level) between forecasts issued ~7 days apart
    for the same target date."""
    b = bt[bt["target"] == target][["model", "origin", "horizon", "target_date", "forecast"]]
    j = b.merge(b, on=["model", "target_date"], suffixes=("_a", "_b"))
    j = j[(j["origin_b"] - j["origin_a"]).dt.days.between(5, 9) & (j["horizon_a"] > j["horizon_b"])]
    j["rev"] = (j["forecast_a"] - j["forecast_b"]).abs() / j["forecast_b"].abs().clip(lower=1)
    return j.groupby("model")["rev"].mean().mul(100).rsub(100).rename("stability_index").reset_index()


def robustness(bt):
    """Model robustness = share of walk-forward folds (origin x horizon, observed targets) where the model's absolute
    error beats naive persistence; long-term reliability = MAE skill vs naive for horizons 14-28 days."""
    res = []
    obs = bt[bt["observed"]].copy()
    obs["ae"] = obs["resid"].abs()
    for t, gt in obs.groupby("target"):
        naive = gt[gt.model == "Naive"].set_index(["origin", "horizon"])["ae"]
        for m, g in gt.groupby("model"):
            a = g.set_index(["origin", "horizon"])["ae"]
            j = pd.concat([a, naive], axis=1, keys=["m", "n"]).dropna()
            win = float((j["m"] < j["n"]).mean() * 100)
            lt = j[j.index.get_level_values("horizon") >= 14]
            skill = float((1 - lt["m"].mean() / max(lt["n"].mean(), 1e-9)) * 100)
            res.append({"target": t, "model": m, "robustness_win_rate": win, "long_term_skill_vs_naive": skill})
    return pd.DataFrame(res)


def main():
    bt = pd.read_csv(OUT / "backtest_forecasts.csv", parse_dates=["origin", "target_date"])
    bt = add_rel_resid(bt)
    obs = bt[bt["observed"]]
    by_h = obs.groupby(["target", "model", "horizon"]).apply(metrics, include_groups=False).reset_index()
    overall = obs.groupby(["target", "model"]).apply(metrics, include_groups=False).reset_index()
    by_h.to_csv(OUT / "metrics_by_horizon.csv", index=False)
    overall.to_csv(OUT / "metrics_overall.csv", index=False)
    rq = residual_quantiles(bt); rq.to_csv(OUT / "residual_quantiles.csv", index=False)
    cov = rolling_coverage(bt); cov.to_csv(OUT / "interval_coverage.csv", index=False)
    sg = surge_events(bt); sg.to_csv(OUT / "surge_warning.csv", index=False)
    st = stability(bt); st.to_csv(OUT / "stability.csv", index=False)
    rb = robustness(bt); rb.to_csv(OUT / "robustness.csv", index=False)

    # model ranking per target (mean MAE rank across horizons, skipping baselines is NOT done - baselines compete too)
    rk = by_h.copy()
    rk["rank"] = rk.groupby(["target", "horizon"])["MAE"].rank()
    best = rk.groupby(["target", "model"])["rank"].mean().reset_index().sort_values(["target", "rank"])
    best.to_csv(OUT / "model_ranking.csv", index=False)
    print(best.to_string())
    print(overall.round(2).to_string())


if __name__ == "__main__":
    main()
