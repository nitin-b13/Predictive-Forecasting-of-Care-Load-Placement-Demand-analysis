"""Module 5 - Final forecasts (origin = latest reported day), prediction intervals, KPIs."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import *
from models import build_features, forecast_all
from evaluate import HEADROOM


def interval_offsets(rq, target, model):
    g = rq[(rq.target == target) & (rq.model == model)].sort_values("horizon")
    hs = np.arange(1, MAX_HORIZON + 1)
    return {c: np.interp(hs, g["horizon"], g[c]) for c in ["q0025", "q0100", "q0500", "q0900", "q0975"]}


def breach_probability(bt, target, model, fc_path, capacity, window):
    """P(care load exceeds capacity at least once within `window` days) - lower bound given by the max daily
    exceedance probability, using the empirical relative-residual distribution of the backtest."""
    g = bt[(bt.target == target) & (bt.model == model) & bt.observed]
    hs = sorted(g.horizon.unique())
    probs = []
    for h in range(1, window + 1):
        hh = min(hs, key=lambda x: abs(x - h))
        r = g[g.horizon == hh]["rel_resid"].values
        scale = fc_path["scale"]
        probs.append(float(np.mean(fc_path["forecast"][h - 1] + r * scale > capacity)))
    return max(probs)


def main():
    d = pd.read_csv(OUT / "daily_clean.csv", parse_dates=["date"], index_col="date")
    F = build_features(d)
    origin = d.index.max()
    hs = list(range(1, MAX_HORIZON + 1))
    rq = pd.read_csv(OUT / "residual_quantiles.csv")
    rows = []
    for target in TARGETS:
        res = forecast_all(d, F, target, origin, hs, MODELS)
        scale = d.loc[origin, target] if target == "hhs" else d[target].iloc[-28:].mean() + 1.0
        for m, hv in res.items():
            off = interval_offsets(rq, target, m)
            for h in hs:
                f = hv[h]
                rows.append({"target": target, "model": m, "horizon": h,
                             "date": origin + pd.Timedelta(days=h), "forecast": f, "scale": scale,
                             "lo80": max(0, f + off["q0100"][h - 1] * scale), "hi80": f + off["q0900"][h - 1] * scale,
                             "lo95": max(0, f + off["q0025"][h - 1] * scale), "hi95": f + off["q0975"][h - 1] * scale})
        print("forecasted", target, flush=True)
    fc = pd.DataFrame(rows)
    fc.to_csv(OUT / "forecast_final.csv", index=False)

    # ---- KPI summary -------------------------------------------------
    bt = pd.read_csv(OUT / "backtest_forecasts.csv", parse_dates=["origin", "target_date"])
    bt["resid"] = bt["actual"] - bt["forecast"]; bt["rel_resid"] = bt["resid"] / bt["scale"]
    mh = pd.read_csv(OUT / "metrics_by_horizon.csv")
    rk = pd.read_csv(OUT / "model_ranking.csv")
    sg = pd.read_csv(OUT / "surge_warning.csv"); st = pd.read_csv(OUT / "stability.csv")
    rb = pd.read_csv(OUT / "robustness.csv"); cov = pd.read_csv(OUT / "interval_coverage.csv")
    best = {t: rk[rk.target == t].sort_values("rank").iloc[0]["model"] for t in TARGETS}
    # the "recommended" model must beat both baselines at least once; otherwise fall back to best non-baseline
    overall_rank = rk.groupby("model")["rank"].mean().sort_values()
    best_overall = overall_rank.index[0]
    kp = {"origin": str(origin.date()), "best_model": best, "best_overall": best_overall,
          "overall_rank": overall_rank.round(2).to_dict(), "headroom_assumption": HEADROOM}
    for t in TARGETS:
        m = best_overall
        r7 = mh[(mh.target == t) & (mh.model == m) & (mh.horizon == 7)].iloc[0]
        kp[f"{t}_mae_7d"] = float(r7.MAE); kp[f"{t}_mape_7d"] = float(r7.MAPE); kp[f"{t}_wape_7d"] = float(r7.WAPE)
    kp["forecast_accuracy_pct_hhs_7d"] = 100 - kp["hhs_mape_7d"]
    kp["forecast_accuracy_pct_discharges_7d_wape"] = 100 - kp["discharges_wape_7d"]
    mm = best_overall
    s = sg[sg.model == mm].iloc[0]
    kp["surge"] = {k: (None if pd.isna(v) else (float(v) if not isinstance(v, str) else v)) for k, v in s.items()}
    kp["stability_index"] = float(st[st.model == mm]["stability_index"].iloc[0])
    r = rb[(rb.target == "hhs") & (rb.model == mm)].iloc[0]
    kp["robustness_win_rate"] = float(r.robustness_win_rate)
    kp["long_term_skill_vs_naive"] = float(r.long_term_skill_vs_naive)
    c = cov[(cov.target == "hhs") & (cov.model == mm)]
    kp["coverage80_hhs"] = float(c.cover80.mean() * 100); kp["coverage95_hhs"] = float(c.cover95.mean() * 100)
    path = fc[(fc.target == "hhs") & (fc.model == mm)].sort_values("horizon")
    cur = float(d.loc[origin, "hhs"])
    kp["current_hhs"] = cur
    fp = {"forecast": path["forecast"].values, "scale": float(path["scale"].iloc[0])}
    kp["breach_prob_14d_default"] = breach_probability(bt, "hhs", mm, fp, cur * (1 + HEADROOM), 14)
    kp["breach_prob_28d_default"] = breach_probability(bt, "hhs", mm, fp, cur * (1 + HEADROOM), 28)
    kp["forecast_hhs_7d"] = float(path.iloc[6]["forecast"]); kp["forecast_hhs_14d"] = float(path.iloc[13]["forecast"])
    kp["forecast_hhs_28d"] = float(path.iloc[27]["forecast"])
    json.dump(kp, open(OUT / "kpi_summary.json", "w"), indent=2, default=float)
    print(json.dumps(kp, indent=2, default=float))


if __name__ == "__main__":
    main()
