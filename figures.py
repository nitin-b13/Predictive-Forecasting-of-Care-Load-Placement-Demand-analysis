"""Module 6 - Static figures for the research paper and README."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import *

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titleweight": "bold", "axes.titlesize": 10})
NAVY, TEAL, AMBER, RED, INDIGO = "#0B1F3A", "#0F9D8A", "#F59E0B", "#E11D48", "#4F46E5"
MC = {"Naive": "#94A3B8", "Moving Avg": "#A78BFA", "ARIMA": "#0EA5E9", "SARIMA": "#0369A1",
      "Exp. Smoothing": "#14B8A6", "Random Forest": "#F97316", "Gradient Boosting": "#E11D48"}
FAM = {"Naive": "Baseline", "Moving Avg": "Baseline", "ARIMA": "Statistical", "SARIMA": "Statistical",
       "Exp. Smoothing": "Statistical", "Random Forest": "Machine learning", "Gradient Boosting": "Machine learning"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def main():
    d = pd.read_csv(OUT / "daily_clean.csv", parse_dates=["date"], index_col="date")
    dec = pd.read_csv(OUT / "decomposition.csv", parse_dates=["date"], index_col="date")
    fc = pd.read_csv(OUT / "forecast_final.csv", parse_dates=["date"])
    mh = pd.read_csv(OUT / "metrics_by_horizon.csv")
    bt = pd.read_csv(OUT / "backtest_forecasts.csv", parse_dates=["origin", "target_date"])
    cov = pd.read_csv(OUT / "interval_coverage.csv")
    kp = json.load(open(OUT / "kpi_summary.json"))
    best = kp["best_overall"]
    obs = d[d.is_observed]

    # 1 care load history
    f, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(obs.index, obs.hhs, color=NAVY, lw=1.4)
    pk, mn = obs.hhs.idxmax(), obs.hhs.idxmin()
    ax.annotate(f"peak {obs.hhs.max():,.0f}\n{pk:%d %b %Y}", (pk, obs.hhs.max()), xytext=(15, -5),
                textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="-", color="grey"))
    ax.annotate(f"low {obs.hhs.min():,.0f}\n{mn:%d %b %Y}", (mn, obs.hhs.min()), xytext=(-70, 25),
                textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="-", color="grey"))
    ax.set_title("Children in HHS care, Jan 2023 - Dec 2025"); ax.set_ylabel("Children")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    save(f, "fig1_care_load_history.png")

    # 2 flows
    f, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True)
    a1.plot(d.index, d.transfers.rolling(7).mean(), color=AMBER, lw=1.5, label="Transfers into HHS")
    a1.plot(d.index, d.discharges.rolling(7).mean(), color=TEAL, lw=1.5, label="Discharges")
    a1.legend(frameon=False); a1.set_title("Daily flows (7-day average)")
    net = (d.transfers - d.discharges).rolling(7).mean()
    a2.fill_between(d.index, net, 0, where=net >= 0, color=RED, alpha=.35, label="Net inflow (pressure)")
    a2.fill_between(d.index, net, 0, where=net < 0, color=TEAL, alpha=.35, label="Net outflow (relief)")
    a2.legend(frameon=False); a2.set_title("Net pressure = transfers - discharges (7-day average)")
    save(f, "fig2_flows_net_pressure.png")

    # 3 decomposition
    f, ax = plt.subplots(4, 1, figsize=(8, 6.2), sharex=True)
    for a, c, t in zip(ax, ["observed", "trend", "seasonal", "resid"],
                       ["Observed", "Trend", "Weekly seasonal", "Residual"]):
        a.plot(dec.index, dec[f"hhs_{c}"], color=NAVY if c != "resid" else SLATE if False else "#64748B", lw=1)
        a.set_ylabel(t, fontsize=8)
    ax[0].set_title("STL decomposition of children in HHS care (period = 7 days)")
    save(f, "fig3_decomposition.png")

    # 4 weekday pattern
    f, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3))
    rec = obs.loc["2025-04-01":]
    w = rec.groupby(rec.index.dayofweek)[["transfers", "discharges"]].mean()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x = np.arange(7)
    a1.bar(w.index - .2, w.transfers, .4, color=AMBER, label="Transfers in")
    a1.bar(w.index + .2, w.discharges, .4, color=TEAL, label="Discharges")
    a1.set_xticks(x); a1.set_xticklabels(names); a1.legend(frameon=False)
    a1.set_title("Average by weekday, Apr-Dec 2025")
    miss = d[~d.is_observed]
    mc = miss.index.dayofweek.value_counts().reindex(range(7), fill_value=0)
    a2.bar(x, mc.values, color=INDIGO); a2.set_xticks(x); a2.set_xticklabels(names)
    a2.set_title("Days with no report, by weekday")
    save(f, "fig4_weekday_gaps.png")

    # 5 accuracy by horizon
    f, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for m in MC:
        g = mh[(mh.target == "hhs") & (mh.model == m)].sort_values("horizon")
        a1.plot(g.horizon, g.MAPE, marker="o", ms=3, color=MC[m], lw=2.2 if m == best else 1.2, label=m)
        g = mh[(mh.target == "discharges") & (mh.model == m)].sort_values("horizon")
        a2.plot(g.horizon, g.WAPE, marker="o", ms=3, color=MC[m], lw=2.2 if m == best else 1.2)
    a1.set_title("Care load: MAPE (%) by horizon"); a2.set_title("Discharges: WAPE (%) by horizon")
    for a in (a1, a2): a.set_xlabel("Days ahead")
    a1.legend(frameon=False, fontsize=7)
    save(f, "fig5_error_by_horizon.png")

    # 6 forecast fan
    f, ax = plt.subplots(3, 1, figsize=(8, 8))
    for a, t, ttl in zip(ax, TARGETS, ["Children in HHS care", "Discharges per day", "Transfers into HHS per day"]):
        p = fc[(fc.target == t) & (fc.model == best)].sort_values("horizon")
        h = d.loc[d.index.max() - pd.Timedelta(days=60):, t]
        if t != "hhs":
            h = h.rolling(7).mean(); pf = p.forecast.rolling(7, min_periods=1).mean()
        else:
            pf = p.forecast
        a.fill_between(p.date, p.lo95, p.hi95, color=INDIGO, alpha=.12, label="95% interval")
        a.fill_between(p.date, p.lo80, p.hi80, color=INDIGO, alpha=.25, label="80% interval")
        a.plot(h.index, h, color=NAVY, lw=1.8, label="Reported" + (" (7-day avg)" if t != "hhs" else ""))
        a.plot(p.date, pf, color=INDIGO, lw=2.2, label=f"{best} forecast" + (" (7-day avg)" if t != "hhs" else ""))
        a.set_title(ttl); a.legend(frameon=False, fontsize=7, ncol=2)
        a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    save(f, "fig6_forecast_fan.png")

    # 7 backtest 7-day ahead
    f, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.4))
    for a, t, ttl in ((a1, "hhs", "Care load: 7-day-ahead forecast vs actual"),
                      (a2, "discharges", "Discharges: 7-day-ahead forecast vs actual")):
        g = bt[(bt.target == t) & (bt.model == best) & (bt.horizon == 7) & bt.observed].sort_values("target_date")
        a.plot(g.target_date, g.actual, color=NAVY, lw=1.8, label="Actual")
        a.plot(g.target_date, g.forecast, color=MC[best], lw=1.8, ls="--", label=f"{best}")
        a.set_title(ttl); a.legend(frameon=False)
    save(f, "fig7_backtest_7day.png")

    # 8 coverage
    f, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3))
    for a, t, ttl in ((a1, "hhs", "Care load"), (a2, "discharges", "Discharges")):
        g = cov[(cov.target == t) & (cov.model == best)].sort_values("horizon")
        a.plot(g.horizon, g.cover80 * 100, marker="o", color=INDIGO, label="achieved (80% band)")
        a.plot(g.horizon, g.cover95 * 100, marker="o", color=TEAL, label="achieved (95% band)")
        a.axhline(80, color=INDIGO, ls=":", lw=1); a.axhline(95, color=TEAL, ls=":", lw=1)
        a.set_ylim(40, 105); a.set_title(f"{ttl}: out-of-sample coverage"); a.set_xlabel("Days ahead")
    a1.legend(frameon=False, fontsize=7)
    save(f, "fig8_interval_coverage.png")

    # 9 statistical vs ML
    f, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3))
    FC = {"Baseline": "#94A3B8", "Statistical": "#0369A1", "Machine learning": "#F97316"}
    for a, t, met, ttl in ((a1, "hhs", "MAPE", "Care load (MAPE %)"), (a2, "discharges", "WAPE", "Discharges (WAPE %)")):
        g = mh[mh.target == t].assign(fam=lambda x: x.model.map(FAM)).groupby(["fam", "horizon"])[met].mean().unstack(0)
        for c in g.columns:
            a.plot(g.index, g[c], marker="o", ms=3, color=FC[c], label=c)
        a.set_title(ttl); a.set_xlabel("Days ahead")
    a1.legend(frameon=False, fontsize=7)
    save(f, "fig9_family_comparison.png")
    print("figures done")


if __name__ == "__main__":
    main()
