"""UAC Care-Load Forecasting - Streamlit dashboard (forecast-first layout)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

OUT = Path(__file__).resolve().parent / "outputs"
st.set_page_config(page_title="UAC Care-Load Forecast Desk", page_icon="📈", layout="wide")

NAVY, TEAL, AMBER, RED, INDIGO, SLATE = "#0B1F3A", "#0F9D8A", "#F59E0B", "#E11D48", "#4F46E5", "#64748B"
MODEL_COLORS = {"Naive": "#94A3B8", "Moving Avg": "#A78BFA", "ARIMA": "#0EA5E9", "SARIMA": "#0369A1",
                "Exp. Smoothing": "#14B8A6", "Random Forest": "#F97316", "Gradient Boosting": "#E11D48"}
FAMILY = {"Naive": "Baseline", "Moving Avg": "Baseline", "ARIMA": "Statistical", "SARIMA": "Statistical",
          "Exp. Smoothing": "Statistical", "Random Forest": "Machine learning", "Gradient Boosting": "Machine learning"}
TLABEL = {"hhs": "Children in HHS care", "discharges": "Discharges to sponsors", "transfers": "Transfers into HHS"}

st.markdown(f"""
<style>
.block-container {{padding-top: 1.2rem; max-width: 1400px;}}
.hero {{border-radius: 16px; padding: 22px 28px; color: white; margin-bottom: 14px;}}
.hero h2 {{margin: 0 0 4px 0; font-size: 1.55rem; color: white;}}
.hero p {{margin: 0; opacity: .92; font-size: 1.02rem;}}
.kpi {{background: #fff; border: 1px solid #E2E8F0; border-top: 4px solid {TEAL}; border-radius: 12px;
       padding: 12px 14px; height: 128px;}}
.kpi .l {{font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; color: {SLATE}; font-weight: 600;}}
.kpi .v {{font-size: 1.9rem; font-weight: 700; color: {NAVY}; line-height: 1.25;}}
.kpi .s {{font-size: .78rem; color: {SLATE};}}
.note {{background:#F1F5F9; border-radius:10px; padding:10px 14px; font-size:.88rem; color:#334155;}}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    r = lambda f, **k: pd.read_csv(OUT / f, **k)
    d = dict(hist=r("daily_clean.csv", parse_dates=["date"], index_col="date"),
             fc=r("forecast_final.csv", parse_dates=["date"]),
             bt=r("backtest_forecasts.csv", parse_dates=["origin", "target_date"]),
             mh=r("metrics_by_horizon.csv"), mo=r("metrics_overall.csv"), rk=r("model_ranking.csv"),
             cov=r("interval_coverage.csv"), sg=r("surge_warning.csv"), st=r("stability.csv"),
             rb=r("robustness.csv"), kp=json.load(open(OUT / "kpi_summary.json")),
             ds=json.load(open(OUT / "data_summary.json")))
    d["bt"]["resid"] = d["bt"]["actual"] - d["bt"]["forecast"]
    d["bt"]["rel_resid"] = d["bt"]["resid"] / d["bt"]["scale"]
    return d


D = load()
hist, fc, bt, kp = D["hist"], D["fc"], D["bt"], D["kp"]
origin = pd.Timestamp(kp["origin"])
cur = float(hist.loc[origin, "hhs"])


def flow_fit():
    """Weekly change in care load regressed on weekly net flow (transfers - discharges), last 365 days."""
    x = hist.iloc[-365:]
    dh, ns = x.hhs.diff(7), (x.transfers - x.discharges).rolling(7).sum()
    ok = dh.notna() & ns.notna()
    b, a = np.polyfit(ns[ok], dh[ok], 1)
    return float(np.clip(b, 0.5, 1.5)), float(a) / 7      # beta, load growth per day not explained by net flow


BETA, OTHER = flow_fit()


def path(target, model, H=28):
    return fc[(fc.target == target) & (fc.model == model) & (fc.horizon <= H)].sort_values("horizon")


def breach_curve(model, capacity, H):
    """P(care load > capacity) at each day of the horizon, from the empirical backtest error distribution."""
    p = path("hhs", model, H)
    g = bt[(bt.target == "hhs") & (bt.model == model) & bt.observed]
    hs = np.array(sorted(g.horizon.unique()))
    out = []
    for h, f, s in zip(p.horizon, p.forecast, p.scale):
        hh = hs[np.argmin(np.abs(hs - h))]
        r = g[g.horizon == hh]["rel_resid"].values
        out.append(float(np.mean(f + r * s > capacity)))
    return p["date"].values, np.array(out)


def kpi(col, label, value, sub, color=TEAL):
    col.markdown(f"<div class='kpi' style='border-top-color:{color}'><div class='l'>{label}</div>"
                 f"<div class='v'>{value}</div><div class='s'>{sub}</div></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ control rail
st.sidebar.markdown("### Control rail")
H = st.sidebar.select_slider("Forecast horizon (days)", options=[7, 14, 21, 28], value=14)
models = list(FAMILY)
best = kp["best_overall"]
model = st.sidebar.selectbox("Primary model", models, index=models.index(best),
                             help="Defaults to the model with the best average rank across all three targets in the walk-forward backtest.")
overlay = st.sidebar.multiselect("Overlay other models", [m for m in models if m != model], default=[])
band = st.sidebar.radio("Uncertainty band", ["80%", "95%"], horizontal=True)
head = st.sidebar.slider("Capacity headroom over today's load (%)", 0, 30, 5)
capacity = cur * (1 + head / 100)
st.sidebar.caption(f"Capacity threshold = **{capacity:,.0f}** children (today's load {cur:,.0f}).")
st.sidebar.caption(f"Forecast origin: {origin:%d %b %Y} (latest reported day).")
lo_c, hi_c = ("lo80", "hi80") if band == "80%" else ("lo95", "hi95")

# ------------------------------------------------------------------ hero + KPIs
ph = path("hhs", model, H)
pd_ = path("discharges", model, H)
pt_ = path("transfers", model, H)
dates, pb = breach_curve(model, capacity, H)
pmax = float(pb.max())
status, color = (("STABLE", TEAL) if pmax < 0.2 else ("WATCH", AMBER) if pmax < 0.5 else ("ALERT", RED))
end = ph.iloc[-1]
net = float(pt_.forecast.sum() - pd_.forecast.sum())
st.markdown(f"""<div class='hero' style='background: linear-gradient(120deg, {NAVY}, {color}CC);'>
<h2>Capacity outlook: {status}</h2>
<p>In {H} days the care load is projected at <b>{end.forecast:,.0f}</b> children
({band} range {end[lo_c]:,.0f} to {end[hi_c]:,.0f}) versus {cur:,.0f} today.
Chance of exceeding {capacity:,.0f} at some point: <b>{pmax*100:.0f}%</b>.
Projected transfers in minus discharges out: <b>{net:+,.0f}</b> children over the window.</p></div>""",
            unsafe_allow_html=True)

sg = D["sg"][D["sg"].model == model].iloc[0]
stab = float(D["st"][D["st"].model == model]["stability_index"].iloc[0])
rob = D["rb"][(D["rb"].target == "hhs") & (D["rb"].model == model)].iloc[0]
m7 = D["mh"][(D["mh"].target == "hhs") & (D["mh"].model == model) & (D["mh"].horizon == 7)].iloc[0]
c = st.columns(5)
kpi(c[0], "Forecast accuracy", f"{100 - m7.MAPE:.1f}%", "care load, 7-day horizon (100 - MAPE)")
lead = "n/a" if pd.isna(sg.avg_lead_days) else f"{sg.avg_lead_days:.0f} days"
kpi(c[1], "Surge lead time", lead, f"avg warning; caught {int(sg.hits)} of {int(sg.events)} stress events", INDIGO)
kpi(c[2], "Capacity breach probability", f"{pmax*100:.0f}%", f"within {H} days at {capacity:,.0f}", color)
kpi(c[3], "Forecast stability index", f"{stab:.1f}", "100 = forecasts barely change week to week", TEAL)
kpi(c[4], "Model robustness", f"{rob.robustness_win_rate:.0f}%", "of backtest folds beating naive persistence", INDIGO)
st.write("")

tabs = st.tabs(["Care load forecast", "Discharge demand", "Model lab", "Confidence intervals", "Scenario comparison"])


def band_trace(fig, p, name, col, rgba):
    fig.add_trace(go.Scatter(x=list(p.date) + list(p.date[::-1]), y=list(p[hi_c]) + list(p[lo_c][::-1]),
                             fill="toself", fillcolor=rgba, line=dict(width=0), hoverinfo="skip",
                             name=f"{name} {band} band", showlegend=True))


def add_origin(fig):
    fig.add_shape(type="line", x0=origin, x1=origin, y0=0, y1=1, yref="paper",
                  line=dict(color=SLATE, dash="dot", width=1))
    fig.add_annotation(x=origin, y=1, yref="paper", text="forecast origin", showarrow=False,
                       xanchor="right", font=dict(size=10, color=SLATE))


# ================================================================== 1. care load
with tabs[0]:
    look = st.slider("History shown (days)", 30, 365, 120, key="lk1")
    h_ = hist.loc[origin - pd.Timedelta(days=look):origin]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h_.index, y=h_.hhs, name="Reported care load", line=dict(color=NAVY, width=2.4)))
    band_trace(fig, ph, model, MODEL_COLORS[model], "rgba(15,157,138,.20)")
    fig.add_trace(go.Scatter(x=ph.date, y=ph.forecast, name=f"{model} forecast",
                             line=dict(color=MODEL_COLORS[model], width=3)))
    for m in overlay:
        o = path("hhs", m, H)
        fig.add_trace(go.Scatter(x=o.date, y=o.forecast, name=m, line=dict(color=MODEL_COLORS[m], width=1.6, dash="dash")))
    fig.add_hline(y=capacity, line=dict(color=RED, dash="dash"), annotation_text=f"capacity {capacity:,.0f}",
                  annotation_position="top left")
    add_origin(fig)
    fig.update_layout(template="plotly_white", height=470, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Children in HHS care", hovermode="x unified", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    ms = path("hhs", model, 28)
    tab = ms[ms.horizon.isin([7, 14, 21, 28])][["horizon", "date", "forecast", lo_c, hi_c]].copy()
    tab["change vs today"] = tab.forecast - cur
    tab.columns = ["Days ahead", "Date", "Forecast", f"Low ({band})", f"High ({band})", "Change vs today"]
    tab["Date"] = tab["Date"].dt.strftime("%d %b %Y")
    st.dataframe(tab.round(0), hide_index=True, width="stretch")
    st.download_button("Download forecast (CSV)", path("hhs", model, 28).to_csv(index=False),
                       file_name=f"care_load_forecast_{model.replace(' ', '_').lower()}.csv")

# ================================================================== 2. discharge demand
with tabs[1]:
    look = st.slider("History shown (days)", 30, 365, 90, key="lk2")
    h_ = hist.loc[origin - pd.Timedelta(days=look):origin]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h_.index, y=h_.discharges.rolling(7).mean(), name="Discharges (7-day avg)",
                             line=dict(color=TEAL, width=2.4)))
    fig.add_trace(go.Scatter(x=h_.index, y=h_.transfers.rolling(7).mean(), name="Transfers in (7-day avg)",
                             line=dict(color=AMBER, width=2.4)))
    band_trace(fig, pd_, "Discharge (single-day)", TEAL, "rgba(15,157,138,.18)")
    fig.add_trace(go.Scatter(x=pd_.date, y=pd_.forecast.rolling(7, min_periods=1).mean(),
                             name="Discharge forecast (7-day avg)", line=dict(color=TEAL, width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=pt_.date, y=pt_.forecast.rolling(7, min_periods=1).mean(),
                             name="Transfers forecast (7-day avg)", line=dict(color=AMBER, width=3, dash="dash")))
    add_origin(fig)
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_title="Children per day", hovermode="x unified", legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, width="stretch")
    wk = []
    for w in range(1, H // 7 + 1):
        a, b = (w - 1) * 7, w * 7
        t_in, d_out = pt_.forecast.iloc[a:b].sum(), pd_.forecast.iloc[a:b].sum()
        wk.append({"Week": f"Week {w}", "Transfers in": round(t_in), "Discharges out": round(d_out),
                   "Net (in - out)": round(t_in - d_out),
                   "Discharges / transfers": f"{d_out / max(t_in, 1) * 100:.0f}%"})
    cA, cB = st.columns([3, 2])
    cA.dataframe(pd.DataFrame(wk), hide_index=True, width="stretch")
    t_sum, d_sum = pt_.forecast.sum(), pd_.forecast.sum()
    extra = OTHER / BETA * H                       # inflow the transfers column does not capture
    ratio = d_sum / max(t_sum + extra, 1)
    verdict = ("Discharge capacity is projected to <b>offset</b> total inflow." if ratio >= 1 else
               "Discharges are projected to <b>fall short</b> of total inflow, so the care load keeps building.")
    cB.markdown(f"<div class='note'>Over {H} days discharges are projected at <b>{d_sum:,.0f}</b> versus "
                f"<b>{t_sum:,.0f}</b> transfers in. Past data also show about <b>{OTHER/BETA:.1f}</b> children/day entering care "
                f"outside the transfers column (+{extra:,.0f} over the window), so coverage of total inflow is "
                f"<b>{ratio*100:.0f}%</b>. {verdict}</div>", unsafe_allow_html=True)

# ================================================================== 3. model lab
with tabs[2]:
    cA, cB = st.columns(2)
    tg = cA.selectbox("Target", list(TLABEL), format_func=TLABEL.get)
    mt = cB.selectbox("Error metric", ["MAE", "RMSE", "MAPE", "WAPE"],
                      help="WAPE = total absolute error / total actual; steadier than MAPE for small daily flows.")
    mh = D["mh"][D["mh"].target == tg]
    fig = go.Figure()
    for m in models:
        g = mh[mh.model == m].sort_values("horizon")
        fig.add_trace(go.Scatter(x=g.horizon, y=g[mt], name=m, mode="lines+markers",
                                 line=dict(color=MODEL_COLORS[m], width=3 if m == model else 1.6)))
    fig.update_layout(template="plotly_white", height=380, xaxis_title="Forecast horizon (days)",
                      yaxis_title=mt, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    lb = D["mo"][D["mo"].target == tg].merge(D["rk"][D["rk"].target == tg][["model", "rank"]], on="model")
    lb["Family"] = lb.model.map(FAMILY)
    lb = lb.sort_values("rank")[["model", "Family", "MAE", "RMSE", "MAPE", "WAPE", "rank"]]
    lb.columns = ["Model", "Family", "MAE", "RMSE", "MAPE %", "WAPE %", "Avg rank (1 = best)"]
    st.dataframe(lb.round(2), hide_index=True, width="stretch")
    fam = mh.assign(Family=mh.model.map(FAMILY)).groupby(["Family", "horizon"])["MAE"].mean().unstack(0)
    short = fam.loc[[h for h in fam.index if h <= 7]].mean()
    long_ = fam.loc[[h for h in fam.index if h >= 14]].mean()
    st.markdown("<div class='note'><b>Statistical vs machine learning (mean MAE):</b> "
                + " | ".join(f"{k}: {short[k]:,.1f} (1-7d), {long_[k]:,.1f} (14-28d)" for k in fam.columns) + "</div>",
                unsafe_allow_html=True)
    hz = st.select_slider("Backtest horizon to inspect (days)", [1, 3, 7, 14, 21, 28], value=7)
    g = bt[(bt.target == tg) & (bt.model == model) & (bt.horizon == hz) & bt.observed].sort_values("target_date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.target_date, y=g.actual, name="Actual", line=dict(color=NAVY, width=2.4)))
    fig.add_trace(go.Scatter(x=g.target_date, y=g.forecast, name=f"{model} ({hz}-day-ahead)",
                             line=dict(color=MODEL_COLORS[model], width=2.4, dash="dash")))
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_title=TLABEL[tg], legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")

# ================================================================== 4. confidence intervals
with tabs[3]:
    tg = st.selectbox("Target", list(TLABEL), format_func=TLABEL.get, key="tg4")
    p = path(tg, model, 28)
    h_ = hist.loc[origin - pd.Timedelta(days=60):origin]
    fig = go.Figure()
    for lo, hi, a, nm in (("lo95", "hi95", ".12", "95% interval"), ("lo80", "hi80", ".26", "80% interval")):
        fig.add_trace(go.Scatter(x=list(p.date) + list(p.date[::-1]), y=list(p[hi]) + list(p[lo][::-1]), fill="toself",
                                 fillcolor=f"rgba(79,70,229,{a})", line=dict(width=0), name=nm, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=h_.index, y=h_[tg], name="Reported", line=dict(color=NAVY, width=2)))
    fig.add_trace(go.Scatter(x=p.date, y=p.forecast, name="Forecast", line=dict(color=INDIGO, width=3)))
    if tg == "hhs":
        fig.add_hline(y=capacity, line=dict(color=RED, dash="dash"), annotation_text="capacity")
    add_origin(fig)
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_title=TLABEL[tg], legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    cA, cB = st.columns(2)
    cv = D["cov"][(D["cov"].target == tg) & (D["cov"].model == model)].sort_values("horizon")
    cv = cv.assign(**{"Achieved 80%": (cv.cover80 * 100).round(0), "Achieved 95%": (cv.cover95 * 100).round(0)})
    cA.markdown("**Does the band hold? Out-of-sample coverage by horizon** (nominal 80% / 95%)")
    cA.dataframe(cv[["horizon", "Achieved 80%", "Achieved 95%", "n"]].rename(columns={"horizon": "Days ahead"}),
                 hide_index=True, width="stretch")
    if tg == "hhs":
        d2, pb2 = breach_curve(model, capacity, 28)
        f2 = go.Figure(go.Scatter(x=d2, y=pb2 * 100, fill="tozeroy", line=dict(color=RED, width=2.5)))
        f2.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=40, b=10),
                         yaxis_title="P(load > capacity), %", title=f"Breach probability by day (capacity {capacity:,.0f})")
        cB.plotly_chart(f2, width="stretch")
    else:
        w = D["cov"][(D["cov"].target == tg) & (D["cov"].model == model)].sort_values("horizon")
        f2 = go.Figure(go.Bar(x=w.horizon, y=w.width80, marker_color=INDIGO))
        f2.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=40, b=10),
                         title="Average 80% interval width (children/day)", xaxis_title="Days ahead")
        cB.plotly_chart(f2, width="stretch")

# ================================================================== 5. scenarios
with tabs[4]:
    beta = BETA
    st.markdown(f"<div class='note'>What-if engine: change in care load per child of net flow = <b>{beta:.2f}</b> "
                "(fitted on the last 365 days, weekly changes). Scenario deltas = this factor x cumulative change in "
                "(transfers in - discharges out) applied to the primary model's forecast. It is a planning aid, not a causal model.</div>",
                unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    up = c1.slider("Custom scenario: change in transfers into HHS (%)", -50, 300, 150, step=10)
    dn = c2.slider("Custom scenario: change in discharges (%)", -75, 100, -25, step=5)
    scen = {"Baseline": (0, 0), "Intake surge (+100% transfers)": (100, 0), "Faster discharges (+50%)": (0, 50),
            "Surge + slower discharges (+100% / -25%)": (100, -25), "Custom": (up, dn)}
    cols = [NAVY, AMBER, TEAL, RED, INDIGO]
    base = ph.set_index("horizon")
    T, Dd = pt_.set_index("horizon").forecast, pd_.set_index("horizon").forecast
    fig = go.Figure()
    rows = []
    for (nm, (pu, pdn)), colr in zip(scen.items(), cols):
        delta = beta * ((pu / 100) * T - (pdn / 100) * Dd).cumsum()
        yv = base.forecast + delta
        fig.add_trace(go.Scatter(x=base.date, y=yv, name=nm, line=dict(color=colr, width=3 if nm == "Baseline" else 2.2)))
        over = base.date[yv > capacity]
        rows.append({"Scenario": nm, f"Load in {H} days": round(yv.iloc[-1]),
                     "vs baseline": round(yv.iloc[-1] - base.forecast.iloc[-1]),
                     "First day above capacity": over.iloc[0].strftime("%d %b") if len(over) else "not within window"})
    fig.add_hline(y=capacity, line=dict(color=RED, dash="dash"), annotation_text=f"capacity {capacity:,.0f}")
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_title="Children in HHS care", hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
