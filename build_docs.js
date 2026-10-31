const fs = require("fs");
const path = require("path");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, HeadingLevel, AlignmentType,
  WidthType, ShadingType, BorderStyle, Footer, PageNumber, LevelFormat, PageBreak } = D;
const N = JSON.parse(fs.readFileSync(path.join(__dirname, "numbers.json")));
const FIG = path.join(__dirname, "..", "outputs", "figures");
const NAVY = "0B1F3A", TEAL = "0F9D8A", GREY = "64748B";
const FONT = "Arial";
const kp = N.kp, ds = N.ds;
const fmt = (x, d = 0) => Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

const p = (text, o = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, ...o.para,
  children: (Array.isArray(text) ? text : [text]).map(t => typeof t === "string" ? new TextRun({ text: t, font: FONT, size: o.size || 21, color: o.color, bold: o.bold, italics: o.italics }) : t) });
const b = (text) => new TextRun({ text, bold: true, font: FONT, size: 21 });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 120 }, children: [new TextRun({ text: t, font: FONT, size: 30, bold: true, color: NAVY })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 80 }, children: [new TextRun({ text: t, font: FONT, size: 24, bold: true, color: TEAL })] });
const bullet = (text) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 70, line: 264 },
  children: (Array.isArray(text) ? text : [text]).map(t => typeof t === "string" ? new TextRun({ text: t, font: FONT, size: 21 }) : t) });
const caption = (t) => new Paragraph({ spacing: { after: 200 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: t, font: FONT, size: 17, italics: true, color: GREY })] });

function pngSize(f) { const b = fs.readFileSync(f); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function fig(name, widthIn = 6.2) {
  const f = path.join(FIG, name); const [w, h] = pngSize(f);
  const W = Math.round(widthIn * 96);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(f), transformation: { width: W, height: Math.round(W * h / w) },
      altText: { title: name, description: name, name } })] });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "CBD5E1" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, c) => a + c, 0);
  const cell = (t, w, head, i, hl) => new TableCell({ width: { size: w, type: WidthType.DXA }, borders,
    shading: { fill: head ? NAVY : hl ? "E6F6F3" : "FFFFFF", type: ShadingType.CLEAR, color: "auto" },
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ alignment: i === 0 || opts.left ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: String(t), font: FONT, size: opts.size || 17, bold: head || hl, color: head ? "FFFFFF" : undefined })] })] });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], true, i)) }),
      ...rows.map(r => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, widths[i], false, i, opts.highlight && opts.highlight(r))) }))] });
}
const gap = () => new Paragraph({ spacing: { after: 100 }, children: [] });

const styles = { default: { document: { run: { font: FONT, size: 21 } } },
  paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, font: FONT, color: TEAL }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } }] };
const numbering = { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] };
const page = { page: { size: { width: 12240, height: 15840 }, margin: { top: 1260, right: 1260, bottom: 1200, left: 1260 } } };
const footer = (label) => ({ default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: label + "  |  Page ", font: FONT, size: 16, color: GREY }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: GREY })] })] }) });

const best = kp.best_overall;
const s28 = N.sums["28"], s14 = N.sums["14"];
const extra28 = N.flow.intercept_per_day / N.flow.beta * 28;
const hdrH = ["Model", "Family", "1d", "3d", "7d", "14d", "21d", "28d"];
const wH = [2000, 1500, 840, 840, 840, 840, 840, 840];

// =============================================================== RESEARCH PAPER
const paper = [];
paper.push(new Paragraph({ spacing: { before: 600, after: 100 }, children: [new TextRun({ text: "From Reporting to Foresight", font: FONT, size: 44, bold: true, color: NAVY })] }));
paper.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Forecasting care load, discharge demand and capacity stress in the HHS Unaccompanied Children Program", font: FONT, size: 26, color: TEAL })] }));
paper.push(p(`Research paper  |  Data: daily program reports, ${ds.start} to ${ds.end}  |  Forecast origin: ${kp.origin}`, { color: GREY, size: 18 }));

paper.push(h1("Abstract"));
paper.push(p(`The Unaccompanied Children (UAC) Program reports daily counts of children in Customs and Border Protection (CBP) custody and in HHS care, but offers no forward view of how many children will need care, whether discharges will keep pace with transfers, or when capacity will be stressed. This study converts the daily series into a forecasting system. We prepared a continuous daily series, engineered lag, rolling, flow and calendar features, and compared seven forecasting approaches (two baselines, three statistical models and two machine-learning models) using strict walk-forward validation across 52 weekly forecast origins and six horizons from 1 to 28 days.`));
paper.push(p([b("Main findings. "), `Exponential smoothing was the most reliable model overall. For children in HHS care it reached a mean absolute percentage error (MAPE) of ${N.hhs_mape[4][4]}% at 7 days and ${N.hhs_mape[4][5]}% at 14 days, and it reduced error by about ${fmt(kp.long_term_skill_vs_naive)}% against naive persistence at 14 to 28 days. Machine-learning models did not outperform simpler statistical models on care load, mainly because the series contains a structural drop in early 2025 that tree models cannot extrapolate. Daily discharges and transfers are far noisier: the best models miss by roughly 38% (discharges) and 64% (transfers) of the average daily volume. Prediction intervals built from backtest errors were close to their nominal 80% level for care load (${fmt(kp.coverage80_hhs)}% achieved) but under-covered for the flow series. Early-warning of capacity stress was weak: no model flagged more than 43% of stress events.`]));
paper.push(p([b("Outlook. "), `On ${kp.origin} the program held ${fmt(kp.current_hhs)} children. The selected model projects ${fmt(kp.forecast_hhs_7d)} at 7 days, ${fmt(kp.forecast_hhs_14d)} at 14 days and ${fmt(kp.forecast_hhs_28d)} at 28 days, a gentle rise that follows a roughly 15% increase since 1 October. Against a capacity threshold of 5% above today's load, the probability of a breach is about ${fmt(kp.breach_prob_14d_default * 100)}% within 14 days and ${fmt(kp.breach_prob_28d_default * 100)}% within 28 days.`]));

paper.push(h1("1. Background and objectives"));
paper.push(p("The UAC Program operates in a high-uncertainty environment. Sudden changes in border activity, policy enforcement or humanitarian conditions can rapidly increase the number of children entering federal care. Descriptive analytics explain what has happened; planners need to know what is coming. Three questions drive this project:"));
paper.push(bullet("How many children will be under HHS care in the coming days or weeks?"));
paper.push(bullet("Will discharge capacity be sufficient to offset incoming transfers?"));
paper.push(bullet("When should shelters, medical staff and caseworkers be scaled up in advance?"));
paper.push(p("The primary objectives are to forecast the number of children in HHS care, to estimate the future imbalance between intake and exits, and to predict short-term discharge demand. Secondary objectives are to provide early warnings for healthcare planners, to quantify forecast uncertainty, and to compare statistical and machine-learning approaches."));

paper.push(h1("2. Data and exploratory analysis"));
paper.push(h2("2.1 Dataset"));
paper.push(p(`The file contains six columns: reporting date, children apprehended and placed in CBP custody (daily intake), children in CBP custody, children transferred out of CBP custody (the flow into HHS), children in HHS care (the active care load) and children discharged from HHS care (sponsor placements). The file has ${fmt(ds.raw_rows)} rows, of which ${ds.empty_rows_dropped} are empty trailing rows that were removed. That leaves ${ds.observed_days} reported days across ${fmt(ds.calendar_days)} calendar days (${ds.start} to ${ds.end}).`));
paper.push(h2("2.2 Reporting gaps"));
paper.push(p(`${ds.missing_days_interpolated} calendar days (${ds.missing_share_pct}%) have no report. The gaps are systematic: ${ds.missing_by_weekday.Friday} are Fridays and ${ds.missing_by_weekday.Saturday} are Saturdays, so the program effectively reports on a Sunday to Thursday rhythm with occasional Friday reports. The longest run without a report is ${ds.longest_gap_days} days. This matters for modelling. We interpolated linearly to obtain a continuous daily index, as the methodology requires, but flagged every interpolated day and excluded those days from all accuracy scoring (masking). Interpolated values are therefore used only to give models a continuous history and never to grade them.`));
paper.push(fig("fig4_weekday_gaps.png", 5.8));
paper.push(caption("Figure 1. Reporting gaps concentrate on Fridays and Saturdays; discharges peak on Thursdays and are lowest on Fridays and Sundays (Apr-Dec 2025)."));
paper.push(h2("2.3 Care load: a structural break"));
paper.push(p(`The care load fell from a peak of ${fmt(ds.peak_hhs)} children on 20 December 2023 to a low of ${fmt(ds.min_hhs)} on 21 August 2025. The yearly average dropped from ${fmt(N.levels.y2023)} (2023) to ${fmt(N.levels.y2024)} (2024) and ${fmt(N.levels.y2025)} (2025). Most of the fall was abrupt: from about 6,400 children on 1 January 2025 to roughly 2,300 by early March. Since the August low the load has risen steadily, from ${fmt(N.levels.oct1)} on 1 October to ${fmt(ds.latest_hhs)} on ${ds.end}.`));
paper.push(fig("fig1_care_load_history.png", 6.0));
paper.push(caption("Figure 2. Children in HHS care (reported days only)."));
paper.push(p("This break shapes every modelling choice. Models fitted on the full history would carry the memory of a very different operating regime, so statistical models use a recent fitting window (180 days for care load, 90 days for the flow series), and the walk-forward test deliberately includes the 2025 decline so that results reflect conditions that are hard, not just calm."));
paper.push(h2("2.4 Flows and net pressure"));
paper.push(p(`Transfers into HHS and discharges move together over long periods, and their difference (net pressure) explains the direction of the care load. Regressing the weekly change in care load on the weekly sum of transfers minus discharges over the last 365 days gives a slope of ${N.flow.beta.toFixed(2)} and a correlation of ${N.flow.corr.toFixed(2)}. The intercept implies about ${N.flow.intercept_per_day.toFixed(1)} children per day entering care through routes not captured by the transfers column. This relationship is used to build the dashboard's scenario view, and it is important when judging whether discharges "offset" inflow: transfers alone understate total inflow.`));
paper.push(fig("fig2_flows_net_pressure.png", 6.0));
paper.push(caption("Figure 3. Seven-day average flows and the net pressure indicator (transfers minus discharges)."));
paper.push(h2("2.5 Trend, seasonality and residuals"));
paper.push(p(`A robust STL decomposition with a seven-day period separates trend, weekly seasonality and residual. Weekly seasonality explains about ${fmt(ds.weekly_seasonal_strength_hhs * 100)}% of the non-trend variation in care load and about ${fmt(ds.weekly_seasonal_strength_discharges * 100)}% for discharges. The discharge figure is partly inflated by the Friday and Saturday gaps, so it should be read as an upper bound; the observed-day averages in Figure 1 are a more direct view of the weekday pattern.`));
paper.push(fig("fig3_decomposition.png", 5.6));
paper.push(caption("Figure 4. STL decomposition of children in HHS care."));

paper.push(h1("3. Methodology"));
paper.push(h2("3.1 Feature engineering"));
paper.push(bullet("Lag features: the value at t-1, t-7 and t-14 for care load, discharges, transfers, intake and CBP custody, plus the current value."));
paper.push(bullet("Rolling features: 7-day and 14-day rolling mean and variance of care load, discharges and transfers."));
paper.push(bullet("Flow signals: net pressure (transfers minus discharges), its 7-day and 14-day means, and the 7-day and 14-day change in care load."));
paper.push(bullet("Calendar effects: day of week and month of the target date, and a US federal holiday flag as a holiday proxy."));
paper.push(h2("3.2 Train-test strategy"));
paper.push(p("The split is strictly time-based with no random sampling. Evaluation uses expanding-window walk-forward validation: 52 weekly forecast origins (every Sunday from 22 December 2024 to 14 December 2025), each using only information available at that origin. Every origin is a real reporting day, so interpolated values never leak future information. At each origin every model forecasts horizons of 1, 3, 7, 14, 21 and 28 days, giving a multi-horizon evaluation of 6,426 forecast-actual pairs across models and targets; scoring uses only days with a real report (267 per model and target)."));
paper.push(h2("3.3 Forecasting models"));
paper.push(bullet([b("Baselines: "), "naive persistence (last value) and a 7-day moving average."]));
paper.push(bullet([b("Statistical: "), "ARIMA, SARIMA with a weekly seasonal term, and exponential smoothing (damped additive trend for care load, additive weekly seasonality for the flows)."]));
paper.push(bullet([b("Machine learning: "), "Random Forest and Gradient Boosting regressors in a direct multi-horizon design (one model per horizon). To cope with the level shift, care-load models predict the change from the current level and flow models predict the deviation from the 7-day mean."]));
paper.push(p("Three targets are forecast: children in HHS care, discharges and transfers into HHS. The last two together give the forecast imbalance between intake and exits."));
paper.push(h2("3.4 Evaluation metrics and uncertainty"));
paper.push(p("Accuracy is measured by MAE, RMSE and MAPE, and horizon error shows how reliability decays with lead time. Because daily discharges and transfers are small and sometimes zero, MAPE explodes for those series, so we also report WAPE (total absolute error divided by total actual volume). Prediction intervals use the empirical distribution of relative backtest errors for each model and horizon, applied to the current level. Their honesty is checked out of sample: the quantiles for each origin use only errors that were already observable at that time."));

paper.push(h1("4. Results"));
paper.push(h2("4.1 Care-load forecast accuracy"));
paper.push(p(`Table 1 shows MAPE by horizon. Over one to three days almost every model is within 1.5%, so the choice of model matters little. From seven days onward the models separate. Exponential smoothing and SARIMA are clearly best, with MAPE of ${N.hhs_mape[4][4]}% and ${N.hhs_mape[3][4]}% at 7 days, and ${N.hhs_mape[4][7]}% and ${N.hhs_mape[3][7]}% at 28 days. The machine-learning models are the weakest beyond two weeks (${N.hhs_mape[5][7]}% and ${N.hhs_mape[6][7]}% at 28 days), worse than naive persistence.`));
paper.push(table(hdrH, N.hhs_mape, wH, { highlight: r => r[0] === best }));
paper.push(caption("Table 1. Children in HHS care: MAPE (%) by forecast horizon (highlighted: best overall model)."));
paper.push(fig("fig5_error_by_horizon.png", 6.2));
paper.push(caption("Figure 5. Error grows with horizon. Left: care load MAPE. Right: discharge WAPE."));
paper.push(h2("4.2 Discharge and transfer forecasts"));
paper.push(p("Daily discharges are much harder to predict than the care load. Table 2 shows WAPE. The best models (moving average and Random Forest, about 38% overall) are only modestly better than naive persistence (49%), and ARIMA is the worst by a wide margin because it reverts toward an older, higher level. The moving average is the best or near-best model at almost every horizon; Random Forest edges it only at three days. Beyond three weeks no model is reliable for daily discharge counts, though weekly totals are far steadier than individual days."));
paper.push(table(hdrH, N.dis_wape, wH, { highlight: r => r[0] === "Moving Avg" || r[0] === "Random Forest" }));
paper.push(caption("Table 2. Discharges: WAPE (%) by forecast horizon (highlighted: two best overall)."));
paper.push(p(`Transfers into HHS are noisier still: the best model (moving average) has a WAPE of ${N.overall[1][10]}% overall, and the range across models is wide. Transfers should be treated as an input to scenarios rather than a precisely forecast quantity.`));
paper.push(h2("4.3 Overall comparison and statistical versus machine learning"));
paper.push(table(["Model", "Family", "HHS MAE", "HHS RMSE", "HHS MAPE %", "Disch. MAE", "Disch. RMSE", "Disch. WAPE %", "Transf. MAE", "Transf. RMSE", "Transf. WAPE %", "Avg rank"],
  N.overall, [1300, 1000, 700, 750, 750, 750, 750, 800, 750, 750, 800, 700], { size: 14, highlight: r => r[0] === best }));
paper.push(caption("Table 3. Overall walk-forward accuracy and average rank across horizons and targets (1 = best)."));
paper.push(p(`Exponential smoothing has the best average rank (${N.overall[4][11]}), followed by the moving average and Random Forest. The comparison between families depends on the target:`));
paper.push(bullet("Care load (a smooth, trending stock): statistical models win at every horizon from 7 days, with about half the error of the machine-learning models at 28 days."));
paper.push(bullet("Discharges (a noisy flow): machine learning and simple baselines are competitive, and the ARIMA and SARIMA models lose because their mean-reverting behaviour follows the old regime."));
paper.push(bullet("Transfers: the simplest approaches perform best; no complex model adds value."));
paper.push(fig("fig9_family_comparison.png", 6.0));
paper.push(caption("Figure 6. Average error by model family and horizon."));
paper.push(fig("fig7_backtest_7day.png", 6.0));
paper.push(caption("Figure 7. Seven-day-ahead forecasts against actuals across the backtest year."));

paper.push(h2("4.4 Forecast uncertainty"));
paper.push(p(`For the selected model the care-load bands behave well: the 80% band contained the actual value ${fmt(kp.coverage80_hhs)}% of the time and the 95% band ${fmt(kp.coverage95_hhs)}% of the time in a fully out-of-sample check. The slight shortfall at 95% reflects the early-2025 decline, which produced unusually large downward errors. Bands for discharges and transfers are less reliable, with 80% bands covering roughly 56% to 74% of outcomes, a sign that day-to-day volatility is not stable over time. Band width grows quickly with horizon for care load, from about ±16 children at one day to a width of about 800 at 28 days.`));
paper.push(table(["Series", "Days ahead", "80% band achieved", "95% band achieved", "Forecasts checked"], N.cov.map(r => [r[0] === "hhs" ? "Care load" : "Discharges", ...r.slice(1)]), [1600, 1400, 2000, 2000, 1800]));
paper.push(caption(`Table 4. Out-of-sample interval coverage for ${best} (nominal 80% and 95%).`));
paper.push(fig("fig8_interval_coverage.png", 5.8));
paper.push(caption("Figure 8. Achieved interval coverage against nominal levels (dotted lines)."));

paper.push(h2("4.5 Key performance indicators"));
paper.push(table(["KPI", "Definition used", "Value"], [
  ["Forecast accuracy", "100 minus MAPE for care load at 7 days", `${kp.forecast_accuracy_pct_hhs_7d.toFixed(1)}%`],
  ["Surge lead time", "Days of warning before an actual stress event (care load 5% above the origin level within 28 days), averaged over events the model caught", `${kp.surge.avg_lead_days.toFixed(0)} days, but only ${kp.surge.hits} of ${kp.surge.events} events caught`],
  ["Capacity breach probability", "Chance care load exceeds today's level + 5% within the window, from backtest error distribution", `${fmt(kp.breach_prob_14d_default * 100)}% (14 days); ${fmt(kp.breach_prob_28d_default * 100)}% (28 days)`],
  ["Forecast stability index", "100 minus average revision (%) between forecasts for the same date issued one week apart", kp.stability_index.toFixed(1)],
  ["Model robustness", "Share of backtest origin-horizon pairs in which the model beats naive persistence", `${fmt(kp.robustness_win_rate)}%`],
  ["Long-term reliability", "Error reduction versus naive persistence at 14 to 28 days", `${fmt(kp.long_term_skill_vs_naive)}%`]],
  [1900, 5400, 2100], { left: true, size: 17 }));
paper.push(caption(`Table 5. KPIs for ${best} on children in HHS care.`));
paper.push(p([b("Early warning is the weakest result. "), `Using a stress event defined as care load rising 5% above the origin level within 28 days, there were ${kp.surge.events} such events in the 52 backtest origins. The selected model caught only ${kp.surge.hits}. Gradient Boosting caught the most (6 of 14) but with 12 false alarms. Detecting turning points in a series that shifted regime once and has since drifted gently is simply hard with one year of testing, and the dashboard therefore presents breach probabilities as ranges of risk rather than as a yes-or-no alarm.`]));
paper.push(table(["Model", "Stress events", "Caught", "False alarms", "Recall", "Precision", "Avg lead (days)"], N.surge, [2200, 1200, 1100, 1300, 1100, 1200, 1300]));
paper.push(caption("Table 6. Capacity-stress warning performance by model."));

paper.push(h1("5. Forecast and capacity outlook"));
paper.push(p(`Using ${best} with all data through ${kp.origin}, the projected care load is shown below. The 80% range around the 28-day forecast is wide on the downside (the backtest year contained a steep fall) and narrower on the upside.`));
paper.push(table(["Days ahead", "Date", "Forecast", "80% range", "95% range"], N.fc_hhs, [1400, 2000, 1500, 2300, 2300]));
paper.push(caption("Table 7. Children in HHS care: forecast from 21 December 2025."));
paper.push(fig("fig6_forecast_fan.png", 5.6));
paper.push(caption("Figure 9. Forecasts with 80% and 95% bands. Flow panels show 7-day averages; the bands apply to single days."));
paper.push(h2("5.1 Will discharges offset incoming transfers?"));
paper.push(p(`The flow models project about ${(s28.discharges / 28).toFixed(1)} discharges and ${(s28.transfers / 28).toFixed(1)} transfers per day, or ${fmt(s28.discharges)} discharges against ${fmt(s28.transfers)} transfers over 28 days. Taken alone, that suggests discharges would exceed transfers by about ${fmt(s28.discharges - s28.transfers)} children. But historical data show roughly ${(N.flow.intercept_per_day / N.flow.beta).toFixed(1)} children per day entering care outside the transfers column (about ${fmt(extra28)} over 28 days). Against total inflow, projected discharges cover about ${fmt(s28.discharges / (s28.transfers + extra28) * 100)}%, a thin margin that is sensitive to the discharge rate. Note also that the care-load model extrapolates recent momentum (a rise of roughly 4 children per day since October) while the flow models revert to recent averages, so the two views are not perfectly consistent. The care-load forecast should be read as "if current momentum continues" and the flow forecast as "if flows settle at their recent norms".`));
paper.push(h2("5.2 Scenario comparison"));
paper.push(p(`The dashboard's scenario view converts changes in transfers and discharges into changes in care load using the fitted factor of ${N.flow.beta.toFixed(2)}. For example, a sustained doubling of transfers adds roughly ${fmt(N.flow.beta * s28.transfers)} children over 28 days, and a 50% increase in discharges removes roughly ${fmt(N.flow.beta * 0.5 * s28.discharges)}. These are planning aids, not causal estimates.`));

paper.push(h1("6. Recommendations"));
paper.push(bullet([b("Plan on a two-tier horizon. "), `Use the forecasts at face value for the next 7 to 14 days (error under 5%), and treat 3 to 4 week forecasts as scenario guidance (error about 8% to 11%). Do not use daily discharge forecasts beyond about two weeks; plan on weekly totals instead.`]));
paper.push(bullet([b("Define scale-up triggers on probabilities, not point forecasts. "), "A practical rule is to begin pre-positioning shelter, medical and caseworker capacity when the dashboard's breach probability for the next 14 to 28 days passes 20% (the amber level), and to commit once it passes 50% (red). The official capacity figure should replace the 5% headroom assumption used here."]));
paper.push(bullet([b("Pair the model with leading indicators. "), "Because automated surge detection caught few events, review intake into CBP custody and the CBP custody count alongside the forecast each week, and treat a rising net-pressure indicator as an early cue."]));
paper.push(bullet([b("Retrain weekly and monitor drift. "), "Refit as new reports arrive and track achieved interval coverage; a sustained drop below nominal levels signals another regime change."]));
paper.push(bullet([b("Staff discharge work around the weekday pattern. "), "Discharges peak midweek and are lowest on Fridays and Sundays; caseworker coverage on low days is the most direct lever on length of stay."]));
paper.push(bullet([b("Close the reporting gap. "), "A third of calendar days have no report, mostly Fridays and Saturdays. Daily reporting would improve flow forecasts, which suffer most from missing days."]));

paper.push(h1("7. Limitations"));
paper.push(bullet("The 2025 structural break dominates the evaluation year. Results describe how models cope with a regime shift and a calm period, but only one such shift is available."));
paper.push(bullet("There is no official capacity figure in the data. The 5% headroom threshold and the 5%-in-28-days stress definition are analyst assumptions and can be changed in the dashboard."));
paper.push(bullet("No external drivers (policy, enforcement activity, events) are modelled, so the forecasts cannot anticipate their effects."));
paper.push(bullet("Missing days are linearly interpolated for history and excluded from scoring; flows on Fridays and Saturdays are therefore poorly observed."));
paper.push(bullet("Interval coverage for the flow series is below nominal; the bands for discharges and transfers should be treated as optimistic."));
paper.push(bullet("The what-if scenarios use a statistical relationship between flows and care load, not a causal model of the program."));

paper.push(h1("8. Conclusion"));
paper.push(p(`Simple, well-tuned statistical models turn the UAC daily reports into a usable forecasting tool. Over a one to two week horizon, care load can be predicted to within a few percent, which is precise enough to support staffing and shelter decisions. Over longer horizons and for individual flows, the honest message is uncertainty, and the dashboard shows it through intervals, breach probabilities and scenarios. The most valuable next steps are an official capacity figure, richer leading indicators, and a more complete reporting calendar.`));

paper.push(h1("Appendix: reproducibility"));
paper.push(p("All results come from the pipeline in this project: pipeline/prepare.py (data preparation), models.py and backtest.py (features, models, walk-forward validation), evaluate.py (metrics, intervals, KPIs), finalize.py (final forecasts), figures.py (charts). Running python run_pipeline.py regenerates every output; the Streamlit app (app.py) reads those outputs. The random seed is fixed at 42."));

// =============================================================== EXECUTIVE SUMMARY
const ex = [];
ex.push(new Paragraph({ spacing: { before: 200, after: 60 }, children: [new TextRun({ text: "Executive Summary", font: FONT, size: 40, bold: true, color: NAVY })] }));
ex.push(p("Forecasting care load, discharge demand and capacity stress in the HHS Unaccompanied Children Program", { color: TEAL, size: 24 }));
ex.push(p(`For government stakeholders  |  Data through ${ds.end}`, { color: GREY, size: 18 }));
ex.push(h2("Why this matters"));
ex.push(p("The program reports daily counts but has had no forward view. Planners need to know how many children will be in care in the coming weeks, whether discharges will keep up with arrivals, and when to scale up shelters, medical staff and caseworkers. This project adds that forward view."));
ex.push(h2("What we found"));
ex.push(bullet([b(`Today: ${fmt(kp.current_hhs)} children in HHS care. `), `That is up about 15% since 1 October (${fmt(N.levels.oct1)}), after a low of ${fmt(ds.min_hhs)} in August 2025 and a peak of ${fmt(ds.peak_hhs)} in December 2023.`]));
ex.push(bullet([b("Next 2 weeks look stable. "), `Projected care load is ${fmt(kp.forecast_hhs_7d)} in 7 days and ${fmt(kp.forecast_hhs_14d)} in 14 days. At 7 days the forecast is typically within ${(kp.hhs_mape_7d).toFixed(1)}% of reality.`]));
ex.push(bullet([b("Next 4 weeks are less certain. "), `Projected ${fmt(kp.forecast_hhs_28d)} at 28 days, with an 80% range of about 1,870 to 2,720. The chance of exceeding a capacity level 5% above today's load is about ${fmt(kp.breach_prob_14d_default * 100)}% within 14 days and ${fmt(kp.breach_prob_28d_default * 100)}% within 28 days.`]));
ex.push(bullet([b("Discharges are keeping pace, narrowly. "), `Projected discharges (about ${(s28.discharges / 28).toFixed(0)} per day) exceed projected transfers (about ${(s28.transfers / 28).toFixed(0)} per day), but once other routes into care are counted the margin is thin, about ${fmt(s28.discharges / (s28.transfers + extra28) * 100)}% coverage of total inflow.`]));
ex.push(h2("How reliable is it?"));
ex.push(table(["Measure", "Result"], [
  ["Care-load accuracy, 7 days ahead", `${kp.forecast_accuracy_pct_hhs_7d.toFixed(1)}%`],
  ["Care-load accuracy, 28 days ahead", `${(100 - Number(N.hhs_mape[4][7])).toFixed(0)}%`],
  ["Daily discharge forecasts", "Roughly 30% to 40% average error; use weekly totals"],
  ["Forecast ranges (80%) holding up in testing", `${fmt(kp.coverage80_hhs)}% for care load`],
  ["Automated surge warning", "Weak: caught 1 of 14 stress events (best model: 6 of 14, with many false alarms)"]],
  [4200, 5200], { left: true, size: 18 }));
ex.push(h2("Recommended actions"));
ex.push(bullet("Use 7 to 14 day forecasts for staffing and shelter decisions; use 3 to 4 week forecasts as scenarios."));
ex.push(bullet("Set scale-up triggers on breach probability: prepare at 20%, commit at 50%, using the official capacity figure."));
ex.push(bullet("Review intake and CBP custody counts weekly alongside the forecast, since automated warnings alone are not enough."));
ex.push(bullet("Retrain weekly and watch forecast accuracy for signs of a new shift in program conditions."));
ex.push(bullet("Restore daily reporting on Fridays and Saturdays, which are missing for about a third of days."));
ex.push(h2("Caveats"));
ex.push(p("Forecasts assume current conditions continue. They do not anticipate policy changes or border events, the capacity threshold used is an assumption, and the data include one large drop in early 2025 that tests the models. Full method and results are in the accompanying research paper; the interactive dashboard shows the forecasts, uncertainty bands and what-if scenarios.", { size: 19 }));

async function save(children, name, label) {
  const doc = new Document({ styles, numbering, creator: "UAC Forecasting Project", title: label,
    sections: [{ properties: name.includes("Executive") ? { page: { size: { width: 12240, height: 15840 }, margin: { top: 800, right: 1200, bottom: 800, left: 1200 } } } : page, footers: footer(label), children }] });
  fs.writeFileSync(path.join(__dirname, name), await Packer.toBuffer(doc));
  console.log("wrote", name);
}
(async () => {
  await save(paper, "UAC_Forecasting_Research_Paper.docx", "UAC Forecasting Research Paper");
  await save(ex, "UAC_Forecasting_Executive_Summary.docx", "UAC Forecasting Executive Summary");
})();
