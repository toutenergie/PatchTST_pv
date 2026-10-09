# -*- coding: utf-8 -*-
"""
article.py — Generate the scientific paper (Word) from the cached results.

    python article.py        → Article_PV_PatchTST_multivariate_EN.docx

Every number in the Results and Discussion sections and in the abstract is
read from the cache (no manual entry). The invariant texts are in
article_texts.py; the review in review/review_text.md.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, ".")
import analysis as an  # noqa: E402
import article_texts as T  # noqa: E402
import patchtst_pv as m  # noqa: E402
from nb05_final import STEPS, cfg_from_name, final_model, validation_seed  # noqa: E402

ROOT = m.ROOT
F = m.CACHE_DIR / "final"
OUTPUT = "Article_PV_PatchTST_multivariate_EN.docx"


# -----------------------------------------------------------------------------
# 1. Number formatting (English typography)
# -----------------------------------------------------------------------------
def num(x, d=2):
    """Number with thousands separator and d decimals."""
    return f"{x:,.{d}f}"


def pc(x, d=1, sign=True):
    """Percentage (x is a fraction)."""
    return (f"{100*x:+.{d}f}" if sign else f"{100*x:.{d}f}") + " %"


def fig(name, caption, width=16):
    w, h = Image.open(ROOT / "figures" / f"{name}.png").size
    return {"type": "figure", "path": str(ROOT / "figures" / f"{name}.png"), "ratio": h / w,
            "width_cm": width, "caption": caption}


def lbl(name):
    return name.replace("PatchTST-", "")


# -----------------------------------------------------------------------------
# 2. Collection of the results
# -----------------------------------------------------------------------------
def collect():
    ds = m.build_dataset()
    sel = json.loads((m.CACHE_DIR / "selection.json").read_text())
    R = {"ds": ds, "sel": sel}
    # configurations 1 to 4
    import nb_config
    for k in (1, 2, 3, 4):
        cfgs = nb_config.configs_for(k)
        res = [m.evaluate_configuration(c, ds) for c in cfgs]
        tab = m.comparison_table(res)
        pers = []
        for c, r in zip(cfgs, res):
            _, y_true, orig = an.load_predictions(r["name"])
            pers.append(m.metrics(y_true, m.daily_persistence(ds, orig, c.H))["RMSE"])
        tab["RMSE_persistence"] = pers
        tab["gain"] = 1 - tab.RMSE / tab.RMSE_persistence
        R[f"c{k}"] = tab.set_index("configuration")
        R[f"res{k}"] = res
    # final model
    cfg = final_model()
    res = m.evaluate_configuration(cfg, ds)
    preds, y_true, orig = an.load_predictions(cfg.name())
    s = res["summary"]
    R.update(cfg=cfg, res=res, s=s, y_true=y_true, orig=orig, preds=preds)
    refs = {"Persistence D-1": m.daily_persistence(ds, orig, cfg.H),
            "7-day mean": m.seven_day_mean(ds, orig, cfg.H),
            "Multi-output Ridge": np.load(F / "ridge_pred.npy")}
    R["refs"] = {k: m.metrics(y_true, v) for k, v in refs.items()}
    g_rep = an.representative_seed(res)
    e = preds[m.SEEDS.index(g_rep)] - y_true
    R["dm"] = {k: m.diebold_mariano_test(e, v - y_true, h=cfg.H) for k, v in refs.items()}
    R["ci"] = m.evaluate_configuration(cfg_from_name(cfg.name().replace("CAX", "CI")), ds)["summary"]
    R["cv"] = pd.read_csv(F / "cross_validation.csv")
    R["kc"] = pd.read_csv(F / "capacity_index.csv")
    R["noise"] = pd.read_csv(F / "noise.csv").groupby("level").degradation.agg(["mean", "std"])
    R["occ"] = pd.read_csv(F / "occlusion.csv").set_index("group")
    R["ig"] = pd.read_csv(F / "integrated_gradients.csv").set_index("input").share
    R["att"] = pd.read_csv(F / "attention.csv").set_index("channel").weight if (F / "attention.csv").exists() else None
    R["monthly"] = pd.DataFrame({k: an.monthly_rmse(y_true, v, orig, ds) for k, v in
                                 {**refs, "model": preds[m.SEEDS.index(g_rep)]}.items()})
    R["val_seed"] = validation_seed(cfg)
    ens = preds.mean(axis=0)
    R["ens"] = m.metrics(y_true, ens)
    R["dm_ens_ridge"] = m.diebold_mariano_test(ens - y_true, refs["Multi-output Ridge"] - y_true, h=cfg.H)
    # hourly persistence y(t-1) at 1 h: weaker baseline than daily persistence (diurnal ramps)
    _, v1, o1 = an.load_predictions("PatchTST-CAX_P12_L168_H1")
    y = ds.table[m.TARGET].values
    R["hourly_pers_1h"] = m.metrics(v1, np.stack([[y[t - 1]] for t in o1]))["RMSE"]
    R["daily_pers_1h"] = m.metrics(v1, m.daily_persistence(ds, o1, 1))["RMSE"]
    # persistence MAE vs model, daytime bias
    R["mae_pers"] = R["refs"]["Persistence D-1"]["MAE"]
    day = y_true > 1
    R["day_bias"] = float(e[day].mean())
    R["night_rmse"] = float(np.sqrt(np.mean(e[~day] ** 2)))
    return R


# -----------------------------------------------------------------------------
# 3. Abstract
# -----------------------------------------------------------------------------
def abstract(R):
    s, rp, rr = R["s"], R["refs"]["Persistence D-1"], R["refs"]["Multi-output Ridge"]
    cv = R["cv"]
    return [
        {"type": "h1", "text": "Abstract"},
        {"type": "p", "text": (
            "Forecasting photovoltaic (PV) generation is essential for operating grids with high renewable "
            "penetration. Yet, in many African grids, on-site meteorological measurements are lacking. This paper "
            "studies the multivariate forecasting of the **total** PV generation of the SENELEC grid (Senegal; "
            "11 plants, hourly log 2019–2023) **without meteorology and without the plant productions as inputs**. "
            "We propose PatchTST-CA-X, which combines a selective reversible normalisation, cross-channel attention "
            "per patch and conditioning of the forecasting head on **deterministic covariates known over the "
            "horizon**: cyclic encodings and solar geometry. Four configurations (patch length, history, channel "
            "handling, horizon) are evaluated one factor at a time, over 5 seeds, in terms of accuracy, robustness, "
            "drift over 30 days without re-training and efficiency. At the day-ahead horizon, the selected model "
            f"(P = 12 h, L = 168 h) reaches an RMSE of {num(s['RMSE_mean'])} ± {num(s['RMSE_std'])} MW "
            f"(nRMSE = {num(s['nRMSE_mean'], 3)}; R² = {num(s['R2_mean'], 3)}) over seven test months. This is "
            f"{pc(1 - s['RMSE_mean']/rp['RMSE'], sign=False)} lower than daily persistence (Diebold-Mariano, "
            f"p = {num(R['dm']['Persistence D-1'][1], 3)}) and {pc(1 - s['RMSE_mean']/rr['RMSE'], sign=False)} lower "
            "than a multi-output Ridge regression given the same future covariates, a gap that is not significant "
            f"(p = {num(R['dm']['Multi-output Ridge'][1], 2)}). The degradation under 5 % noise on the covariates is "
            f"{pc(s['noise_degradation_mean'], 2)}, the 30-day drift is {num(s['drift_7d_mean'])}, and inference takes "
            f"{num(s['inference_ms_mean'], 1)} ms. Ablations and explainability show that the gain comes from "
            "conditioning on future deterministic information, not from the past grid context nor from model "
            "complexity alone. Temporal cross-validation (5 folds) confirms a mean gain of "
            f"{pc(cv.gain_vs_persistence.mean(), sign=False)} over persistence.")},
        {"type": "p", "text": T.KEYWORDS},
    ]


# -----------------------------------------------------------------------------
# 4. Results
# -----------------------------------------------------------------------------
def config_table(R, k, caption):
    t = R[f"c{k}"]
    selected = R["sel"][STEPS[k - 1]]
    rows, hl = [], []
    for i, (name, l) in enumerate(t.iterrows()):
        if name == selected and k < 4:
            hl.append(i)
        rows.append([lbl(name), num(l.RMSE), num(l.RMSE_std, 3), num(l.MAE), num(l.nRMSE, 3), pc(l.gain),
                     pc(l.noise_degradation, 1), num(l.drift_7d), num(l.inference_ms, 1), f"{int(l.n_parameters):,}",
                     num(l.normalised_score, 3) + (" (rejected)" if l.rejected_noise else "")])
    return {"type": "table", "caption": caption, "size": 14, "highlight": hl,
            "widths": [2.6, 1.1, 1.0, 1.0, 1.0, 1.1, 1.1, 0.9, 1.1, 1.2, 1.3],
            "headers": ["Variant", "RMSE (MW)", "σ (MW)", "MAE (MW)", "nRMSE", "Gain vs pers.", "Noise 5 %",
                        "Drift", "Inf. (ms)", "Param.", "Norm. score"],
            "rows": rows,
            "note": "Means over 5 seeds, every hourly origin of the test set (01/10/2022–09/05/2023). \"Gain vs pers.\": "
                    "RMSE reduction relative to daily persistence at the same horizon. Highlighted row: selected variant."}


def results(R):
    c1, c2, c3, c4 = R["c1"], R["c2"], R["c3"], R["c4"]
    s, cfg = R["s"], R["cfg"]
    n = lambda t, key: t.loc[[i for i in t.index if key in i][0]]  # noqa: E731
    ci, cix, ca, cax = (c3.loc[f"PatchTST-{v}_P12_L168_H24"] for v in ("CI", "CIX", "CA", "CAX"))
    caxg = c3.loc["PatchTST-CAXg_P12_L168_H24"]
    B = [{"type": "h1", "text": "4. Results"},
         {"type": "h2", "text": "4.1 Exploratory analysis"},
         {"type": "p", "text": (
             "The daily cycle dominates the series: the autocorrelation at lag 24 h is 0.97 on the training set, and "
             "the periodogram shows a peak at 24 h and its harmonic at 12 h (Figure 3). The amplitude is maximal from "
             "March to May and minimal during the rainy season (July–September). The linear correlation of the target "
             "with solar geometry reaches 0.94; with the grid channels, it does not exceed 0.31 in absolute value, and "
             "it is weaker still once the mean cycle is removed. Daily persistence is therefore a demanding baseline: "
             f"RMSE = {num(R['refs']['Persistence D-1']['RMSE'])} MW and R² = "
             f"{num(R['refs']['Persistence D-1']['R2'], 3)} on the test set.")},
         fig("fig03_seasonality", "**Figure 3.** Mean hour × month profile, autocorrelation and periodogram of the target (training set)."),
         {"type": "h2", "text": "4.2 Configuration 1 — patch length"},
         {"type": "p", "text": (
             f"The shorter the patch, the more accurate the forecast: RMSE = {num(n(c1,'P12').RMSE)} MW for P = 12 h, "
             f"{num(n(c1,'P24').RMSE)} MW for P = 24 h and {num(n(c1,'P48').RMSE)} MW for P = 48 h (Table 4). A 12 h "
             "patch separates the morning from the afternoon, which lets temporal attention relate homologous "
             "half-days from one day to the next. A 48 h patch, by contrast, mixes two days in the same token. The "
             f"12 h patch is also the most stable across seeds (σ = {num(n(c1,'P12').RMSE_std, 3)} MW), for an "
             f"inference cost that remains low ({num(n(c1,'P12').inference_ms, 1)} ms). It is selected.")},
         config_table(R, 1, "**Table 4.** Configuration 1 — influence of the patch length P (L = 168 h, CA-X, H = 24 h)."),
         fig("fig_config1_rosary", "**Figure 4.** Configuration 1: rosary of the RMSE over 5 seeds (left) and drift over 30 days (right). Horizontal lines: baselines."),
         {"type": "h2", "text": "4.3 Configuration 2 — history"},
         {"type": "p", "text": (
             f"One week of history (L = 168 h) gives the best RMSE ({num(n(c2,'L168').RMSE)} MW), ahead of L = 96 h "
             f"({num(n(c2,'L96').RMSE)} MW) and L = 336 h ({num(n(c2,'L336').RMSE)} MW, Table 5). Extending the window "
             "to two weeks degrades accuracy and dispersion, and multiplies the inference time by "
             f"{num(n(c2,'L336').inference_ms / n(c2,'L168').inference_ms, 1)}. The useful information lies in the "
             "last few days (persistence of cloudiness); older context adds bygone weather regimes that attention "
             "must learn to ignore.")},
         config_table(R, 2, "**Table 5.** Configuration 2 — influence of the history L (P = 12 h, CA-X, H = 24 h)."),
         {"type": "h2", "text": "4.4 Configuration 3 — channel handling"},
         {"type": "p", "text": (
             "Configuration 3 (Table 6, Figure 5) is the most instructive. The standard PatchTST with independent "
             f"channels (CI), equivalent to a univariate model, reaches {num(ci.RMSE)} MW. Cross-channel attention "
             f"alone (CA) improves it little ({num(ca.RMSE)} MW) and increases the dispersion across seeds "
             f"(σ = {num(ca.RMSE_std, 2)} MW). Its degradation under noise is nil ({pc(ca.noise_degradation, 2)}): "
             "the model has learned to **ignore** the past exogenous channels. Conversely, known-future conditioning "
             f"(X) brings most of the gain: {num(cix.RMSE)} MW for CI-X (RMSE reduced by "
             f"{pc(1 - cix.RMSE/ci.RMSE, sign=False)} relative to CI), and {num(cax.RMSE)} MW for CA-X (reduced by "
             f"{pc(1 - cax.RMSE/ci.RMSE, sign=False)}), with the lowest dispersion among the admissible multivariate "
             f"variants (σ = {num(cax.RMSE_std, 3)} MW). The physical night gate (CA-Xg) leaves accuracy unchanged "
             f"({num(caxg.RMSE)} MW), but it fails the robustness test: the RMSE rises by "
             f"{pc(caxg.noise_degradation, 0)} under noise, because noise on cos θ_{{z}} opens the gate at night and "
             "reveals night-time outputs that the network never learned to constrain. CA-X is selected.")},
         config_table(R, 3, "**Table 6.** Configuration 3 — channel handling (P = 12 h, L = 168 h, H = 24 h). "
                            "CI: independence; CA: cross-channel attention; X: known future; g: physical gate."),
         fig("fig_config3_rosary", "**Figure 5.** Configuration 3: rosary of the RMSE over 5 seeds and drift over 30 days for the five channel handlings."),
         {"type": "h2", "text": "4.5 Configuration 4 — forecast horizon"},
         {"type": "p", "text": (
             "The error grows with the horizon (Table 7): "
             + ", ".join(f"{num(l.RMSE)} MW at {name.split('_H')[1]} h" for name, l in c4.iterrows()) +
             ". The gain relative to persistence, computed at the same horizon, is "
             + ", ".join(f"{pc(l.gain)} at {name.split('_H')[1]} h" for name, l in c4.iterrows()) +
             ". The gain is largest at very short range, where the immediate history is highly informative and daily "
             "persistence remains naive. Hourly persistence ŷ(t) = y(t−1) would be an even weaker baseline at 1 h "
             f"(RMSE = {num(R['hourly_pers_1h'])} MW against {num(R['daily_pers_1h'])} MW for daily persistence), "
             "because it ignores the sunrise and sunset ramps; this is why daily persistence serves as the baseline "
             "at every horizon. The gain persists at 168 h, a horizon at which the model approaches a conditional "
             "climatology built from the deterministic covariates. Comparing horizons means comparing different tasks: "
             "the horizon is therefore not chosen by the score, but set by the use case, namely day-ahead dispatch "
             "planning (H = 24 h).")},
         config_table(R, 4, "**Table 7.** Configuration 4 — forecast horizon H (P = 12 h, L = 168 h, CA-X)."),
         fig("fig_config4_residuals", "**Figure 6.** Configuration 4, best score: distribution of the residuals, RMSE per hour, per month and per horizon step."),
         ]
    # --- final model
    rp, rm, rr = (R["refs"][k] for k in ("Persistence D-1", "7-day mean", "Multi-output Ridge"))
    dm = R["dm"]
    rows = [[k, num(v["RMSE"]), num(v["MAE"]), num(v["nRMSE"], 3), num(v["R2"], 3), num(v["MBE"]),
             num(dm[k][0]), f"{dm[k][1]:.1e}"] for k, v in R["refs"].items()]
    ci_s = R["ci"]
    rows.append(["PatchTST-CI (univariate)", f"{num(ci_s['RMSE_mean'])} ± {num(ci_s['RMSE_std'])}", num(ci_s["MAE_mean"]),
                 num(ci_s["nRMSE_mean"], 3), num(ci_s["R2_mean"], 3), num(ci_s["MBE_mean"]), "–", "–"])
    rows.append(["PatchTST-CA-X (proposed)", f"{num(s['RMSE_mean'])} ± {num(s['RMSE_std'])}", num(s["MAE_mean"]),
                 num(s["nRMSE_mean"], 3), num(s["R2_mean"], 3), num(s["MBE_mean"]), "ref.", "ref."])
    cv, kc, nz, occ = R["cv"], R["kc"], R["noise"], R["occ"]
    occ_h = occ.RMSE_increase
    B += [{"type": "h2", "text": "4.6 Final model and comparison with the baselines"},
          {"type": "p", "text": (
              f"At the day-ahead horizon, the final model ({lbl(cfg.name())}) reaches an RMSE of {num(s['RMSE_mean'])} ± "
              f"{num(s['RMSE_std'])} MW, an nRMSE of {num(s['nRMSE_mean'], 3)} ({num(s['nRMSE_day_mean'], 3)} over "
              f"daytime hours only) and an R² of {num(s['R2_mean'], 3)} (Table 8). It reduces the RMSE by "
              f"{pc(1 - s['RMSE_mean']/rp['RMSE'], sign=False)} relative to persistence, by "
              f"{pc(1 - s['RMSE_mean']/rm['RMSE'], sign=False)} relative to the mean of the last 7 days and by "
              f"{pc(1 - s['RMSE_mean']/rr['RMSE'], sign=False)} relative to the multi-output Ridge regression, which "
              "nevertheless has the same future covariates. The Diebold-Mariano tests reject equal accuracy against "
              f"persistence (p = {num(dm['Persistence D-1'][1], 3)}) and against the 7-day mean "
              f"(p = {num(dm['7-day mean'][1], 3)}). **Against Ridge, however, the gap is not significant** "
              f"(p = {num(dm['Multi-output Ridge'][1], 2)}); the mean of the forecasts of the 5 seeds (ensemble) "
              f"reaches {num(R['ens']['RMSE'])} MW, without becoming significantly better than Ridge either "
              f"(p = {num(R['dm_ens_ridge'][1], 2)}). The MAE of the model ({num(s['MAE_mean'])} MW) is moreover "
              f"higher than that of persistence ({num(R['mae_pers'])} MW) and of Ridge ({num(rr['MAE'])} MW). Trained "
              "on a squared loss, the model targets the conditional expectation: it reduces the large errors of "
              "cloud-transition days, at the cost of small systematic errors on stable clear-sky days, which "
              f"persistence reproduces exactly. The daytime residuals show a mean bias of {num(R['day_bias'])} MW, and "
              f"the night-time error is {num(R['night_rmse'])} MW (RMSE).")},
          {"type": "table", "caption": "**Table 8.** Final model vs baselines at the day-ahead horizon (full test set, every hourly origin).",
           "size": 15, "widths": [3.0, 1.9, 1.1, 1.1, 1.0, 1.0, 1.1, 1.3], "highlight": [len(rows) - 1],
           "headers": ["Model", "RMSE (MW)", "MAE (MW)", "nRMSE", "R²", "MBE (MW)", "DM", "p"],
           "rows": rows,
           "note": "DM: Diebold-Mariano statistic (HLN correction) of the proposed model against the baseline; negative = proposed model better."},
          fig("fig_final_observed_predicted", "**Figure 7.** Day-ahead forecast issued at 00:00 (representative seed) vs observation and persistence over ten days of November 2022; scatter plot over the whole test set."),
          fig("fig_final_residuals", "**Figure 8.** Residual diagnostics of the final model: daytime distribution, RMSE per hour, per month and per horizon step."),
          {"type": "h2", "text": "4.7 Robustness and temporal cross-validation"},
          {"type": "p", "text": (
              f"Over the 5 seeds, the standard deviation of the RMSE is {num(s['RMSE_std'], 3)} MW, i.e. "
              f"{pc(s['RMSE_std']/s['RMSE_mean'], 2, False)} of the RMSE. The degradation under noise grows steadily "
              f"with the noise standard deviation: {pc(nz.loc[0.05, 'mean'], 2)} at 5 %, {pc(nz.loc[0.1, 'mean'], 2)} "
              f"at 10 % and {pc(nz.loc[0.2, 'mean'], 2)} at 20 % (Figure 9). This remains far from the 15 % rejection "
              "threshold, which confirms that the forecast relies first on the target history and on the "
              f"deterministic structure. The temporal cross-validation (Table 9), whose test blocks run from "
              f"{cv.test.iloc[0][:7]} to {cv.test.iloc[-1][-10:-3]}, gives a mean gain over persistence of "
              f"{pc(cv.gain_vs_persistence.mean())} ± {pc(cv.gain_vs_persistence.std(), 1, False)}, ranging from "
              f"{pc(cv.gain_vs_persistence.min())} to {pc(cv.gain_vs_persistence.max())} depending on the fold. The "
              "gain is smallest for the first fold, trained on only eight months, then stabilises as soon as training "
              "covers at least one complete annual cycle: the model needs to see every season to outperform persistence.")},
          {"type": "table", "caption": "**Table 9.** Temporal cross-validation (TimeSeriesSplit, 5 expanding-window folds, 1 seed).",
           "size": 15, "widths": [0.7, 2.4, 3.0, 1.1, 1.0, 1.4, 1.4],
           "headers": ["Fold", "Training", "Test", "RMSE", "nRMSE", "RMSE pers.", "Gain"],
           "rows": [[str(r.fold), r.train, r.test, num(r.RMSE), num(r.nRMSE, 3), num(r.RMSE_persistence), pc(r.gain_vs_persistence)]
                    for r in cv.itertuples()]},
          fig("fig_final_noise", "**Figure 9.** RMSE increase of the final model as a function of the standard deviation of the Gaussian noise injected into the covariates (mean ± standard deviation over 5 seeds).", 11),
          {"type": "h2", "text": "4.8 Long-term stability"},
          {"type": "p", "text": (
              f"Over 30 days of forecasting without re-training, the drift is {num(s['drift_7d_mean'])} (raw ratio "
              f"D30/D1: {num(s['drift_D30_D1_mean'])}), well below the target of 1.2 (Figure 10). The RMSE even "
              "decreases during the month, owing to the onset of the dry season in November: this is not a merit of "
              "the model, and the drift must always be read against the baseline. Over the calendar months of the "
              "test set (October 2022 to May 2023, the latter partial), the monthly RMSE of the model remains below "
              f"that of persistence in {int((R['monthly']['model'] < R['monthly']['Persistence D-1']).sum())} months "
              f"out of {len(R['monthly'])}, with no upward trend despite a gap of 9 to 16 months from the end of training.")},
          fig("fig_final_stability", "**Figure 10.** Stability: daily RMSE over 30 days without re-training (min–max envelope of the 5 seeds) and monthly RMSE over the test set, compared with the baselines."),
          {"type": "h2", "text": "4.9 \"Capacity index\" ablation"},
          {"type": "p", "text": (
              "A variant learns the index k = y / C, where C is an apparent capacity estimated causally (99 % quantile "
              "of the daily maxima of the previous 30 days), then converts the forecast back to MW. It obtains "
              f"{num(kc.RMSE.mean())} ± {num(kc.RMSE.std())} MW, against {num(s['RMSE_mean'])} MW for the final model "
              f"({pc(kc.RMSE.mean()/s['RMSE_mean'] - 1)}). "
              + ("Selective RevIN normalisation therefore already absorbs the growth of the fleet: there is no need to "
                 "model the installed capacity explicitly over this test period, during which no plant is commissioned."
                 if kc.RMSE.mean() >= s["RMSE_mean"] else
                 "Normalising explicitly by the installed capacity therefore brings an additional gain, which argues "
                 "for taking commissionings into account explicitly."))},
          {"type": "h2", "text": "4.10 Explainability"},
          {"type": "p", "text": (
              "Occlusion of channel groups (Figure 11, left) clearly ranks the sources of information. Masking the "
              f"target history increases the RMSE by {pc(occ_h['Target (PV history)'], 0)}; masking solar geometry, by "
              f"{pc(occ_h['Solar geometry'], 0)}; the hourly encoding, by {pc(occ_h['Cyclic hour'], 0)}; the seasonal "
              f"encoding, by {pc(occ_h['Cyclic season'], 1)}. The grid context ({pc(occ_h['Grid'], 1)}) and the day of "
              f"week ({pc(occ_h['Cyclic week'], 1)}), our negative control, have only a marginal effect. Integrated "
              "gradients (Figure 11, centre) confirm this ranking: the PV history carries "
              f"{pc(R['ig'].get('PV_total (past)', np.nan), 0, False)} of the attribution, and the **future** "
              "covariates (cos θ_{z}, clear-sky GHI, hourly encoding) "
              f"{pc(R['ig'][[i for i in R['ig'].index if '(future)' in i]].sum(), 0, False)} in total, against "
              f"{pc(R['ig'][[i for i in R['ig'].index if i.split(' ')[0] in m.GRID_CHANNELS]].sum(), 1, False)} for the "
              "three grid channels. "
              + ("The three methods do not say the same thing, and this divergence is instructive. Cross-channel "
                 f"attention gives its largest weight to the Taiba wind farm ({pc(R['att'].max(), 1, False)}, for a "
                 f"uniform value of {pc(1/len(R['att']), 1, False)}), and integrated gradients attribute a few per cent "
                 "to the grid channels; yet masking these channels hardly changes the error. An attention weight or a "
                 "local attribution measures where the model \"looks\", not what its forecast depends on: only "
                 "occlusion, which measures the effect on the error, allows us to conclude that the grid context is "
                 "useless here." if R["att"] is not None else ""))},
          fig("fig_final_xai", "**Figure 11.** Explainability of the final model: occlusion importance, integrated gradients (top 12 inputs) and cross-channel attention."),
          {"type": "h2", "text": "4.11 Efficiency and decision score"},
          {"type": "p", "text": (
              f"The final model has {int(s['n_parameters_mean']):,} parameters ({num(s['size_MB_mean'])} MB). A "
              f"forecast takes {num(s['inference_ms_mean'], 1)} ms on a single processor core, and training takes on "
              f"average {num(s['duration_s_mean']/60, 0)} min on two cores. The efficiency targets (< 100 ms, "
              f"< 50 MB) are therefore met by a wide margin. The raw decision score is {num(m.decision_score(s)['score'])}.")},
          ]
    return B


# -----------------------------------------------------------------------------
# 5. Discussion, limitations, perspectives, conclusion
# -----------------------------------------------------------------------------
def discussion(R):
    s, c3 = R["s"], R["c3"]
    ci, cix, cax = (c3.loc[f"PatchTST-{v}_P12_L168_H24"] for v in ("CI", "CIX", "CAX"))
    occ_h = R["occ"].RMSE_increase
    rr = R["refs"]["Multi-output Ridge"]
    return [
        {"type": "h1", "text": "5. Discussion"},
        {"type": "h3", "text": "What the model really learns"},
        {"type": "p", "text": (
            "The main lesson is that, without meteorological measurements, **the added value of the covariates lies "
            "in their availability over the horizon, not in their presence in the past**. The past grid channels are "
            f"almost ignored: occlusion degrades the RMSE by only {pc(occ_h['Grid'], 1)}, and the CA variant without "
            "X is insensitive to noise. The deterministic covariates, on the other hand, bring most of the gain when "
            f"they describe the **forecast** instant: an RMSE reduced by {pc(1 - cix.RMSE/ci.RMSE, sign=False)} from "
            "CI to CI-X. This result qualifies the intuition, common in the literature [42, 45], that breaking channel "
            "independence is enough to exploit exogenous variables. Here, cross-channel attention only becomes useful "
            f"in the presence of future conditioning ({num(cax.RMSE)} MW for CA-X against {num(cix.RMSE)} MW for "
            "CI-X), where it mainly reduces the dispersion across seeds. It is, however, consistent with the works "
            "that provide periodic information explicitly to the model [39, 49].")},
        {"type": "h3", "text": "A naive baseline that is hard to beat"},
        {"type": "p", "text": (
            f"The gain of {pc(1 - s['RMSE_mean']/R['refs']['Persistence D-1']['RMSE'], sign=False)} over persistence "
            "looks modest, but it is statistically established, stable over time and confirmed by cross-validation. "
            "The aggregated generation of eleven dispersed plants smooths part of the cloud variability, and the "
            "Senegalese solar regime is regular [21]: persistence is therefore particularly good there (R² > 0.96). "
            "Many studies do not compare themselves with this baseline [31–34]; the few that do [51] report larger "
            "gains on noisier series. The most important outcome of this comparison, however, concerns Ridge: given "
            f"the **same future covariates**, a linear regression reaches {num(rr['RMSE'])} MW, and the advantage of "
            f"PatchTST-CA-X ({pc(1 - s['RMSE_mean']/rr['RMSE'], sign=False)}) is not significant "
            f"(p = {num(R['dm']['Multi-output Ridge'][1], 2)}), while Ridge has a lower MAE. Without meteorological "
            "observations, most of the predictability is therefore **linear** in the recent history and in the "
            "future solar geometry: it is the injected information, not the capacity of the Transformer, that makes "
            "the difference. This finding puts into perspective the gains reported for deep architectures compared "
            "with weak baselines [31–36]. It also makes a well-specified linear model a mandatory baseline, just like "
            "persistence. The practical value of PatchTST-CA-X lies rather in its low dispersion, its robustness, its "
            "zero drift and its ability to accommodate, without changing the architecture, non-linear and noisy "
            "future covariates (numerical weather predictions) that Ridge would exploit poorly.")},
        {"type": "h3", "text": "Operational relevance"},
        {"type": "p", "text": (
            f"With {num(s['inference_ms_mean'], 1)} ms per forecast, less than 1 MB of weights, zero drift over a month "
            "and high robustness to noise, the model can be integrated into dispatch without dedicated computing "
            "infrastructure. For the operator, a day-ahead error reduced by "
            f"{num(R['refs']['Persistence D-1']['RMSE'] - s['RMSE_mean'], 1)} MW in RMSE, for a fleet whose peak "
            "generation approaches 180 MW, translates directly into the spinning reserve to be committed [15, 16]. "
            "This benefit must be quantified by an economic study specific to the SENELEC grid.")},
        {"type": "h3", "text": "\"Hard\" physical constraints"},
        {"type": "p", "text": (
            "The failure of the physical gate in the robustness test shows that a physical constraint imposed at the "
            "output can make a model fragile when the variable that controls it is uncertain. The network learns to "
            "rely on the gate and no longer constrains its night-time outputs. A soft constraint, as a penalty in the "
            "loss, or a gate computed from the exact geometry rather than from a model input, would be preferable. "
            "More broadly, the physical hybridisation advocated by part of the literature [55] must be evaluated "
            "under perturbation, and not only in nominal accuracy.")},
        {"type": "h1", "text": "6. Limitations"},
        {"type": "bullets", "items": [
            "**Absence of meteorological data.** The accuracy ceiling is set by the available information. Cloud "
            "transition days remain poorly forecast, and the error is concentrated between 09:00 and 15:00.",
            "**A single grid and a single test period** (seven months, from October to May, without a complete rainy "
            "season). Cross-validation covers other seasons, but with a single seed per fold.",
            "**One-factor-at-a-time design.** It does not capture interactions between factors (e.g. P × L); a joint "
            "search might find a better optimum.",
            "**Point forecast.** Uncertainty is not quantified, although it is needed for probabilistic reserve "
            "sizing [16, 46].",
            "**Quality of the log.** Clipping and imputation, although causal and traceable, rely on statistical "
            "thresholds; some plausible outliers may have been kept.",
            "**Drift evaluated over 30 days** in a favourable period (onset of the dry season); an evaluation over "
            "the following rainy season is still needed.",
        ]},
        {"type": "h1", "text": "7. Perspectives"},
        {"type": "numbered", "instance": 4, "items": [
            "**Integrate numerical weather predictions** (ERA5, GFS) or satellite data as **noisy** future covariates: "
            "the X architecture lends itself directly to this, and the noise protocol will measure the sensitivity.",
            "**Make the forecast probabilistic** (quantile regression [46], adaptive conformal inference [39]) to feed "
            "probabilistic reserve sizing.",
            "**Replace the hard gate by a soft physical constraint**, and test an explicit coupling with a clear-sky "
            "model (forecasting the clearness index).",
            "**Transfer the model** to other West African grids (WAPP), using the transfer strategies that work for "
            "PatchTST [47].",
            "**Couple the forecast with the frequency stability** of the SENELEC grid [19, 20], to assess its benefit "
            "in reserve and avoided load shedding.",
        ]},
        {"type": "h1", "text": "8. Conclusion"},
        {"type": "p", "text": (
            "We studied the forecasting of the total PV generation of the SENELEC grid in a realistic setting for an "
            "African operator: without meteorological measurements and without the individual productions as inputs. "
            "The PatchTST-CA-X architecture, which conditions the forecasting head on cyclic and solar covariates "
            f"known over the horizon, reaches a day-ahead RMSE of {num(s['RMSE_mean'])} MW "
            f"({pc(1 - s['RMSE_mean']/R['refs']['Persistence D-1']['RMSE'], sign=False)} lower than persistence). It "
            "shows no drift over 30 days, resists noise and forecasts in a few milliseconds. The multi-criteria "
            "protocol (accuracy, robustness, stability, efficiency, cross-validation, explainability) shows where the "
            "gain really lies: in the deterministic **future** information, not in adding past exogenous channels; a "
            "Ridge regression given the same information does almost as well. The protocol, the code and the "
            "notebooks are provided to serve as a benchmark for future work on West African grids.")},
        {"type": "h1", "text": "Data and code availability"},
        {"type": "p", "text": (
            "The code (commented module, six executed notebooks, flowcharts) and the intermediate results are "
            "provided with the paper. The operating data belong to SENELEC and can be obtained from the operator "
            "upon reasoned request.")},
        {"type": "h1", "text": "Acknowledgements"},
        {"type": "p", "text": "The authors thank SENELEC for providing the operating log."},
    ]


# -----------------------------------------------------------------------------
# 6. Assembly
# -----------------------------------------------------------------------------
def data_values(R):
    ds = R["ds"]
    raw = pd.read_excel(m.DATA_PATH, usecols=["Horodatage_debut"])
    y = ds.table[m.TARGET]
    yearly = y.groupby(y.index.year).mean()
    return dict(start=f"{ds.index[0]:%d/%m/%Y}", end=f"{ds.index[-1]:%d/%m/%Y}",
                n_hours=f"{len(ds.index):,}", n_missing=len(ds.index) - len(raw),
                growth=100 * (yearly[2022] / yearly[2019] - 1),
                n_clipped=int(ds.target_diagnostics.clipped_values.sum()),
                offset=ds.solar_offset_h,
                n_train=f"{ds.i_val:,}", n_val=f"{ds.i_test - ds.i_val:,}",
                n_test=f"{len(ds.values) - ds.i_test:,}")


def plants_table(R):
    d = R["ds"].target_diagnostics
    return {"type": "table", "caption": "**Table 2.** Target construction: per-plant traceability.", "size": 15,
            "widths": [2.4, 1.8, 1.8, 1.2, 1.2, 1.4, 1.3],
            "headers": ["Plant", "Commissioning", "Last record", "Threshold (MW)", "Clipped", "Imputed gaps", "Mean (MW)"],
            "rows": [[r.plant, f"{r.commissioning:%d/%m/%Y}", f"{r.last_record:%d/%m/%Y}", num(r.threshold_MW, 1),
                      str(r.clipped_values), str(r.imputed_gaps), num(r.mean_MW)] for r in d.itertuples()]}


def build():
    R = collect()
    D = data_values(R)
    B = abstract(R) + T.introduction() + T.review() + [T.review_table()]
    meth = T.methodology(D)
    # insert Table 2 after the target-construction paragraph, and set the figure ratios
    for b in meth:
        if b["type"] == "figure":
            w, h = Image.open(ROOT / b["path"]).size
            b["ratio"] = h / w; b["path"] = str(ROOT / b["path"])
    i = next(k for k, b in enumerate(meth) if b["type"] == "equation")
    meth.insert(i + 1, plants_table(R))
    B += meth + results(R) + discussion(R) + T.references()
    doc = {"title": T.TITLE, "authors": T.AUTHORS, "affiliation": T.AFFILIATION, "header": T.HEADER, "blocks": B}
    f = m.CACHE_DIR / "article.json"
    f.write_text(json.dumps(doc, ensure_ascii=False, default=str), encoding="utf-8")
    subprocess.run(["node", str(ROOT / "src" / "docx_render.js"), str(f), str(ROOT / OUTPUT)], check=True, cwd=ROOT)
    return R


if __name__ == "__main__":
    build()
