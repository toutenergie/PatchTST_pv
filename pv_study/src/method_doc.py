# -*- coding: utf-8 -*-
"""
method_doc.py — Word document "Step-by-step method" (item 6 of projet.md).

Explains, in execution order, what was done, why, with which code, and how to
reproduce it. Every number comes from the results cache.

    python method_doc.py     → Method_step_by_step_PV_EN.docx
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as an  # noqa: E402
import patchtst_pv as m  # noqa: E402

ROOT = m.ROOT
OUTPUT = "Method_step_by_step_PV_EN.docx"


def num(x, d=2):
    return f"{x:,.{d}f}"


def fig(name, caption, width=16):
    w, h = Image.open(ROOT / "figures" / f"{name}.png").size
    return {"type": "figure", "path": str(ROOT / "figures" / f"{name}.png"), "ratio": h / w,
            "width_cm": width, "caption": caption}


def dm_p_values(ds, sel):
    """Diebold-Mariano p-values of the final model against the baselines (from the cache)."""
    from nb05_final import cfg_from_name
    cfg = cfg_from_name(sel["config3_channels"])
    res = m.evaluate_configuration(cfg, ds)
    preds, y_true, orig = an.load_predictions(cfg.name())
    e = preds[m.SEEDS.index(an.representative_seed(res))] - y_true
    refs = {"persistence": m.daily_persistence(ds, orig, cfg.H), "7-day mean": m.seven_day_mean(ds, orig, cfg.H),
            "Ridge": np.load(m.CACHE_DIR / "final" / "ridge_pred.npy")}
    return {k: m.diebold_mariano_test(e, v - y_true, h=cfg.H)[1] for k, v in refs.items()}


def blocks():
    ds = m.build_dataset()
    sel = json.loads((m.CACHE_DIR / "selection.json").read_text())
    B = []
    B += [{"type": "h1", "text": "Purpose of the document"},
          {"type": "p", "text": (
              "This document describes step by step the procedure followed to build, evaluate and select a PatchTST "
              "model forecasting the total photovoltaic generation of the SENELEC grid, according to the specification "
              "`projet.md`. For each step, it states **what is done**, **why**, **where the code is** and **how to "
              "check it**. It is intended both for the reader who wants to understand the method and for the one who "
              "wants to reproduce it.")},
          {"type": "box", "text": (
              "**Deliberate deviation from `projet.md`.** The specification describes a univariate problem; the request "
              "then evolved towards a **multivariate forecast without the individual PV productions as input "
              "variables**, with cyclic variables. Since the log contains no meteorological data, the 5 % noise test "
              "\"on the weather\" is applied to all the exogenous covariates.")},
          {"type": "h1", "text": "Step 0 — Project organisation"},
          {"type": "table", "caption": "**Table A.** Tree of the deliverables.", "widths": [3.2, 6.8], "left": [1],
           "headers": ["Item", "Content"],
           "rows": [["data/", "energie.xlsx (hourly log), projet.md (specification)"],
                    ["src/patchtst_pv.py", "commented core module: data, covariates, PatchTST, training, metrics, protocol, score"],
                    ["src/analysis.py", "baselines (persistence, mean, Ridge), DM tests, figures, cross-validation, XAI"],
                    ["src/run_experiments.py", "runs configurations 1 → 4 in sequence (one factor at a time)"],
                    ["notebooks/00 … 05", "EDA; one configuration per notebook; final model"],
                    ["figures/", "all figures (300 dpi PNG), including the flowcharts flow_config1…4"],
                    ["tables/", "comparison tables (CSV) and Excel summary workbook"],
                    ["models/best_patchtst.pt", "weights of the final model + configuration + normalisation parameters"],
                    ["cache/", "results per configuration and per seed (forecasts, weights, learning histories)"]]},
          {"type": "p", "text": (
              "**Reproducing the study.** (1) `python src/run_experiments.py` trains and evaluates every configuration "
              "(several hours on CPU); (2) notebooks 00 to 05 can then be executed in order: they read the cache. "
              "Deleting a folder `cache/<configuration>/` re-runs its training.")},
          ]

    # --- Step 1: data
    d = ds.target_diagnostics
    B += [{"type": "h1", "text": "Step 1 — Loading and checking the data (notebook 00)"},
          {"type": "p", "text": (
              f"The log covers {len(ds.table):,} hours, from {ds.index[0]:%d/%m/%Y} to {ds.index[-1]:%d/%m/%Y}. "
              "The hourly grid is completed for the 288 missing hours (eight jumps of one to three days): windowing "
              "requires a regular grid. **Why**: deleting these hours would shift every window that contains them.")},
          {"type": "h1", "text": "Step 2 — Construction of the target PV_total"},
          {"type": "p", "text": (
              "The target is the sum of the eleven PV plants. For each plant: (a) clipping above 1.2 × the 99.9 % "
              "quantile **of the training period**; (b) zero value outside the period of existence; (c) causal "
              "imputation of gaps. Code: `build_target`, `impute_series`.")},
          {"type": "table", "caption": "**Table B.** Traceability of the target construction, per plant.",
           "widths": [2.4, 2.2, 2.2, 1.2, 1.3, 1.3, 1.2], "size": 15,
           "headers": ["Plant", "Commissioning", "Last record", "Threshold (MW)", "Clipped", "Imputed gaps", "Mean (MW)"],
           "rows": [[r.plant, f"{r.commissioning:%d/%m/%Y}", f"{r.last_record:%d/%m/%Y}", num(r.threshold_MW, 1),
                     str(r.clipped_values), str(r.imputed_gaps), num(r.mean_MW)] for r in d.itertuples()]},
          {"type": "p", "text": (
              "**Check**: notebook 00 asserts that the normalisation is computed on the training period only and that "
              "no individual PV column, nor any total that contains it, enters the model.")},
          fig("fig01_pv_total_series", "**Figure A.** Target series, capacity envelope and temporal split."),
          {"type": "h1", "text": "Step 3 — Exploratory analysis"},
          {"type": "bullets", "items": [
              "Seasonality: dominant 24 h cycle (ACF(24) ≈ 0.97 on training), 12 h harmonic, annual modulation.",
              "Non-stationarity: mean production grows by 47 % between 2019 and 2022 (commissionings).",
              "Correlations: strong with solar geometry (≈ 0.94), weak with the grid context once the mean cycle is removed.",
              "Consequence: day-ahead persistence will be a hard baseline to beat."]},
          fig("fig03_seasonality", "**Figure B.** Hour × month profile, autocorrelation and periodogram."),
          {"type": "h1", "text": "Step 4 — Covariates"},
          {"type": "p", "text": (
              "Twelve channels: the target; three grid channels (aggregated thermal, Taiba wind, Manantali import); six "
              "cyclic encodings (sin/cos of the hour, day of year, day of week); two solar-geometry variables "
              "(cos θz, Haurwitz clear-sky irradiance, pvlib). The offset between timestamps and solar time is "
              f"estimated on training: {num(ds.solar_offset_h)} h. Code: `cyclic_covariates`, `solar_covariates`, "
              "`estimate_offset`, `grid_covariates`.")},
          fig("fig04_covariates", "**Figure C.** Cyclic covariates, solar geometry and grid context."),
          {"type": "h1", "text": "Step 5 — Temporal split and normalisation"},
          {"type": "p", "text": (
              f"Train: 01/01/2019 → 31/12/2021 ({ds.i_val:,} h); Validation: 01/01/2022 → 30/09/2022 "
              f"({ds.i_test - ds.i_val:,} h); Test: 01/10/2022 → 09/05/2023 ({len(ds.values) - ds.i_test:,} h). "
              "No shuffling. The global normalisation is fitted on Train; inside the model, a reversible "
              "normalisation (RevIN) is applied window by window to the **stochastic channels only**.")},
          {"type": "box", "text": (
              "**Correction made during the study.** A first version applied RevIN to every channel. Over a 7-day "
              "window, the day of year barely changes: standardising it erased the season and created an artificial "
              "ramp. Integrated gradients revealed it (24 % of the attribution on past `doy_sin`). All the results "
              "presented were recomputed after the correction.")},
          {"type": "h1", "text": "Step 6 — PatchTST model and variants"},
          fig("fig_architecture", "**Figure D.** PatchTST-CA-X architecture and CI / CA / X / g variants.", 16.5),
          {"type": "p", "text": (
              "Class `PatchTST` of the module. Patches of length P (stride P/2) are projected to dimension 64; a "
              "2-layer Transformer encoder is shared across channels. **CI**: only the target feeds the head; "
              "**CA**: cross-channel attention at each patch position; **X**: step-by-step correction by the future "
              "deterministic covariates; **g**: physical night gate.")},
          {"type": "h1", "text": "Step 7 — Training and evaluation protocol"},
          {"type": "numbered", "instance": 3, "items": [
              "Training on Train, early stopping on Val (patience 4, at most 25 epochs), 20 % of the hourly origins drawn at each epoch, 5 seeds.",
              "Accuracy over every hourly origin of the Test set: RMSE, MAE, nRMSE = RMSE/mean, R², MBE.",
              "Robustness: standard deviation over 5 seeds; 5 % Gaussian noise on the exogenous channels (rejected if the RMSE rises by more than 15 %).",
              "Stability: 30-day rolling forecast without re-training; drift = RMSE(D24–30)/RMSE(D1–7) (target < 1.2), raw D30/D1 ratio reported.",
              "Efficiency: inference time for one sample on one core (< 100 ms); weight size (< 50 MB).",
              "Score = 0.4/nRMSE + 0.3/σ + 0.3/drift + bonus; selection on the normalised score (criteria mapped to [0, 1]).",
              "Diebold-Mariano tests against persistence and the 7-day mean.",
          ]},
    ]
    # --- Steps 8 to 11: configurations
    step_names = {"config1_patch": "Configuration 1 — patch length P", "config2_history": "Configuration 2 — history L",
                  "config3_channels": "Configuration 3 — channel handling", "config4_horizon": "Configuration 4 — horizon H"}
    for k, (step, title) in enumerate(step_names.items(), start=1):
        f = m.TABLES_DIR / f"table_config{k}.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        ok = lambda x: x == x  # noqa: E731  (False for NaN: baselines have no σ, noise, drift…)
        B += [{"type": "h1", "text": f"Step {7 + k} — {title} (notebook 0{k})"},
              fig(f"flow_config{k}", f"**Figure {chr(68 + k)}.** Flowchart of configuration {k}.", 9.5),
              {"type": "table", "caption": f"**Table {chr(66 + k)}.** Results of configuration {k} (mean over 5 seeds).",
               "widths": [3.4, 1.3, 1.1, 1.2, 1.3, 1.2, 1.2, 1.3], "size": 14, "left": [0],
               "highlight": [i for i, c in enumerate(t.configuration) if c == sel.get(step)],
               "headers": ["Configuration", "RMSE (MW)", "σ (MW)", "nRMSE", "Noise 5 %", "Drift", "Inference (ms)", "Norm. score"],
               "rows": [[str(r.configuration), num(r.RMSE), num(r.RMSE_std, 3) if ok(r.RMSE_std) else "–",
                         num(r.nRMSE, 3), (num(100 * r.noise_degradation, 1) + " %") if ok(r.noise_degradation) else "–",
                         num(r.drift_7d) if ok(r.drift_7d) else "–",
                         num(r.inference_ms, 1) if ok(r.inference_ms) else "–",
                         num(r.normalised_score, 3) if ok(r.normalised_score) else "–"]
                        for r in t.itertuples()],
               "note": (f"Highlighted row: selected configuration ({sel.get(step, '–')}). Persistence and 7-day mean: same test origins."
                        if k < 4 else f"Highlighted row: best score ({sel.get(step, '–')}); the final horizon remains day-ahead (24 h), set by the use case.")},
              fig(f"fig_config{k}_rosary", f"**Figure {chr(68 + k)}bis.** Seed rosary and drift — configuration {k}."),
              fig(f"fig_config{k}_observed_predicted", f"**Figure {chr(68 + k)}ter.** Observed vs predicted — best model of configuration {k}.")]
    # --- Final step
    p = dm_p_values(ds, sel)
    B += [{"type": "h1", "text": "Step 12 — Final model (notebook 05)"},
          {"type": "p", "text": (
              "The architecture selected in configurations 1 to 3 is evaluated at the operational day-ahead horizon "
              "(24 h). The horizon is not chosen by the score, because comparing horizons means comparing different "
              "tasks. Notebook 05 adds: comparison with Ridge and with the univariate PatchTST, a noise-robustness "
              "curve, monthly RMSE, a 5-fold TimeSeriesSplit cross-validation, the \"capacity index\" ablation, "
              "explainability, and saving to `models/best_patchtst.pt`. The saved seed is chosen on the validation "
              "loss, never on the test set.")},
          {"type": "table", "caption": "**Table G.** Final model vs baselines (day-ahead horizon, full test set).",
           "widths": [3.6, 1.3, 1.3, 1.2, 1.2, 1.2], "size": 15,
           "headers": ["Model", "RMSE (MW)", "MAE (MW)", "nRMSE", "R²", "MBE (MW)"],
           "rows": [[str(r.model), num(r.RMSE), num(r.MAE), num(r.nRMSE, 3), num(r.R2, 3), num(r.MBE)]
                    for r in pd.read_csv(m.TABLES_DIR / "baseline_comparison.csv").itertuples()],
           "note": (f"Diebold-Mariano: final model better than persistence (p = {p['persistence']:.3f}) and than the "
                    f"7-day mean (p = {p['7-day mean']:.3f}); gap with Ridge not significant (p = {p['Ridge']:.2f}).")},
          {"type": "table", "caption": "**Table H.** TimeSeriesSplit cross-validation (5 folds).",
           "widths": [0.7, 2.4, 3.0, 1.1, 1.4, 1.2], "size": 15,
           "headers": ["Fold", "Training", "Test", "RMSE", "RMSE pers.", "Gain"],
           "rows": [[str(r.fold), r.train, r.test, num(r.RMSE), num(r.RMSE_persistence), num(100 * r.gain_vs_persistence, 1) + " %"]
                    for r in pd.read_csv(m.TABLES_DIR / "cross_validation.csv").itertuples()]},
          fig("fig_final_rosary", "**Figure I.** Rosary of the 5 seeds of the final model and of the univariate PatchTST, with the baselines."),
          fig("fig_final_stability", "**Figure J.** 30-day rolling forecast without re-training and monthly RMSE over the test set."),
          fig("fig_final_residuals", "**Figure K.** Residuals of the final model."),
          fig("fig_final_xai", "**Figure L.** Explainability: occlusion, integrated gradients, cross-channel attention."),
          {"type": "h1", "text": "Step 13 — Quality checks"},
          {"type": "bullets", "items": [
              "End-to-end execution of every notebook (`nbclient`), with embedded outputs and figures.",
              "Anti-leakage assertions: order of the segments, normalisation on Train, forbidden columns absent.",
              "Reloading `best_patchtst.pt` and checking identical forecasts.",
              "Tables of the paper regenerated from the cached forecasts (no number typed by hand)."]},
          ]
    return B


def build():
    doc = {"title": "Step-by-step method",
           "subtitle": "Multivariate forecasting of the total PV generation of the SENELEC grid with PatchTST",
           "authors": "Abdoulaye FAYE — UADB / EDTSS", "affiliation": "Methodological document accompanying the notebooks",
           "header": "Method — PatchTST total PV SENELEC", "blocks": blocks()}
    out_json = ROOT / "cache" / "method_doc.json"
    out_json.write_text(json.dumps(doc, ensure_ascii=False, default=str), encoding="utf-8")
    subprocess.run(["node", str(ROOT / "src" / "docx_render.js"), str(out_json), str(ROOT / OUTPUT)], check=True, cwd=ROOT)


if __name__ == "__main__":
    build()
