# -*- coding: utf-8 -*-
"""
nb05_final.py — build and execute notebooks/05_Final_Model.ipynb.

Heavy steps are cached (cache/final/) by `precompute()`: Ridge, noise curve,
TimeSeriesSplit cross-validation, "capacity index" ablation, XAI. The
Markdown conclusions are written from the actual results.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
import analysis as an  # noqa: E402
import patchtst_pv as m  # noqa: E402
from nb_tools import build, code_header  # noqa: E402

STEPS = ["config1_patch", "config2_history", "config3_channels", "config4_horizon"]


# -----------------------------------------------------------------------------
# 1. Helpers: final configuration, model loading, delivered seed
# -----------------------------------------------------------------------------
def cfg_from_name(name: str) -> m.PatchTSTConfig:
    """'PatchTST-CAX_P12_L168_H24' → PatchTSTConfig."""
    mode = name.split("-")[1].split("_")[0]
    p = name.split("_")
    return m.PatchTSTConfig(P=int(p[1][1:]), L=int(p[2][1:]), H=int(p[3][1:]), channel_mode=mode[:2],
                            known_future="X" in mode[2:], physical_gate="g" in mode[2:])


def final_model() -> m.PatchTSTConfig:
    """Architecture selected in configurations 1 to 3, evaluated at the
    operational day-ahead horizon (24 h) — the horizon is not "selected" by
    the score (comparing horizons means comparing different tasks)."""
    sel = json.loads((m.CACHE_DIR / "selection.json").read_text())
    return cfg_from_name(sel["config3_channels"])


def load_model(cfg, seed):
    model = m.PatchTST(cfg)
    model.load_state_dict(torch.load(m.CACHE_DIR / cfg.name() / f"weights_seed{seed}.pt"))
    return model.eval()


def validation_seed(cfg):
    """Seed of the delivered model: lowest VALIDATION loss (not test loss,
    so as not to select on the Test set)."""
    losses = {g: pd.read_csv(m.CACHE_DIR / cfg.name() / f"history_seed{g}.csv").val_loss.min()
              for g in m.SEEDS}
    return min(losses, key=losses.get)


# -----------------------------------------------------------------------------
# 2. Pre-computation of the final analyses (cached in cache/final/)
# -----------------------------------------------------------------------------
def precompute():
    ds = m.build_dataset()
    cfg = final_model()
    res = m.evaluate_configuration(cfg, ds)
    g = validation_seed(cfg)
    model = load_model(cfg, g)

    def ridge():
        yp, yv, o, a = an.multi_output_ridge(ds, cfg.L, cfg.H)
        np.save(m.CACHE_DIR / "final" / "ridge_pred.npy", yp.astype(np.float32))
        return pd.DataFrame([dict(alpha=a, **m.metrics(yv, yp))])
    an.cached("ridge.csv", ridge)
    an.cached("noise.csv", lambda: an.noise_curve(cfg, ds, cfg.name()))
    an.cached("cross_validation.csv", lambda: an.cross_validation(cfg, ds))
    an.cached("capacity_index.csv", lambda: an.evaluate_capacity_index(cfg, ds))
    an.cached("occlusion.csv", lambda: an.occlusion_importance(model, ds)[0])
    an.cached("integrated_gradients.csv",
              lambda: an.integrated_gradients(model, ds).rename("share").rename_axis("input").reset_index())
    att = an.channel_attention_weights(model, ds) if not (m.CACHE_DIR / "final" / "attention.csv").exists() else None
    if att is not None:
        an.cached("attention.csv", lambda: att.rename("weight").rename_axis("channel").reset_index())
    return ds, cfg, res, g


def conclusion() -> str:
    """Final conclusion written from the cached results."""
    ds, cfg, res, g = precompute()
    s = res["summary"]
    preds, y_true, orig = an.load_predictions(cfg.name())
    rp = m.metrics(y_true, m.daily_persistence(ds, orig, cfg.H))["RMSE"]
    rmean = m.metrics(y_true, m.seven_day_mean(ds, orig, cfg.H))["RMSE"]
    F = m.CACHE_DIR / "final"
    rr = pd.read_csv(F / "ridge.csv").iloc[0]
    cv = pd.read_csv(F / "cross_validation.csv")
    kc = pd.read_csv(F / "capacity_index.csv")
    nz = pd.read_csv(F / "noise.csv").groupby("level").degradation.mean()
    occ = pd.read_csv(F / "occlusion.csv").set_index("group").RMSE_increase
    sc = m.decision_score(s)
    g_rep = an.representative_seed(res)
    y_ridge = np.load(F / "ridge_pred.npy")
    p_ridge = m.diebold_mariano_test(preds[m.SEEDS.index(g_rep)] - y_true, y_ridge - y_true, h=cfg.H)[1]
    return f"""
## Conclusion — final model `{cfg.name()}`

**Accuracy (40 %).** RMSE = {s['RMSE_mean']:.2f} ± {s['RMSE_std']:.2f} MW, nRMSE = {s['nRMSE_mean']:.3f},
R² = {s['R2_mean']:.3f} over the 7 Test months. Baselines at the same horizon: persistence D-1 {rp:.2f} MW
({100*(1-s['RMSE_mean']/rp):+.1f} %), 7-day mean {rmean:.2f} MW ({100*(1-s['RMSE_mean']/rmean):+.1f} %),
multi-output Ridge {rr.RMSE:.2f} MW ({100*(1-s['RMSE_mean']/rr.RMSE):+.1f} %). **The gap with Ridge is not significant**
(Diebold-Mariano, p = {p_ridge:.2f}): without meteorology, predictability is mostly linear in the recent history and the
future solar geometry, information that Ridge also has.

**Robustness (30 %).** Between-seed standard deviation {s['RMSE_std']:.3f} MW; degradation under 5 % noise on the exogenous
channels {100*nz.get(0.05, np.nan):+.2f} % (20 % noise: {100*nz.get(0.2, np.nan):+.2f} %), far below the 15 % rejection threshold.
TimeSeriesSplit cross-validation (5 folds): mean gain vs persistence {100*cv.gain_vs_persistence.mean():+.1f} %
(min {100*cv.gain_vs_persistence.min():+.1f} %, max {100*cv.gain_vs_persistence.max():+.1f} %).

**Long-term stability (30 %).** Smoothed drift over 30 days without re-training = {s['drift_7d_mean']:.2f}
(target < 1.2); raw D30/D1 ratio = {s['drift_D30_D1_mean']:.2f}.

**Efficiency.** {s['inference_ms_mean']:.1f} ms per forecast, {s['size_MB_mean']:.2f} MB, {int(s['n_parameters_mean']):,} parameters.

**Decision score** = {sc['score']:.2f} (efficiency bonus {sc['bonus']:.1f}).

**"Capacity index" ablation.** RMSE = {kc.RMSE.mean():.2f} ± {kc.RMSE.std():.2f} MW versus {s['RMSE_mean']:.2f} MW
with RevIN normalisation alone, i.e. {100*(kc.RMSE.mean()/s['RMSE_mean']-1):+.1f} %.

**Explainability.** Masking the target history increases the RMSE by {100*occ.get('Target (PV history)', np.nan):+.0f} %,
solar geometry by {100*occ.get('Solar geometry', np.nan):+.0f} %, the hourly encoding by {100*occ.get('Cyclic hour', np.nan):+.0f} %,
the grid context by {100*occ.get('Grid', np.nan):+.1f} % and the day of week (negative control) by {100*occ.get('Cyclic week', np.nan):+.1f} %.

The delivered model (`models/best_patchtst.pt`) corresponds to seed {g}, chosen on the **validation** loss.
"""


# -----------------------------------------------------------------------------
# 3. Notebook cells
# -----------------------------------------------------------------------------
CELLS = [
("md", """
# 05 — Final model: synthesis of the configurations and in-depth evaluation

This notebook gathers:
1. the **best model of each configuration** (P, L, channels, horizon);
2. the **final model** (architecture selected in configurations 1 to 3, operational day-ahead horizon = 24 h);
3. its comparison with the baselines (persistence, 7-day mean, **multi-output Ridge**, univariate PatchTST) with Diebold-Mariano tests;
4. observed/predicted plots, residuals and the seed **rosary**;
5. robustness (noise curve), stability (30-day rolling, monthly RMSE), **TimeSeriesSplit cross-validation**;
6. the "capacity index" ablation;
7. explainability (occlusion, integrated gradients, cross-channel attention);
8. efficiency, the final score and saving `models/best_patchtst.pt`.

> **Why is the final horizon not chosen by the score?** Comparing horizons means comparing different tasks:
> the 1 h error is mechanically lower than the 24 h error. The horizon is set by the use case:
> day-ahead dispatch planning. Configuration 4 describes how performance evolves with the horizon.
"""),
("code", code_header("patchtst_pv") + "import analysis as an, json, torch, dataclasses\n"
         "from nb05_final import cfg_from_name, final_model, validation_seed, load_model, STEPS"),
("md", "## 1. Best model of each configuration"),
("code", """
ds = m.build_dataset()
selection = json.loads((m.CACHE_DIR / "selection.json").read_text())
rows = []
for step in STEPS:
    r = m.evaluate_configuration(cfg_from_name(selection[step]), ds)
    t = m.comparison_table([r]).iloc[0]
    rows.append(dict(step=step, selected=selection[step], RMSE=t.RMSE, RMSE_std=t.RMSE_std, nRMSE=t.nRMSE,
                     R2=t.R2, noise_degradation=t.noise_degradation, drift_7d=t.drift_7d, inference_ms=t.inference_ms,
                     score=t.score))
best = pd.DataFrame(rows); best.to_csv(m.TABLES_DIR / "best_per_configuration.csv", index=False)
best.round(4)
"""),
("md", "## 2. Final model and comparison with the baselines"),
("code", """
cfg = final_model()
res = m.evaluate_configuration(cfg, ds)
preds, y_true, orig = an.load_predictions(cfg.name())
print("Final model:", cfg.name(), "|", cfg)
ridge = pd.read_csv(m.CACHE_DIR / "final" / "ridge.csv")
y_ridge = np.load(m.CACHE_DIR / "final" / "ridge_pred.npy")
res_ci = m.evaluate_configuration(dataclasses.replace(cfg, channel_mode="CI", known_future=False, physical_gate=False), ds)
refs = {"Persistence D-1": m.daily_persistence(ds, orig, cfg.H), "7-day mean": m.seven_day_mean(ds, orig, cfg.H),
        "Multi-output Ridge": y_ridge}
rows = [dict(model=k, **m.metrics(y_true, v)) for k, v in refs.items()]
for r in [res_ci, res]:
    s = r["summary"]
    rows.append(dict(model=r["name"] + " (mean 5 seeds)", RMSE=s["RMSE_mean"], MAE=s["MAE_mean"], nRMSE=s["nRMSE_mean"],
                     R2=s["R2_mean"], MBE=s["MBE_mean"]))
comparison = pd.DataFrame(rows); comparison.to_csv(m.TABLES_DIR / "baseline_comparison.csv", index=False)
comparison.round(4)
"""),
("code", """
# Diebold-Mariano tests (representative seed of the final model against each baseline)
g_rep = an.representative_seed(res)
e_mod = preds[m.SEEDS.index(g_rep)] - y_true
dm = []
for k, v in refs.items():
    stat, p = m.diebold_mariano_test(e_mod, v - y_true, h=cfg.H)
    dm.append(dict(baseline=k, DM=stat, p_value=p, skill_score=1 - np.sqrt((e_mod**2).mean()) / np.sqrt(((v - y_true)**2).mean())))
pd.DataFrame(dm).round(4)
"""),
("md", "## 3. Seed rosary, observed vs predicted series and residuals"),
("code", """
fig, ax = plt.subplots(figsize=(7.5, 3.4))
groups = {an.label(r["name"]): [g["RMSE"] for g in r["per_seed"]] for r in [res_ci, res]}
ax.boxplot(list(groups.values()), widths=0.4, showfliers=False, medianprops=dict(color=fs.COLORS["model"]))
for i, d in enumerate(groups.values()):
    ax.scatter(np.full(len(d), i + 1) + np.linspace(-0.07, 0.07, len(d)), d, s=16, color=fs.COLORS["observed"], zorder=3)
styles = {"Persistence D-1": ("--", fs.COLORS["persistence"]), "7-day mean": (":", fs.COLORS["mean"]),
          "Multi-output Ridge": ("-.", "#27ae60")}
for k, v in refs.items():
    r = m.metrics(y_true, v)["RMSE"]; ax.axhline(r, ls=styles[k][0], color=styles[k][1], lw=1.1, label=f"{k} ({r:.2f})")
ax.set_xticks(range(1, len(groups) + 1), list(groups.keys())); ax.set_ylabel("Test RMSE (MW)")
ax.legend(fontsize=7, loc="upper right"); ax.set_title("Rosary of the 5 seeds — final model vs univariate and baselines")
fs.save(fig, "fig_final_rosary")
an.fig_observed_vs_predicted(res, ds, "fig_final_observed_predicted")
an.fig_observed_vs_predicted(res, ds, "fig_final_observed_predicted_dry_season", days=10, start_day=150)
an.fig_residuals(res, ds, "fig_final_residuals")
"""),
("md", "## 4. Robustness: degradation as a function of the noise level on the exogenous channels"),
("code", """
noise = pd.read_csv(m.CACHE_DIR / "final" / "noise.csv")
b = noise.groupby("level").degradation.agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.errorbar(100 * b.index, 100 * b["mean"], yerr=100 * b["std"], marker="o", color=fs.COLORS["model"], capsize=3)
ax.axhline(15, color="k", ls="--", lw=0.9, label="rejection threshold (15 %)")
ax.axvline(5, color=fs.COLORS["persistence"], ls=":", lw=0.9, label="projet.md test (5 %)")
ax.set_xlabel("noise standard deviation (% of the Train std)"); ax.set_ylabel("RMSE increase (%)"); ax.legend(fontsize=7)
ax.set_title("Robustness to noise on the covariates")
fs.save(fig, "fig_final_noise")
(100 * b).round(3)
"""),
("md", "## 5. Long-term stability: 30-day rolling and monthly RMSE over the whole Test set"),
("code", """
fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
rd = np.array([g["daily_rmse"] for g in res["per_seed"]])
k30 = orig < ds.i_test + 24 * 30
rd_p = m.daily_rmse(y_true[k30], refs["Persistence D-1"][k30], orig[k30], ds)
days = np.arange(1, rd.shape[1] + 1)
ax[0].fill_between(days, rd.min(0), rd.max(0), color=fs.COLORS["model"], alpha=0.2, label="min–max 5 seeds")
ax[0].plot(days, rd.mean(0), color=fs.COLORS["model"], marker=".", label=an.label(cfg.name()))
ax[0].plot(days, rd_p.values, color=fs.COLORS["persistence"], ls="--", label="persistence D-1")
ax[0].set_xlabel("day (no re-training)"); ax[0].set_ylabel("daily RMSE (MW)"); ax[0].legend(fontsize=7)
ax[0].set_title("30-day rolling forecast")
mm = {k: an.monthly_rmse(y_true, v, orig, ds) for k, v in refs.items()}
mm[an.label(cfg.name())] = an.monthly_rmse(y_true, preds[m.SEEDS.index(g_rep)], orig, ds)
colors = {"Persistence D-1": fs.COLORS["persistence"], "7-day mean": fs.COLORS["mean"],
          "Multi-output Ridge": "#27ae60", an.label(cfg.name()): fs.COLORS["model"]}
for k, s in mm.items():
    ax[1].plot(range(len(s)), s.values, marker="o", label=k, color=colors[k],
               ls="--" if k == "Persistence D-1" else "-")
ax[1].set_xticks(range(len(s)), s.index, rotation=30); ax[1].set_ylabel("monthly RMSE (MW)"); ax[1].legend(fontsize=7)
ax[1].set_title("Stability over the 7 Test months")
fs.save(fig, "fig_final_stability")
pd.DataFrame(mm).round(3)
"""),
("md", """
## 6. Temporal cross-validation (TimeSeriesSplit, 5 expanding-window folds)

For each fold, the normalisation is **re-fitted** on the Train part of the fold (R2); 15 % of the training window
is used for early stopping; the next block is the test. The folds cover 2019–2023: this checks that performance
does not depend on a particular period.
"""),
("code", """
cv = pd.read_csv(m.CACHE_DIR / "final" / "cross_validation.csv")
cv.to_csv(m.TABLES_DIR / "cross_validation.csv", index=False)
display(cv.round(4))
print(f"Mean gain vs persistence: {100*cv.gain_vs_persistence.mean():+.1f} % ± {100*cv.gain_vs_persistence.std():.1f} %")
"""),
("md", """
## 7. Ablation: causal "capacity index"

Variant: the model learns k = PV_total / C(t), where C(t) is the 99 % quantile of the daily maxima of the **previous**
30 days (apparent capacity, updated causally after each commissioning). The forecast is converted back to MW by
multiplying by C(t). Same configuration, 5 seeds.
"""),
("code", """
kc = pd.read_csv(m.CACHE_DIR / "final" / "capacity_index.csv")
s = res["summary"]
pd.DataFrame([dict(variant="RevIN only (final model)", RMSE=s["RMSE_mean"], RMSE_std=s["RMSE_std"], drift_7d=s["drift_7d_mean"]),
              dict(variant="Capacity index + RevIN", RMSE=kc.RMSE.mean(), RMSE_std=kc.RMSE.std(), drift_7d=kc.drift_7d.mean())]).round(4)
"""),
("md", "## 8. Explainability"),
("code", """
occ = pd.read_csv(m.CACHE_DIR / "final" / "occlusion.csv")
ig = pd.read_csv(m.CACHE_DIR / "final" / "integrated_gradients.csv")
f_att = m.CACHE_DIR / "final" / "attention.csv"
att = pd.read_csv(f_att) if f_att.exists() else None
fig, ax = plt.subplots(1, 3 if att is not None else 2, figsize=(14, 4.2), gridspec_kw={"wspace": 0.75})
o = occ.sort_values("RMSE_increase")
ax[0].barh(o.group, 100 * o.RMSE_increase, color=[fs.COLORS["model"] if v > 0.05 else fs.COLORS["persistence"] for v in o.RMSE_increase])
ax[0].set_xlabel("RMSE increase when the group is masked (%)"); ax[0].set_title("Occlusion importance")
i = ig.sort_values("share").tail(12)
ax[1].barh(i.input, 100 * i.share, color=fs.COLORS["mean"]); ax[1].set_xlabel("share of the attribution (%)")
ax[1].set_title("Integrated gradients (top 12 inputs)")
if att is not None:
    ax[2].bar(range(len(att)), att.weight, color=fs.COLORS["observed"]); ax[2].axhline(1 / len(att), ls="--", color=fs.COLORS["model"], lw=0.9, label="uniform")
    ax[2].set_xticks(range(len(att)), att.channel, rotation=60, fontsize=7); ax[2].legend(fontsize=7)
    ax[2].set_title("Cross-channel attention (target → channels)")
fs.save(fig, "fig_final_xai")
display(occ.round(4)); display(ig.sort_values("share", ascending=False).round(4))
"""),
("md", "## 9. Efficiency, final score and saving the model"),
("code", """
s = res["summary"]
eff = pd.DataFrame([dict(inference_ms=s["inference_ms_mean"], size_MB=s["size_MB_mean"], n_parameters=int(s["n_parameters_mean"]),
                         training_duration_s=s["duration_s_mean"], epochs=s["epochs_mean"], **m.decision_score(s))])
display(eff.round(3))
g = validation_seed(cfg)
model = load_model(cfg, g)
(ROOT / "models").mkdir(exist_ok=True)
torch.save({"state": model.state_dict(), "config": dataclasses.asdict(cfg), "seed": g, "channels": m.CHANNELS,
            "train_mean": ds.mean, "train_std": ds.std,
            "solar_offset_h": ds.solar_offset_h}, ROOT / "models" / "best_patchtst.pt")
size = (ROOT / "models" / "best_patchtst.pt").stat().st_size / 2**20
print(f"Model saved: models/best_patchtst.pt (seed {g}, chosen on the validation loss) — {size:.2f} MB")
# check: reloading gives identical forecasts
check = torch.load(ROOT / "models" / "best_patchtst.pt", weights_only=False)
model2 = m.PatchTST(m.PatchTSTConfig(**check["config"])); model2.load_state_dict(check["state"]); model2.eval()
o_ = orig[:50]
assert np.allclose(m.predict(model, ds, o_)[0], m.predict(model2, ds, o_)[0])
print("Reload verified: identical forecasts.")
"""),
]


def cells():
    return CELLS + [("md", conclusion())]


if __name__ == "__main__":
    print(build("05_Final_Model", cells()))
