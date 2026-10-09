# -*- coding: utf-8 -*-
"""
analysis.py — Analysis and figure functions shared by notebooks 01 to 05.

    A. Loading cached results
    B. Naive baselines and Diebold-Mariano tests
    C. Figures: seed "rosary" plots, learning curves, 30-day drift,
       observed vs predicted, residuals
    D. Linear baseline: multi-output Ridge
    E. Temporal cross-validation (TimeSeriesSplit) of the final configuration
    F. Robustness: degradation curve versus noise level
    G. Explainability (XAI)
    H. "Capacity index" variant
    I. Caching of the heavy analyses of the final model
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_style as fs
import patchtst_pv as m


# =============================================================================
# A. Loading cached results
# =============================================================================

def folder(name):
    return m.CACHE_DIR / name


def load_predictions(name: str):
    """Forecasts of the 5 seeds [g, n, H], observations [n, H], origins [n]."""
    d = folder(name)
    preds = np.stack([np.load(d / f"pred_seed{g}.npy") for g in m.SEEDS])
    return preds, np.load(d / "true.npy"), np.load(d / "origins.npy")


def representative_seed(res: dict) -> int:
    """Seed whose RMSE is the median of the 5 (used for the figures)."""
    rm = [(r["RMSE"], r["seed"]) for r in res["per_seed"]]
    return sorted(rm)[len(rm) // 2][1]


def selected_at(step: str):
    """Configuration selected at a step (log written by run_experiments.py)."""
    sel = json.loads((m.CACHE_DIR / "selection.json").read_text())
    return sel.get(step)


# =============================================================================
# B. Naive baselines and statistical tests
# =============================================================================

def baseline_rows(ds, orig, y_true, H) -> pd.DataFrame:
    """Daily persistence and 7-day mean on the same origins."""
    rows = []
    for name, f in [("Persistence D-1", m.daily_persistence), ("7-day mean", m.seven_day_mean)]:
        yp = f(ds, orig, H)
        mt = m.metrics(y_true, yp)
        rows.append(dict(configuration=name, RMSE=mt["RMSE"], RMSE_std=0.0, MAE=mt["MAE"],
                         nRMSE=mt["nRMSE"], R2=mt["R2"], MBE=mt["MBE"]))
    return pd.DataFrame(rows)


def dm_tests(ds, results: list[dict]) -> pd.DataFrame:
    """Diebold-Mariano test of each configuration (representative seed) against
    daily persistence and the 7-day mean. Stat < 0: the model is better."""
    rows = []
    for r in results:
        preds, true, orig = load_predictions(r["name"])
        H = true.shape[1]
        g = representative_seed(r)
        e = preds[m.SEEDS.index(g)] - true
        for ref_name, f in [("persistence", m.daily_persistence), ("7-day mean", m.seven_day_mean)]:
            stat, p = m.diebold_mariano_test(e, f(ds, orig, H) - true, h=H)
            ss = 1 - np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean((f(ds, orig, H) - true) ** 2))
            rows.append(dict(configuration=r["name"], reference=ref_name, skill_score=ss,
                             DM=stat, p_value=p))
    return pd.DataFrame(rows)


# =============================================================================
# C. Figures
# =============================================================================

def label(name: str) -> str:
    return name.replace("PatchTST-", "")


def fig_rosary(results, ds, file_name, title):
    """"Rosary" plot (box + points) of the RMSE over the 5 seeds, with the baselines."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    names = [label(r["name"]) for r in results]
    for k, crit in enumerate(["RMSE", "drift_7d"]):
        data = [[g[crit] for g in r["per_seed"]] for r in results]
        ax[k].boxplot(data, widths=0.45, showfliers=False,
                      medianprops=dict(color=fs.COLORS["model"]))
        for i, d in enumerate(data):
            ax[k].scatter(np.full(len(d), i + 1) + np.linspace(-0.08, 0.08, len(d)), d, s=14,
                          color=fs.COLORS["observed"], zorder=3)
        ax[k].set_xticks(range(1, len(names) + 1), names, rotation=20, ha="right")
    _, true, orig = load_predictions(results[0]["name"])
    H = true.shape[1]
    ref = baseline_rows(ds, orig, true, H)
    if len({r["config"]["H"] for r in results}) == 1:   # same horizon: baselines comparable
        for _, l in ref.iterrows():
            ax[0].axhline(l.RMSE, ls="--" if "Pers" in l.configuration else ":", lw=1,
                          color=fs.COLORS["persistence" if "Pers" in l.configuration else "mean"],
                          label=f"{l.configuration} ({l.RMSE:.2f})")
        ax[0].legend(fontsize=7)
    ax[0].set_ylabel("Test RMSE (MW)"); ax[0].set_title("Accuracy and dispersion (5 seeds)")
    ax[1].axhline(1.2, color=fs.COLORS["model"], ls="--", lw=1, label="threshold 1.2")
    ax[1].axhline(1.0, color="k", lw=0.6)
    ax[1].set_ylabel("drift RMSE(D24–30)/RMSE(D1–7)"); ax[1].legend(fontsize=7)
    ax[1].set_title("Stability: drift over 30 days")
    fig.suptitle(title, fontsize=10)
    fs.save(fig, file_name)


def fig_learning_and_drift(results, file_name):
    """Learning curves (representative seed) and daily RMSE over 30 days."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    for r in results:
        g = representative_seed(r)
        h = pd.read_csv(folder(r["name"]) / f"history_seed{g}.csv")
        l, = ax[0].plot(h.epoch, h.val_loss, marker="o", ms=3, label=label(r["name"]))
        ax[0].plot(h.epoch, h.train_loss, ls=":", color=l.get_color())
        rd = np.mean([ps["daily_rmse"] for ps in r["per_seed"]], axis=0)
        ax[1].plot(np.arange(1, len(rd) + 1), rd, marker=".", label=label(r["name"]))
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE (normalised units)")
    ax[0].set_title("Learning (solid: Val, dotted: Train)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("day of the rolling forecast (no re-training)"); ax[1].set_ylabel("daily RMSE (MW)")
    ax[1].set_title("30-day rolling — mean of the 5 seeds"); ax[1].legend(fontsize=7)
    fs.save(fig, file_name)


def continuous_series(pred, true, orig, ds, H):
    """Assemble non-overlapping forecasts issued every H hours (starting from
    an origin at 00:00) into a continuous observed/forecast series."""
    hours = ds.index[orig].hour
    start = int(np.argmax(hours == 0))
    sel = np.arange(start, len(orig), H)
    idx = np.concatenate([np.arange(orig[i], orig[i] + H) for i in sel])
    return (pd.Series(np.concatenate([true[i] for i in sel]), index=ds.index[idx]),
            pd.Series(np.concatenate([pred[i] for i in sel]), index=ds.index[idx]))


def fig_observed_vs_predicted(res, ds, file_name, days=10, start_day=35):
    """Observed vs predicted (representative seed) + scatter plot + persistence."""
    preds, true, orig = load_predictions(res["name"])
    H = true.shape[1]
    g = representative_seed(res)
    yt, yp = continuous_series(preds[m.SEEDS.index(g)], true, orig, ds, H)
    ypers = pd.Series(ds.table[m.TARGET].shift(24).reindex(yt.index).values, index=yt.index)
    t0 = yt.index[0].normalize() + pd.Timedelta(days=start_day)
    f = (yt.index >= t0) & (yt.index < t0 + pd.Timedelta(days=days))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [2.6, 1]})
    ax[0].plot(yt[f].index, yt[f], color=fs.COLORS["observed"], lw=1.3, label="observed")
    ax[0].plot(yp[f].index, yp[f], color=fs.COLORS["model"], lw=1.2, label=f"{label(res['name'])} (seed {g})")
    ax[0].plot(ypers[f].index, ypers[f], color=fs.COLORS["persistence"], lw=0.9, ls="--", label="persistence D-1")
    ax[0].set_ylabel("PV_total (MW)"); ax[0].legend(fontsize=7, ncol=3, loc="upper left")
    ax[0].set_title(f"Observed vs predicted — forecasts issued every {H} h ({days} days of Test)")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].hexbin(yt, yp, gridsize=40, mincnt=1, cmap="Greys", bins="log")
    lim = [0, max(yt.max(), yp.max()) * 1.02]
    ax[1].plot(lim, lim, color=fs.COLORS["model"], lw=1)
    mt = m.metrics(yt, yp)
    ax[1].set_title(f"Test, forecasts issued every {H} h\nR² = {mt['R2']:.3f}, RMSE = {mt['RMSE']:.2f} MW")
    ax[1].set_xlabel("observed (MW)"); ax[1].set_ylabel("predicted (MW)")
    fs.save(fig, file_name)


def fig_residuals(res, ds, file_name):
    """Residual diagnostics: distribution, per hour, per month, per horizon step."""
    preds, true, orig = load_predictions(res["name"])
    H = true.shape[1]
    g = representative_seed(res)
    e = preds[m.SEEDS.index(g)] - true
    hours = (ds.index[orig].hour.values[:, None] + np.arange(H)[None]) % 24
    months = ds.index[orig].to_period("M").astype(str)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.0))
    day = (true > 1)
    ax[0].hist(e[day], bins=80, color=fs.COLORS["mean"], density=True)
    ax[0].set_title(f"Daytime residuals (mean {e[day].mean():+.2f} MW)"); ax[0].set_xlabel("predicted − observed (MW)")
    rmse_h = [np.sqrt(np.mean(e[hours == h] ** 2)) for h in range(24)]
    ax[1].bar(range(24), rmse_h, color=fs.COLORS["model"]); ax[1].set_title("RMSE per hour"); ax[1].set_xlabel("hour")
    em = pd.Series((e ** 2).mean(axis=1), index=months).groupby(level=0).mean().pipe(np.sqrt)
    ax[2].plot(range(len(em)), em.values, marker="o", color=fs.COLORS["observed"])
    ax[2].set_xticks(range(len(em)), em.index, rotation=45, fontsize=7); ax[2].set_title("Monthly RMSE (Test)")
    ax[3].plot(np.arange(1, H + 1), np.sqrt((e ** 2).mean(axis=0)), color=fs.COLORS["model"], marker="." if H < 50 else None)
    ax[3].set_title("RMSE per horizon step"); ax[3].set_xlabel("step h")
    fs.save(fig, file_name)


# =============================================================================
# D. Linear baseline: multi-output Ridge
# =============================================================================

def multi_output_ridge(ds, L: int, H: int, alphas=(0.1, 1, 10, 100, 1000)):
    """Direct multi-horizon Ridge regression (strong linear baseline).

    Inputs: last L values of the target + deterministic covariates of the H
    future steps. Alpha chosen on Val; refitted on Train only (R2).
    Returns the Test forecasts (MW) on every hourly origin.
    """
    from sklearn.linear_model import Ridge
    v = ds.values
    i_det = [m.CHANNELS.index(c) for c in m.DETERMINISTIC_CHANNELS]

    def X_y(orig):
        X = np.stack([np.concatenate([v[t - L:t, 0], v[t:t + H][:, i_det].ravel()]) for t in orig])
        y = np.stack([v[t:t + H, 0] for t in orig])
        return X, y
    o_tr = m.origins(0, ds.i_val, L, H, 2)
    o_va = m.origins(ds.i_val, ds.i_test, L, H, 3)
    o_te = m.origins(ds.i_test, len(v), L, H, 1)
    Xtr, ytr = X_y(o_tr); Xva, yva = X_y(o_va)
    scores = {a: np.mean((Ridge(alpha=a).fit(Xtr, ytr).predict(Xva) - yva) ** 2) for a in alphas}
    a = min(scores, key=scores.get)
    model = Ridge(alpha=a).fit(Xtr, ytr)
    Xte, _ = X_y(o_te)
    yp = np.clip(ds.to_mw(model.predict(Xte)), 0, None)
    yt = np.stack([ds.table[m.TARGET].values[t:t + H] for t in o_te])
    return yp, yt, o_te, a


# =============================================================================
# E. Temporal cross-validation (TimeSeriesSplit) of the final configuration
# =============================================================================

def dataset_for_fold(ds, i_end_train: int, i_end_val: int, i_end_test: int):
    """Copy of the dataset for one fold: normalisation refitted on the fold's
    Train part only (no later information, R2)."""
    import copy as _copy
    raw = ds.table.values.astype(np.float64)[:i_end_test]
    mu, sd = raw[:i_end_train].mean(0), raw[:i_end_train].std(0)
    sd[sd < 1e-8] = 1.0
    j = _copy.copy(ds)
    j.table = ds.table.iloc[:i_end_test]
    j.values = ((raw - mu) / sd).astype(np.float32)
    j.mean, j.std = mu, sd
    j.i_val, j.i_test = i_end_train, i_end_val
    return j


def cross_validation(cfg, ds, n_folds: int = 5, seed: int = 0):
    """TimeSeriesSplit (expanding window) over the whole series.

    For each fold: Train = start of the series → 85 % of the learning segment,
    Val = remaining 15 % (early stopping), Test = next block. Blocks are
    aligned on whole days. Each fold is compared with daily persistence.
    """
    from sklearn.model_selection import TimeSeriesSplit
    days = np.arange(len(ds.values) // 24)
    rows = []
    for k, (learn, te) in enumerate(TimeSeriesSplit(n_splits=n_folds).split(days)):
        end_learn, end_te = (learn[-1] + 1) * 24, (te[-1] + 1) * 24
        end_tr = int(end_learn * 0.85) // 24 * 24
        j = dataset_for_fold(ds, end_tr, end_learn, end_te)
        model, hist, _ = m.train_model(cfg, j, seed)
        o = m.origins(end_learn, end_te, cfg.L, cfg.H, 1)
        yp, yt = m.predict(model, j, o)
        mt = m.metrics(yt, yp)
        rp = m.metrics(yt, m.daily_persistence(j, o, cfg.H))["RMSE"]
        rows.append(dict(fold=k + 1, train=f"{j.index[0]:%Y-%m} → {j.index[end_tr-1]:%Y-%m}",
                         test=f"{j.index[end_learn]:%Y-%m-%d} → {j.index[end_te-1]:%Y-%m-%d}",
                         RMSE=mt["RMSE"], nRMSE=mt["nRMSE"], R2=mt["R2"], RMSE_persistence=rp,
                         gain_vs_persistence=1 - mt["RMSE"] / rp, epochs=len(hist)))
    return pd.DataFrame(rows)


# =============================================================================
# F. Robustness: degradation curve versus noise level
# =============================================================================

def noise_curve(cfg, ds, name, levels=(0, 0.01, 0.02, 0.05, 0.10, 0.20)):
    """RMSE (each of the 5 seeds) when the exogenous channels are perturbed at σ = level."""
    import torch
    o = m.origins(ds.i_test, len(ds.values), cfg.L, cfg.H, 1)
    rows = []
    for g in m.SEEDS:
        model = m.PatchTST(cfg)
        model.load_state_dict(torch.load(folder(name) / f"weights_seed{g}.pt"))
        model.eval()
        for lv in levels:
            v = m.perturb_exogenous(ds, lv) if lv > 0 else ds.values
            yp, yt = m.predict(model, ds, o, v)
            rows.append(dict(seed=g, level=lv, RMSE=m.metrics(yt, yp)["RMSE"]))
    df = pd.DataFrame(rows)
    base = df[df.level == 0].set_index("seed").RMSE
    df["degradation"] = df.apply(lambda r: r.RMSE / base[r.seed] - 1, axis=1)
    return df


# =============================================================================
# G. Explainability (XAI)
# =============================================================================

GROUPS = {"Target (PV history)": [m.TARGET], "Grid": m.GRID_CHANNELS,
          "Cyclic hour": ["h_sin", "h_cos"], "Cyclic season": ["doy_sin", "doy_cos"],
          "Cyclic week": ["dow_sin", "dow_cos"], "Solar geometry": m.SOLAR_CHANNELS}


def occlusion_importance(model, ds, n_samples: int = 1500, seed: int = 0):
    """Importance by occlusion of channel groups: the group is replaced by its
    training mean (0 in normalised units) in the past AND, for deterministic
    channels, in the future. RMSE increase = importance."""
    cfg = model.cfg
    rng = np.random.default_rng(seed)
    o = np.sort(rng.choice(m.origins(ds.i_test, len(ds.values), cfg.L, cfg.H, 1), n_samples, replace=False))
    yp, yt = m.predict(model, ds, o)
    base = m.metrics(yt, yp)["RMSE"]
    rows = []
    for gname, cols in GROUPS.items():
        v = ds.values.copy()
        v[:, [m.CHANNELS.index(c) for c in cols]] = 0.0
        if gname.startswith("Target"):
            # the target must remain observable to compute the error: only its
            # "past" part is occluded, through a dedicated dataset
            yp2 = _predict_with_occluded_target(model, ds, o)
        else:
            yp2, _ = m.predict(model, ds, o, v)
        rows.append(dict(group=gname, RMSE=m.metrics(yt, yp2)["RMSE"],
                         RMSE_increase=m.metrics(yt, yp2)["RMSE"] / base - 1))
    return pd.DataFrame(rows), base


def _predict_with_occluded_target(model, ds, o):
    """Forecast when the target history is replaced by its mean daily training
    profile (the diurnal shape is kept, the recent cloud information removed)."""
    v = ds.values.copy()
    y = pd.Series(v[: ds.i_val, 0], index=ds.index[: ds.i_val])
    profile = y.groupby(y.index.hour).mean()
    v[:, 0] = profile.reindex(ds.index.hour).values
    yp, _ = m.predict(model, ds, o, v)
    return yp


def integrated_gradients(model, ds, n_samples: int = 300, steps: int = 32, seed: int = 0):
    """Integrated gradients (Sundararajan et al., 2017) on the past input [L, C]
    and, for the X variant, on the future covariates [H, C_det]. Baseline:
    zero input (Train mean). Explained output: forecast energy over the
    horizon. Returns the mean absolute attribution per past and future channel."""
    import torch
    cfg = model.cfg
    rng = np.random.default_rng(seed)
    o = np.sort(rng.choice(m.origins(ds.i_test, len(ds.values), cfg.L, cfg.H, 1), n_samples, replace=False))
    wds = m.WindowDataset(ds.values, o, cfg.L, cfg.H)
    past = torch.stack([wds[i][0] for i in range(len(o))])
    future = torch.stack([wds[i][1] for i in range(len(o))])
    cz = m._raw_cos_zenith(ds, future)
    att_p = torch.zeros_like(past); att_f = torch.zeros_like(future)
    model.eval()
    for a in torch.linspace(1 / steps, 1, steps):
        p = (a * past).requires_grad_(True); f = (a * future).requires_grad_(True)
        out = model(p, f, cz).sum()
        gp, gf = torch.autograd.grad(out, [p, f], allow_unused=True)
        if gp is not None: att_p += gp / steps
        if gf is not None: att_f += gf / steps
    att_p = (att_p * past).abs().mean(dim=(0, 1)).numpy()
    att_f = (att_f * future).abs().mean(dim=(0, 1)).numpy()
    ser_p = pd.Series(att_p, index=[f"{c} (past)" for c in m.CHANNELS])
    ser_f = pd.Series(att_f, index=[f"{c} (future)" for c in m.DETERMINISTIC_CHANNELS])
    allv = pd.concat([ser_p, ser_f])
    return allv / allv.sum()


def channel_attention_weights(model, ds, n_samples: int = 1000, seed: int = 0):
    """Mean cross-channel attention weights (CA variants): share of attention
    that the target channel gives to each channel."""
    import torch
    cfg = model.cfg
    if cfg.channel_mode != "CA":
        return None
    rng = np.random.default_rng(seed)
    o = np.sort(rng.choice(m.origins(ds.i_test, len(ds.values), cfg.L, cfg.H, 1), n_samples, replace=False))
    wds = m.WindowDataset(ds.values, o, cfg.L, cfg.H)
    past = torch.stack([wds[i][0] for i in range(len(o))]); future = torch.stack([wds[i][1] for i in range(len(o))])
    with torch.no_grad():
        model(past, future, m._raw_cos_zenith(ds, future))
    w = model.last_channel_weights.mean(dim=(0, 1)).numpy()
    return pd.Series(w, index=m.CHANNELS)


# =============================================================================
# H. "Capacity index" variant (causal normalisation by the installed fleet)
# =============================================================================

def causal_capacity(ds, window_days: int = 30) -> pd.Series:
    """Causal apparent capacity: 99 % quantile of the daily maxima of days
    D-30…D-1 (uses the past only), brought back to the hourly step."""
    dmax = ds.table[m.TARGET].resample("D").max()
    cap = dmax.shift(1).rolling(window_days, min_periods=7).quantile(0.99).bfill()
    return cap.reindex(ds.index, method="ffill")


def capacity_index_dataset(ds):
    """Dataset whose target is k = PV_total / causal capacity. The
    normalisation is refitted on Train (R2)."""
    import copy as _copy
    cap = causal_capacity(ds).values
    j = _copy.copy(ds)
    tab = ds.table.copy()
    tab[m.TARGET] = tab[m.TARGET].values / cap
    raw = tab.values.astype(np.float64)
    mu, sd = raw[: ds.i_val].mean(0), raw[: ds.i_val].std(0)
    sd[sd < 1e-8] = 1.0
    j.table, j.values, j.mean, j.std = tab, ((raw - mu) / sd).astype(np.float32), mu, sd
    return j, cap


# =============================================================================
# I. Caching of the heavy analyses of the final model
# =============================================================================

def cached(file_name: str, compute):
    """Run `compute()` once and cache the resulting DataFrame (CSV)."""
    f = m.CACHE_DIR / "final" / file_name
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        return pd.read_csv(f)
    df = compute()
    df.to_csv(f, index=False)
    return df


def evaluate_capacity_index(cfg, ds):
    """Ablation: the same configuration trained on k = PV / causal capacity
    (5 seeds); forecasts are brought back to MW by multiplying by the
    capacity of the forecast time (causally known: computed on previous days)."""
    jk, cap = capacity_index_dataset(ds)
    o = m.origins(ds.i_test, len(ds.values), cfg.L, cfg.H, 1)
    cap_fut = np.stack([cap[t:t + cfg.H] for t in o])
    rows = []
    for g in m.SEEDS:
        model, hist, _ = m.train_model(cfg, jk, g)
        k_pred, _ = m.predict(model, jk, o)
        yp = k_pred * cap_fut
        yt = np.stack([ds.table[m.TARGET].values[t:t + cfg.H] for t in o])
        mt = m.metrics(yt, yp)
        sel = o < ds.i_test + 24 * 30
        rd = m.daily_rmse(yt[sel], yp[sel], o[sel], ds)
        rows.append(dict(seed=g, RMSE=mt["RMSE"], nRMSE=mt["nRMSE"], R2=mt["R2"], MBE=mt["MBE"],
                         drift_7d=float(rd.iloc[-7:].mean() / rd.iloc[:7].mean()), epochs=len(hist)))
    return pd.DataFrame(rows)


def monthly_rmse(yt, yp, orig, ds):
    """RMSE per month of the origin (stability over the 7 Test months)."""
    months = ds.index[orig].to_period("M").astype(str)
    return pd.Series(((yp - yt) ** 2).mean(axis=1), index=months).groupby(level=0).mean().pipe(np.sqrt)
