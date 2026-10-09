# -*- coding: utf-8 -*-
"""
nb_config.py — Build and execute notebooks 01 to 04 (one configuration per notebook).

Usage: python nb_config.py 1   (or 2, 3, 4)

The Markdown conclusions are written from the actual results read from the
cache: no number is typed by hand.
"""
import dataclasses
import json
import sys

sys.path.insert(0, ".")
import analysis as an  # noqa: E402
import patchtst_pv as m  # noqa: E402
from nb_tools import build, code_header  # noqa: E402

# -----------------------------------------------------------------------------
# 1. Definition of the four configurations (question, hypotheses, fixed factors)
# -----------------------------------------------------------------------------
DEFS = {
    1: dict(step="config1_patch", file="01_Config1_Patch", factor="P",
            title="Configuration 1 — Influence of temporal granularity (patch length P)",
            question="Which patch size best represents the hourly dynamics of PV generation?",
            hypothesis=("A 24 h patch matches the solar cycle and yields \"daily\" tokens; a 12 h patch "
                        "separates morning and afternoon; a 48 h patch reduces the number of tokens (cost) "
                        "but merges two days. The stride between patches is P/2 (50 % overlap)."),
            fixed="L = 168 h, CA-X channel handling (cross-channel attention + known future), H = 24 h"),
    2: dict(step="config2_history", file="02_Config2_History", factor="L",
            title="Configuration 2 — Influence of the history (input window L)",
            question="How many hours of the past should the model see?",
            hypothesis=("L = 96 h (4 days) captures recent cloud persistence; L = 168 h (1 week) adds a "
                        "complete weekly cycle; L = 336 h (2 weeks) gives more context but is more exposed "
                        "to regime changes and doubles the attention cost."),
            fixed="P selected in configuration 1, CA-X channels, H = 24 h"),
    3: dict(step="config3_channels", file="03_Config3_Channels_CI_CA", factor="channels",
            title="Configuration 3 — Channel independence: PatchTST-CI vs PatchTST-CA",
            question="Do the covariates (grid, cyclic, solar geometry) improve the forecast, and how should they be injected?",
            hypothesis=("**CI**: shared encoder, only the target channel feeds the head: the exogenous channels "
                        "have no influence (univariate equivalent). **CA**: cross-channel attention at each patch: "
                        "the target \"queries\" the past exogenous channels. **X**: the FUTURE deterministic "
                        "covariates (hour, season, solar geometry of the forecast instant) correct the forecast "
                        "step by step. **g**: physical gate: the forecast is forced to 0 when the sun is down."),
            fixed="P and L selected in configurations 1 and 2, H = 24 h"),
    4: dict(step="config4_horizon", file="04_Config4_Horizon", factor="H",
            title="Configuration 4 — Forecast horizon H = 1, 6, 12, 24, 168 h",
            question="How do accuracy and stability evolve with the horizon?",
            hypothesis=("The error grows with the horizon up to the daily cycle; beyond it (168 h), the recent "
                        "cloud information is lost and the forecast tends towards a conditional climatology. "
                        "The comparison with persistence is made at the same horizon."),
            fixed="P, L and channel handling selected in configurations 1 to 3"),
}


def configs_for(num: int) -> list:
    """Rebuild exactly the list of configurations of run_experiments.py."""
    sel = json.loads((m.CACHE_DIR / "selection.json").read_text())

    def parse(name):
        # e.g. "PatchTST-CAX_P24_L168_H24" → mode "CAX", P=24, L=168, H=24
        mode = name.split("-")[1].split("_")[0]
        parts = name.split("_")
        return m.PatchTSTConfig(P=int(parts[1][1:]), L=int(parts[2][1:]), H=int(parts[3][1:]),
                                channel_mode=mode[:2], known_future="X" in mode[2:],
                                physical_gate="g" in mode[2:])
    base = m.PatchTSTConfig(P=24, L=168, H=24, channel_mode="CA", known_future=True)
    if num == 1:
        return [dataclasses.replace(base, P=p) for p in (12, 24, 48)]
    if num == 2:
        c1 = parse(sel["config1_patch"])
        return [dataclasses.replace(c1, L=l) for l in (96, 168, 336)]
    if num == 3:
        c2 = parse(sel["config2_history"])
        return [dataclasses.replace(c2, channel_mode=a, known_future=b, physical_gate=c)
                for a, b, c in [("CI", False, False), ("CI", True, False), ("CA", False, False),
                                ("CA", True, False), ("CA", True, True)]]
    c3 = parse(sel["config3_channels"])
    return [dataclasses.replace(c3, H=h) for h in (1, 6, 12, 24, 168)]


# -----------------------------------------------------------------------------
# 2. Conclusion written from the cached results
# -----------------------------------------------------------------------------
def conclusion(num: int) -> str:
    """Write the conclusion of a configuration from the cached results."""
    ds = m.build_dataset()
    cfgs = configs_for(num)
    res = [m.evaluate_configuration(c, ds) for c in cfgs]
    tab = m.comparison_table(res).set_index("configuration")
    selected = json.loads((m.CACHE_DIR / "selection.json").read_text())[DEFS[num]["step"]]
    r = tab.loc[selected]
    lines = [f"### Conclusion of configuration {num}", "",
             f"**Selected configuration: `{selected}`** (normalised decision score = {r.normalised_score:.3f}; "
             f"raw score = {r.score:.2f}).", ""]
    if num == 4:
        lines[2] = (f"**Best score: `{selected}`.** Here the score compares different *tasks*: the 1 h error is "
                    "mechanically lower than the 24 h error, so the horizon is not chosen by the score. The final "
                    "model (notebook 05) is evaluated at the operational day-ahead horizon (24 h); this configuration "
                    "describes how accuracy and stability evolve with the horizon, and the relevant comparison is the "
                    "gain over persistence **at the same horizon**.")
    # position relative to persistence at the same horizon
    for c, rr in zip(cfgs, res):
        _, y_true, orig = an.load_predictions(rr["name"])
        ref = an.baseline_rows(ds, orig, y_true, c.H).set_index("configuration")
        tab.loc[rr["name"], "RMSE_persistence"] = ref.loc["Persistence D-1", "RMSE"]
    tab["gain_vs_persistence"] = 1 - tab["RMSE"] / tab["RMSE_persistence"]
    for name, l in tab.iterrows():
        lines.append(f"- `{name}`: RMSE = {l.RMSE:.2f} ± {l.RMSE_std:.2f} MW, nRMSE = {l.nRMSE:.3f}, "
                     f"gain vs persistence = {100*l.gain_vs_persistence:+.1f} %, degradation under noise = "
                     f"{100*l.noise_degradation:+.1f} %, 7-day drift = {l.drift_7d:.2f}, "
                     f"inference = {l.inference_ms:.1f} ms, {l.n_parameters:,} parameters.")
    lines += ["", f"- RMSE gap between the best and the worst variant: "
              f"{tab.RMSE.max() - tab.RMSE.min():.2f} MW ({100*(tab.RMSE.max()/tab.RMSE.min()-1):.1f} %).",
              f"- All variants meet the robustness criterion (degradation < 15 %): "
              f"{'yes' if (tab.noise_degradation < 0.15).all() else 'no'}; "
              f"all meet the drift target < 1.2: {'yes' if (tab.drift_7d < 1.2).all() else 'no'}; "
              f"all meet the efficiency targets (< 100 ms, < 50 MB): "
              f"{'yes' if ((tab.inference_ms < 100) & (tab.size_MB < 50)).all() else 'no'}."]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 3. Notebook cells
# -----------------------------------------------------------------------------
def cells(num: int) -> list:
    d = DEFS[num]
    short = d["title"].split(" — ")[0]
    return [
("md", f"""
# {d['file'][:2]} — {d['title']}

**Question.** {d['question']}

**Hypotheses.** {d['hypothesis']}

**Fixed parameters.** {d['fixed']}.

**Protocol** (identical for every configuration — module `patchtst_pv`, section 8):
1. training on Train (2019–2021), early stopping on Val (01–09/2022), **5 seeds**;
2. accuracy on **every hourly origin of the Test set** (10/2022–05/2023): RMSE, MAE, nRMSE = RMSE / mean, R², MBE;
3. **robustness**: standard deviation over the seeds and 5 % Gaussian noise on the exogenous channels (rejected if ΔRMSE > 15 %);
4. **stability**: 30-day rolling forecast without re-training, drift = RMSE(D24–30) / RMSE(D1–7) (target < 1.2);
5. **efficiency**: inference time for one sample (< 100 ms) and weight size (< 50 MB);
6. **decision score** of `projet.md` (raw and normalised) and Diebold-Mariano tests against persistence and the 7-day mean.

> Trainings are cached (`cache/<configuration>/`): deleting the corresponding folder re-runs the full
> training from this notebook.
"""),
("code", code_header("patchtst_pv") + "import analysis as an, flowchart as fc, json, dataclasses\nfrom nb_config import configs_for"),
("md", "## 1. Flowchart of the configuration"),
("code", f'fc.draw({num}, "Flowchart — {short}", "flow_config{num}")'),
("md", "## 2. Configurations evaluated"),
("code", f"""
ds = m.build_dataset()
configs = configs_for({num})
pd.DataFrame([dict(name=c.name(), P=c.P, L=c.L, H=c.H, channels=c.channel_mode, known_future=c.known_future,
                   physical_gate=c.physical_gate, n_patches=c.n_patches) for c in configs])
"""),
("md", "## 3. Training and evaluation (5 seeds each)"),
("code", """
results = [m.evaluate_configuration(c, ds) for c in configs]   # reads the cache if it exists
detail = pd.DataFrame([dict(configuration=r["name"], **{k: g[k] for k in
         ["seed", "epochs", "duration_s", "RMSE", "MAE", "nRMSE", "R2", "noise_degradation", "drift_7d", "drift_D30_D1"]})
         for r in results for g in r["per_seed"]])
detail
"""),
("md", "## 4. Comparison table of the metrics (mean over 5 seeds) and naive baselines"),
("code", """
table = m.comparison_table(results)
_, true0, orig0 = an.load_predictions(results[0]["name"])
refs = an.baseline_rows(ds, orig0, true0, configs[0].H) if len({c.H for c in configs}) == 1 else None
columns = ["configuration", "RMSE", "RMSE_std", "MAE", "nRMSE", "nRMSE_day", "R2", "MBE", "noise_degradation",
           "drift_D30_D1", "drift_7d", "inference_ms", "size_MB", "n_parameters", "score", "normalised_score", "rejected_noise"]
shown = pd.concat([table[columns], refs], ignore_index=True) if refs is not None else table[columns]
m.TABLES_DIR.mkdir(exist_ok=True)
shown.to_csv(m.TABLES_DIR / "table_configNUM.csv", index=False)
shown.round(4)
""".replace("NUM", str(num))),
("md", """
*Reading the columns.* `RMSE_std`: standard deviation over the 5 seeds (robustness). `nRMSE_day`: nRMSE restricted to
daytime hours. `noise_degradation`: relative RMSE increase under 5 % noise on the exogenous channels. `drift_D30_D1`: raw
ratio of `projet.md` (sensitive to the cloudiness of two isolated days); `drift_7d`: smoothed ratio used in the score.
`score`: raw formula of `projet.md`; `normalised_score`: each criterion mapped to [0, 1] across the compared variants.
"""),
("code", """
dm = an.dm_tests(ds, results)
dm.round(4)
"""),
("md", "## 5. Seed rosary and 30-day stability"),
("code", f'an.fig_rosary(results, ds, "fig_config{num}_rosary", "{short}")'),
("code", f'an.fig_learning_and_drift(results, "fig_config{num}_learning_drift")'),
("md", "## 6. Observed vs predicted series — best model of the configuration"),
("code", f"""
selected = an.selected_at("{d['step']}")
print("Selected configuration:", selected)
res_selected = [r for r in results if r["name"] == selected][0]
an.fig_observed_vs_predicted(res_selected, ds, "fig_config{num}_observed_predicted")
an.fig_residuals(res_selected, ds, "fig_config{num}_residuals")
"""),
("md", conclusion(num)),
]


if __name__ == "__main__":
    num = int(sys.argv[1])
    print(build(DEFS[num]["file"], cells(num)))
