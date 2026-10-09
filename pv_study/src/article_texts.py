# -*- coding: utf-8 -*-
"""
article_texts.py — Texts of the paper that do not depend on the results:
title, introduction, review (read from review/review_text.md), data and
methodology. Each section is a list of blocks for docx_render.js.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TITLE = ("Multivariate forecasting of the total photovoltaic generation of a national grid without meteorological "
         "data: a cross-channel-attention PatchTST conditioned on cyclic covariates and solar geometry — "
         "application to the SENELEC grid (Senegal)")
AUTHORS = "Abdoulaye FAYE^1,*^, Alphousseyni NDIAYE^1^"
AFFILIATION = ("^1^ Research Team in Energy Efficiency and Energy Systems, Doctoral School of Sciences and Techniques "
               "and Social Sciences (EDTSS), Alioune Diop University of Bambey (UADB), Senegal — "
               "^*^ Corresponding author: abdoulaye7.faye@uadb.edu.sn")
HEADER = "Faye & Ndiaye — PatchTST forecasting of total PV generation, SENELEC grid"

KEYWORDS = ("**Keywords:** photovoltaic forecasting; PatchTST; Transformer; cyclic variables; solar geometry; "
            "cross-channel attention; long-term stability; SENELEC grid; West Africa")


def introduction():
    return [
        {"type": "h1", "text": "1. Introduction"},
        {"type": "p", "text": (
            "The massive integration of photovoltaic (PV) plants is transforming the operation of power grids. "
            "Because it is inverter-connected and depends on cloudiness, PV generation reduces synchronous inertia "
            "and introduces fast variability that the operator must compensate with reserves [1–5]. Recent work "
            "shows that, beyond a certain integration rate, the resilience of a grid can degrade abruptly in the "
            "absence of storage and anticipation [6, 7]. Generation forecasting is therefore one of the first "
            "levers of safe and economic integration [10, 11].")},
        {"type": "p", "text": (
            "The SENELEC grid (Senegal) illustrates these issues. About ten PV plants of a few tens of MWp each have "
            "been connected to the transmission grid since the mid-2010s. Their intermittency has been associated with "
            "more difficult frequency control and more frequent load shedding [12]. Admissible penetration rates have "
            "been estimated at between 16 and 24 % depending on the voltage level and the presence of storage [13, 14], "
            "and PV generation is among the main statistical determinants of the grid frequency [19, 20]. Yet the "
            "dispatch centre has no synchronous meteorological measurements at the sites; the only systematically "
            "available information is the hourly operating log.")},
        {"type": "p", "text": (
            "The literature on PV forecasting with deep learning is abundant [23–30], but three limitations restrict "
            "its transfer to this context. First, it almost always assumes meteorological input variables "
            "[28, 38, 45]. Second, it deals with individual plants, not with the aggregated generation of a national "
            "grid, whose installed capacity changes with every commissioning. Third, evaluation protocols are "
            "heterogeneous: comparison with a naive baseline, dispersion across initialisations, robustness, the "
            "stability of performance over time and the inference cost are rarely measured together [24, 41, 51]. "
            "Patch Transformers, and PatchTST in particular [37], have renewed time-series forecasting and are "
            "beginning to be applied to photovoltaics [38–40]. Their channel-independence assumption is however "
            "debated as soon as informative covariates are available [42, 45].")},
        {"type": "p", "text": "This paper makes four contributions:"},
        {"type": "numbered", "instance": 1, "items": [
            "**A multivariate formulation without meteorology** of forecasting the total PV generation of a national "
            "grid. The productions of the individual plants are excluded from the inputs (they are only used to "
            "build the target). The model combines the history of the target, the grid operating context (thermal, "
            "wind, import) and **deterministic** covariates: cyclic encodings and solar geometry computed for the "
            "barycentre of the fleet.",
            "**A PatchTST-CA-X architecture** that combines (i) a **selective** reversible normalisation, applied to the "
            "stochastic channels only; (ii) **cross-channel** attention per patch position, in which the target "
            "queries the covariates; (iii) **conditioning of the forecasting head on the known future**, since the "
            "deterministic covariates are available without error at any horizon.",
            "**A decision-oriented evaluation protocol.** Four configurations are studied one factor at a time "
            "(patch length, history, channel handling, horizon), each with 5 seeds. The protocol measures accuracy, "
            "robustness to noise, drift over 30 days of forecasting without re-training, efficiency, a temporal "
            "cross-validation and Diebold-Mariano tests. Everything is aggregated into a decision score.",
            "**An explainability analysis** (occlusion of channel groups, integrated gradients, attention weights) that "
            "quantifies what each family of covariates actually contributes. A channel with no physical link to "
            "irradiance (the day of week) serves as a negative control.",
        ]},
        {"type": "p", "text": (
            "Section 2 presents the literature review. Section 3 describes the data, the architecture and the "
            "protocol. Section 4 reports the results, which Section 5 discusses. Sections 6 and 7 present the "
            "limitations and perspectives, and Section 8 concludes.")},
    ]


def review():
    """Read review/review_text.md and convert it into blocks."""
    text = (ROOT / "review" / "review_text.md").read_text(encoding="utf-8")
    blocks, items = [], []

    def flush():
        nonlocal items
        if items:
            blocks.append({"type": "numbered", "instance": 2, "items": items}); items = []
    for line in text.split("\n"):
        l = line.strip()
        if not l:
            continue
        if l.startswith("### "):
            flush(); blocks.append({"type": "h2", "text": l[4:]})
        elif l.startswith("## "):
            flush(); blocks.append({"type": "h1", "text": l[3:]})
        elif re.match(r"^\d+\. ", l):
            items.append(re.sub(r"^\d+\. ", "", l))
        elif l.startswith("(i)") or l.startswith("(ii)") or l.startswith("(iii)"):
            blocks[-1]["text"] += " " + l
        else:
            flush(); blocks.append({"type": "p", "text": l})
    flush()
    return blocks


def review_table():
    import sys
    sys.path.insert(0, str(ROOT / "review"))
    from synthesis_table_56 import TABLE
    return {"type": "table", "size": 14, "widths": [0.5, 1.9, 1.2, 3.2, 1.9, 3.0], "left": [1, 3, 5],
            "caption": "**Table 1.** Synthesis of the 56 works of the review (axes: A stability and PV integration, "
                       "B Senegal/Africa, C reviews of forecasting, D deep learning for PV/RE, E PatchTST and Transformers).",
            "headers": ["No.", "Reference", "Axis", "Method / object", "Data / validation", "Contribution to this study"],
            "rows": [[f"[{n}]", ref, f"{ax}", meth, dat, con] for n, ref, ax, meth, dat, con in TABLE],
            "note": "Written from the abstracts and records of the corpus; \"n/a\": not documented in the source."}


def methodology(D):
    """D: dictionary of values derived from the data (period, counts, offset…)."""
    return [
        {"type": "h1", "text": "3. Data and methodology"},
        {"type": "h2", "text": "3.1 Data"},
        {"type": "p", "text": (
            f"The data come from the SENELEC hourly operating log, from {D['start']} to {D['end']} "
            f"({D['n_hours']} hours; {D['n_missing']} hours missing, in eight jumps of one to three days). The log "
            "contains the production of the thermal units, of the Taiba Ndiaye wind farm, of imports (Manantali, "
            "Félou, SOMELEC) and of eleven photovoltaic plants. The target is the total PV generation of the grid "
            f"(Figure 1). Its mean level rises by {D['growth']:.0f} % between 2019 and 2022 as plants are commissioned: "
            "Diass (June 2019), replacement of Kahone by ERS and Scaling Kahone (January 2021), Kael Touba (March 2021).")},
        {"type": "figure", "path": "figures/fig01_pv_total_series.png", "caption":
            "**Figure 1.** Hourly total PV generation of the SENELEC grid (2019–2023), apparent capacity envelope and "
            "daily energy. Orange and red backgrounds: validation and test periods; dotted lines: commissionings."},
        {"type": "h2", "text": "3.2 Target construction and preprocessing"},
        {"type": "p", "text": (
            "The target is the sum of the eleven PV plants, built in three steps for each plant c. (i) Values above "
            "1.2 times the 99.9 % quantile computed over the training period are treated as entry errors "
            f"({D['n_clipped']} values, up to 958 MW for a 20 MW plant). (ii) Outside the period of existence of the "
            "plant, production is zero. (iii) Gaps are imputed **causally**: linear interpolation up to 3 h, then the "
            "mean of the same hour over the previous seven days. Table 2 details these operations. The individual "
            "productions are **never** used as model inputs.")},
        {"type": "equation", "text": "y(t) = Σ_{c=1..11} p̃_{c}(t)", "number": 1},
        {"type": "h2", "text": "3.3 Covariates"},
        {"type": "p", "text": (
            "Three families of covariates are built (twelve channels in total, target included). The **cyclic "
            "encodings** ensure continuity at the boundaries of the cycles (23:00 → 00:00, 31 December → 1 January). "
            "They are computed at the middle of each hour, for the hour h (T = 24), the day of year (T = 365.25) and "
            "the day of week (T = 7):")},
        {"type": "equation", "text": "x_{sin}(t) = sin(2π·u(t)/T),   x_{cos}(t) = cos(2π·u(t)/T)", "number": 2},
        {"type": "p", "text": (
            "The **solar geometry** is computed with pvlib [62] at the approximate barycentre of the plants (14.9° N; "
            "16.4° W). It includes the cosine of the apparent zenith angle θ_{z} (bounded at 0 at night) and the "
            "clear-sky global irradiance according to the Haurwitz model [61]:")},
        {"type": "equation", "text": "GHI_{cs}(t) = 1098 · cos θ_{z}(t) · exp(−0.057 / cos θ_{z}(t))", "number": 3},
        {"type": "p", "text": (
            "A time offset δ between the log timestamps and solar time is estimated over the training period by "
            "maximising the correlation between the mean daily profile of the target and that of GHI_{cs} "
            f"(δ = {D['offset']:+.2f} h). These eight covariates are **deterministic**: they are known without error "
            "at any horizon. The **grid context** (aggregated thermal production, Taiba wind, Manantali import) serves "
            "as a proxy for the atmospheric state: net demand and the wind regime reflect cloudiness and advection. "
            "These three channels are used over the past window only. The aggregated log columns that include PV "
            "generation (total of units, interconnected grid) are excluded, to avoid any indirect leakage.")},
        {"type": "h2", "text": "3.4 PatchTST-CA-X architecture"},
        {"type": "p", "text": (
            "PatchTST [37] splits each channel of the input window of length L into N patches of length P, with a "
            "stride S = P/2:")},
        {"type": "equation", "text": "N = ⌊(L − P) / S⌋ + 1", "number": 4},
        {"type": "p", "text": (
            "Each patch is projected into a space of dimension D = 64 and added to a position embedding and a "
            "channel-identity embedding. A two-layer Transformer encoder [58] (four heads, pre-normalisation), whose "
            "weights are **shared** across channels, then models the temporal dependencies. The proposed "
            "architecture (Figure 2) adds three elements:")},
        {"type": "bullets", "items": [
            "**Selective RevIN.** Reversible instance normalisation [57] is applied to the target and to the grid "
            "channels, whose level drifts with installed capacity. The deterministic covariates keep their global "
            "normalisation. Indeed, over a window of a few days the day of year barely changes: standardising it "
            "would erase the position in the year in favour of an artificial ramp. We observed this defect in a "
            "first version, where it showed up as an abnormal attribution of 24 % to the past day of year.",
            "**Cross-channel attention (CA).** At each patch position n, the token of the target channel serves as "
            "the query and the C channel tokens serve as keys and values: z'_{n} = LN(z_{n,0} + MHA(z_{n,0}, Z_{n}, Z_{n})). "
            "Under strict channel independence (CI), the head only receives the target token: the covariates then "
            "have **no** influence on the forecast, and the model is equivalent to a univariate PatchTST with shared weights.",
            "**Known-future conditioning (X).** The linear head produces a base forecast ŷ_{0} ∈ R^H^. For each step "
            "h, a perceptron with one hidden layer receives a context c = W·vec(z') ∈ R^32^, the eight deterministic "
            "covariates of instant t+h and ŷ_{0}(h), and produces an additive correction: "
            "ŷ(h) = ŷ_{0}(h) + g([c ; f(t+h) ; ŷ_{0}(h)]). A variant (g) adds a physical gate that sets the forecast "
            "to zero when cos θ_{z}(t+h) = 0.",
        ]},
        {"type": "figure", "path": "figures/fig_architecture.png", "width_cm": 16.5, "caption":
            "**Figure 2.** PatchTST-CA-X architecture. In red: proposed elements (cross-channel attention, "
            "conditioning on the future deterministic covariates). The CI variant (standard PatchTST) only uses "
            "the target token."},
        {"type": "h2", "text": "3.5 Experimental protocol"},
        {"type": "p", "text": (
            "**Strict temporal split**, without shuffling: training from 01/01/2019 to 31/12/2021 "
            f"({D['n_train']} h), validation from 01/01/2022 to 30/09/2022 ({D['n_val']} h), test from 01/10/2022 "
            f"to 09/05/2023 ({D['n_test']} h). Every operation fitted on the data (clipping thresholds, global "
            "normalisation, penalty of the Ridge baseline) is fitted on the training period only. A forecast origin "
            "t means that the last observed hour is t − 1; the model directly forecasts y(t), …, y(t+H−1).")},
        {"type": "p", "text": (
            "**Training.** Squared loss; Adam (initial rate 10^−3^, halved after two epochs without improvement); "
            "batches of 128; gradient clipping at 1; at most 25 epochs; early stopping (patience 4) on the "
            "validation loss, with restoration of the best weights. At each epoch, 20 % of the hourly training "
            "origins are drawn at random, which guarantees that every hourly phase is seen. A fixed step of 3 h, for "
            "example, would only show 8 out of 24. Each configuration is trained with 5 seeds. The model is "
            "implemented in PyTorch [64].")},
        {"type": "p", "text": (
            "**One-factor-at-a-time design of experiments** (Table 3). The value selected at one step is carried "
            "over to the next. The choice is made on the normalised decision score, among the variants that pass "
            "the noise test.")},
        {"type": "table", "caption": "**Table 3.** Design of experiments.", "widths": [1.4, 2.2, 3.4, 3.0],
         "left": [3],
         "headers": ["Configuration", "Factor", "Values", "Fixed parameters"],
         "rows": [["1", "Patch length P", "12, 24, 48 h", "L = 168 h, CA-X, H = 24 h"],
                  ["2", "History L", "96, 168, 336 h", "P*, CA-X, H = 24 h"],
                  ["3", "Channel handling", "CI, CI-X, CA, CA-X, CA-Xg", "P*, L*, H = 24 h"],
                  ["4", "Horizon H", "1, 6, 12, 24, 168 h", "P*, L*, channels*"]],
         "note": "*: value selected at the previous step."},
        {"type": "p", "text": (
            "**Baselines.** (i) Daily persistence: ŷ(t+h) = y(t+h−24), repeating the last observed day beyond 24 h. "
            "(ii) Mean of the last seven days at the same hour. (iii) Direct multi-output Ridge regression, whose "
            "inputs are the last L values of the target and the deterministic covariates of the H future steps; the "
            "penalty is chosen on the validation set. (iv) PatchTST-CI, the univariate equivalent.")},
        {"type": "h2", "text": "3.6 Evaluation criteria and decision score"},
        {"type": "p", "text": (
            "The metrics are computed over **every hourly origin** of the test set, all horizon steps combined:")},
        {"type": "equation", "text": "RMSE = √( (1/n) Σ (ŷ_{i} − y_{i})² ),   nRMSE = RMSE / ȳ", "number": 5},
        {"type": "equation", "text": "MAE = (1/n) Σ |ŷ_{i} − y_{i}|", "number": 6},
        {"type": "p", "text": (
            "complemented by R², the mean bias (MBE) and the nRMSE restricted to daytime hours. **Robustness** "
            "combines the standard deviation of the RMSE over the 5 seeds, σ, and the relative degradation of the "
            "RMSE when the exogenous channels (past and future) are perturbed by Gaussian noise with a standard "
            "deviation equal to 5 % of their training standard deviation. A degradation above 15 % leads to the "
            "rejection of the configuration. **Long-term stability** is evaluated by a 30-day rolling forecast without "
            "re-training, at the start of the test period. The ratio RMSE(D30)/RMSE(D1) depends on the cloudiness of "
            "two isolated days; we therefore use a smoothed ratio:")},
        {"type": "equation", "text": "Drift = RMSE(D24…D30) / RMSE(D1…D7)   (target < 1.2)", "number": 7},
        {"type": "p", "text": (
            "**Efficiency** is measured by the inference time for one sample on one processor core (target "
            "< 100 ms) and by the size of the weights (< 50 MB). The **decision score** aggregates these criteria:")},
        {"type": "equation", "text": "S = 0.4 / nRMSE + 0.3 / σ + 0.3 / Drift + bonus", "number": 8},
        {"type": "p", "text": (
            "where the bonus is 0.1 per efficiency criterion met. The three terms have different scales: 1/σ, in "
            "particular, can dominate when σ is very small. We therefore also report a normalised score, in which "
            "each criterion is mapped to [0, 1] across the compared variants; this is the score that guides the "
            "selection. Equal accuracy against the baselines is tested with the Diebold-Mariano test [59], with the "
            "correction of Harvey et al. [60] and a long-run variance with H − 1 lags. Finally, the final "
            "configuration undergoes a **temporal cross-validation** (TimeSeriesSplit, 5 expanding-window folds "
            "covering 2019–2023, normalisation re-fitted at each fold), an ablation with a causal **capacity index**, "
            "and an **explainability** analysis: occlusion of channel groups, integrated gradients [63] and "
            "cross-channel attention weights.")},
    ]


METHOD_REFERENCES = [
    "[57] T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, J. Choo, Reversible instance normalization for accurate time-series forecasting against distribution shift, in: Proc. 10th Int. Conf. Learn. Represent. (ICLR), 2022.",
    "[58] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, in: Adv. Neural Inf. Process. Syst. 30 (NeurIPS), 2017.",
    "[59] F.X. Diebold, R.S. Mariano, Comparing predictive accuracy, Journal of Business & Economic Statistics 13 (1995) 253–263. https://doi.org/10.1080/07350015.1995.10524599",
    "[60] D. Harvey, S. Leybourne, P. Newbold, Testing the equality of prediction mean squared errors, International Journal of Forecasting 13 (1997) 281–291. https://doi.org/10.1016/S0169-2070(96)00719-4",
    "[61] B. Haurwitz, Insolation in relation to cloudiness and cloud density, Journal of Meteorology 2 (1945) 154–166. https://doi.org/10.1175/1520-0469(1945)002<0154:IIRTCA>2.0.CO;2",
    "[62] W.F. Holmgren, C.W. Hansen, M.A. Mikofski, pvlib python: a python package for modeling solar energy systems, Journal of Open Source Software 3 (2018) 884. https://doi.org/10.21105/joss.00884",
    "[63] M. Sundararajan, A. Taly, Q. Yan, Axiomatic attribution for deep networks, in: Proc. 34th Int. Conf. Mach. Learn. (ICML), PMLR 70, 2017, pp. 3319–3328.",
    "[64] A. Paszke et al., PyTorch: An imperative style, high-performance deep learning library, in: Adv. Neural Inf. Process. Syst. 32 (NeurIPS), 2019.",
]


def references():
    lines = (ROOT / "review" / "references_56.txt").read_text(encoding="utf-8").strip().split("\n")
    return [{"type": "h1", "text": "References"},
            {"type": "references", "items": [l.strip() for l in lines] + METHOD_REFERENCES}]
