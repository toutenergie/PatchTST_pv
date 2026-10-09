# Multivariate PatchTST forecasting — Total PV generation of the SENELEC grid (English version)

| Item | Content |
|---|---|
| `Article_PV_PatchTST_multivariate_EN.docx` | scientific paper (review of 56 articles, methodology, results, discussion, limitations, perspectives) |
| `Method_step_by_step_PV_EN.docx` | step-by-step method document (item 6 of projet.md) |
| `src/patchtst_pv.py` | commented core module: data, covariates, PatchTST (CI/CA/X), training, metrics, protocol, score |
| `src/analysis.py` | baselines, Diebold-Mariano tests, figures, cross-validation, XAI |
| `src/run_experiments.py` | trains configurations 1 → 4 (one factor at a time) |
| `src/nb00_eda.py`, `src/nb_config.py`, `src/nb05_final.py` | build and execute the notebooks |
| `src/article.py`, `src/method_doc.py` | build the Word documents from the cached results (docx_render.js) |
| `notebooks/00 … 05` | executed notebooks (outputs and figures embedded) |
| `figures/` | all figures (300 dpi), flowcharts `flow_config1…4` |
| `tables/` | CSV tables + `Results_summary.xlsx` |
| `models/best_patchtst.pt` | final model (weights, configuration, normalisation parameters) |
| `cache/` | per-configuration results (forecasts, weights, learning histories, final analyses) |
| `review/` | review text, list of the 56 references |

**Reproduce:** `cd src && python run_experiments.py` (several hours on CPU; skipped when the cache exists), then
`python nb00_eda.py; python nb_config.py 1..4; python nb05_final.py; python article.py; python method_doc.py; python export_tables.py`.
Requirements: python ≥ 3.10, torch, pvlib, pandas, scikit-learn, statsmodels, matplotlib, nbclient; node + docx (npm) for the Word files.

The results are identical to the French version (same cached trainings, converted with `tools/convert_cache.py`).
