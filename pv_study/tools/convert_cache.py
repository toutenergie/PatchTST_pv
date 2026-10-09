# -*- coding: utf-8 -*-
"""
convert_cache.py — Convert the cached results of the French code base into the
English code base, so that the English notebooks and documents reproduce the
exact same numbers WITHOUT re-training.

    python convert_cache.py <french_cache_dir> <english_cache_dir>

What is converted:
  * results.json   : French keys → English keys (config, per-seed, summary);
  * history CSVs   : column names;
  * model weights  : state_dict keys (attribute names of the PatchTST class);
  * file names     : pred_graine → pred_seed, poids_graine → weights_seed, …;
  * final analyses : noise curve, cross-validation, capacity ablation, XAI
                     tables (column names and group / channel labels);
  * selection.json : step names translated (configuration names are notation).
The dataset pickle is NOT copied: it is rebuilt by the English pipeline.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

# --- results.json -------------------------------------------------------------
KEYS = {
    "par_graine": "per_seed", "synthese": "summary", "nom": "name", "graine": "seed",
    "duree_s": "duration_s", "epoques": "epochs", "RMSE_jour": "RMSE_day", "nRMSE_jour": "nRMSE_day",
    "RMSE_bruit": "RMSE_noise", "degradation_bruit": "noise_degradation",
    "drift_J30_J1": "drift_D30_D1", "drift_7j": "drift_7d", "pente_MW_par_jour": "slope_MW_per_day",
    "rmse_journaliers": "daily_rmse", "taille_Mo": "size_MB", "n_parametres": "n_parameters",
    # configuration fields
    "mode_canaux": "channel_mode", "futur_connu": "known_future", "porte_physique": "physical_gate",
    "bornage_physique": "physical_bounds", "n_tetes": "n_heads", "n_couches": "n_layers",
    "taille_lot": "batch_size", "epoques_max": "max_epochs", "pas_train": "train_step",
    "fraction_epoque": "epoch_fraction", "pas_val": "val_step",
}


def key(k: str) -> str:
    """Translate a key, including summary keys with a _moy suffix."""
    if k.endswith("_moy"):
        return key(k[:-4]) + "_mean"
    if k.endswith("_std") and k[:-4] in KEYS:
        return KEYS[k[:-4]] + "_std"
    return KEYS.get(k, k)


def translate(obj):
    if isinstance(obj, dict):
        return {key(k): translate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [translate(v) for v in obj]
    return obj


# --- names of the selection steps (selection.json) ------------------------------
STEPS = {"config2_historique": "config2_history", "config3_canaux": "config3_channels"}


# --- model weights --------------------------------------------------------------
WEIGHT_PREFIXES = {
    "id_canal": "channel_id", "encodeur_temporel.": "temporal_encoder.",
    "attention_canaux.": "channel_attention.", "norme_canaux.": "channel_norm.",
    "tete.": "head.", "contexte.": "context.", "borne_basse": "lower_bound", "borne_haute": "upper_bound",
}


def translate_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for fr, en in WEIGHT_PREFIXES.items():
            if k.startswith(fr):
                k = en + k[len(fr):]
                break
        out[k] = v
    return out


# --- labels of the final analyses ------------------------------------------------
GROUPS = {"Cible (historique PV)": "Target (PV history)", "Cible (historique éolien)": "Target (wind history)",
          "Réseau": "Grid", "Cycliques heure": "Cyclic hour", "Cycliques saison": "Cyclic season",
          "Cycliques semaine": "Cyclic week", "Géométrie solaire": "Solar geometry"}
CHANNELS = {"Thermique": "Thermal", "Eolien_Taiba": "Taiba_wind", "ghi_ciel_clair": "ghi_clear_sky"}


def channel_label(s: str) -> str:
    s = s.replace("(passé)", "(past)").replace("(futur)", "(future)")
    for fr, en in CHANNELS.items():
        s = s.replace(fr, en)
    return s


FINAL = {
    "bruit.csv": ("noise.csv", {"graine": "seed", "niveau": "level"}),
    "validation_croisee.csv": ("cross_validation.csv", {"pli": "fold", "RMSE_persistance": "RMSE_persistence",
                                                        "gain_vs_persistance": "gain_vs_persistence", "epoques": "epochs"}),
    "indice_capacite.csv": ("capacity_index.csv", {"graine": "seed", "epoques": "epochs", "drift_7j": "drift_7d"}),
    "xai_occultation.csv": ("occlusion.csv", {"groupe": "group", "hausse_RMSE": "RMSE_increase"}),
    "xai_gradients_integres.csv": ("integrated_gradients.csv", {"entree": "input", "part": "share"}),
    "xai_attention.csv": ("attention.csv", {"canal": "channel", "poids": "weight"}),
    "ridge.csv": ("ridge.csv", {}),
}


def convert(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    n_models = 0
    for folder in sorted(src.glob("PatchTST-*")):
        if not (folder / "resultats.json").exists():
            continue
        out = dst / folder.name
        out.mkdir(exist_ok=True)
        res = translate(json.loads((folder / "resultats.json").read_text()))
        (out / "results.json").write_text(json.dumps(res, indent=1))
        for f in folder.iterdir():
            n = f.name
            if n.startswith("pred_graine"):
                shutil.copy(f, out / n.replace("pred_graine", "pred_seed"))
            elif n.startswith("poids_graine"):
                sd = torch.load(f, weights_only=False)
                torch.save(translate_state_dict(sd), out / n.replace("poids_graine", "weights_seed"))
            elif n.startswith("historique_graine"):
                h = pd.read_csv(f).rename(columns={"epoque": "epoch", "perte_train": "train_loss", "perte_val": "val_loss"})
                h.to_csv(out / n.replace("historique_graine", "history_seed"), index=False)
            elif n == "vrai.npy":
                shutil.copy(f, out / "true.npy")
            elif n == "origines.npy":
                shutil.copy(f, out / "origins.npy")
        n_models += 1
    sel = json.loads((src / "selection.json").read_text())
    (dst / "selection.json").write_text(json.dumps({STEPS.get(k, k): v for k, v in sel.items()}, indent=1))
    fin_src, fin_dst = src / "final", dst / "final"
    if fin_src.exists():
        fin_dst.mkdir(exist_ok=True)
        for fr_name, (en_name, cols) in FINAL.items():
            if (fin_src / fr_name).exists():
                df = pd.read_csv(fin_src / fr_name).rename(columns=cols)
                if "group" in df:
                    df["group"] = df["group"].map(lambda g: GROUPS.get(g, g))
                if "input" in df:
                    df["input"] = df["input"].map(channel_label)
                if "channel" in df:
                    df["channel"] = df["channel"].map(channel_label)
                if "train" in df:
                    df["train"] = df["train"].astype(str)
                df.to_csv(fin_dst / en_name, index=False)
        if (fin_src / "ridge_pred.npy").exists():
            shutil.copy(fin_src / "ridge_pred.npy", fin_dst / "ridge_pred.npy")
    print(f"{n_models} configurations converted → {dst}")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
