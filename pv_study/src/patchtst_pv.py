# -*- coding: utf-8 -*-
"""
===============================================================================
 patchtst_pv.py — Core module of the study
 "Multivariate forecasting of the total photovoltaic production of the SENELEC
  grid with PatchTST, cyclic variables and solar geometry"
===============================================================================

Author      : Abdoulaye FAYE — UADB / EDTSS, Energy Efficiency and Energy
              Systems Research Team
Data        : energie.xlsx (SENELEC hourly operation log, 01/01/2019 → 09/05/2023)

Module layout (each block is self-contained and documented):

    0. GLOBAL CONFIGURATION ......... paths, split dates, column lists
    1. LOADING & CLEANING ........... hourly log, clipping, imputation, target
    2. COVARIATES ................... cyclic, solar geometry, grid context
    3. DATASET ...................... assembly, temporal split, normalisation
    4. WINDOWING .................... samples (past L, future H) without leakage
    5. PatchTST MODEL ............... RevIN, patches, encoder, CI / CA / X
    6. TRAINING & PREDICTION ........ loop, early stopping, inference
    7. METRICS & BASELINES .......... RMSE/MAE/nRMSE, persistence, mean, DM test
    8. EVALUATION PROTOCOL .......... 5 seeds, 5 % noise, 30-day rolling, efficiency
    9. DECISION SCORE ............... composite score of projet.md

Golden rules (projet.md), enforced by assertions in the code:
    R1. Never shuffle the time axis: Train < Val < Test.
    R2. No leakage: clipping, imputation and normalisation are fitted on Train.
    R3. Systematic comparison with persistence and with the moving average.
    R4. Individual PV plant productions are NEVER model inputs: they are only
        used to build the PV_total target.
===============================================================================
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =============================================================================
# 0. GLOBAL CONFIGURATION
# =============================================================================

#: Project root, derived from the location of this file.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "energie.xlsx"
CACHE_DIR = ROOT / "cache"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"
MODELS_DIR = ROOT / "models"

#: Strict temporal split (R1). Bounds are inclusive on the left.
VAL_START = pd.Timestamp("2022-01-01 00:00")
TEST_START = pd.Timestamp("2022-10-01 00:00")

#: Seeds used to measure dispersion (robustness criterion).
SEEDS = [0, 1, 2, 3, 4]

#: Individual PV plant columns: ONLY used to build the target.
PV_COLUMNS = [
    "PV-Mbour", "PV-CICAD", "PV Kahone", "PV ERS Kahone", "PV Scaling Kahone",
    "PV Kael Touba", "PV-Bokhol", "PV Mékhé Mérina", "PV S Mekhe Santhiou",
    "PV DIASS", "PV Sakal",
]

#: Thermal (and similar) units, aggregated into a single "Thermal" channel.
THERMAL_COLUMNS = [
    "106", "301", "303", "TAG2", "TAG4", "401", "402", "403", "404", "405",
    "Agg", "Agg1", "Agg2", "Sendou", "Koun", "M Power", "TP", "PSHIP", "APR",
    "CG", "Boutoute", "Kah1", "C7", "C6", "ICS", "Dang",
]

#: Columns excluded because they CONTAIN the PV production (indirect leakage).
EXCLUDED_COLUMNS = ["Total_unites", "RI", "RGI_1", "RGI_2", "RGI_3", "RGI_4", "RGI_5"]

#: Reference site: approximate centroid of the grid's PV plants.
LATITUDE, LONGITUDE, ALTITUDE = 14.9, -16.4, 30.0

#: Channel names (fixed order; channel 0 is ALWAYS the target).
TARGET = "PV_total"
GRID_CHANNELS = ["Thermal", "Taiba_wind", "Import_Manantali"]
CYCLIC_CHANNELS = ["h_sin", "h_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"]
SOLAR_CHANNELS = ["cos_zenith", "ghi_clear_sky"]
#: Deterministic covariates: also known in the future ("X" conditioning).
DETERMINISTIC_CHANNELS = CYCLIC_CHANNELS + SOLAR_CHANNELS
CHANNELS = [TARGET] + GRID_CHANNELS + DETERMINISTIC_CHANNELS


# =============================================================================
# 1. LOADING & CLEANING
# =============================================================================

def load_log(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the hourly log and re-index it on a complete hourly grid.

    The 288 hours missing from the file appear as NaN: they are imputed later,
    never dropped (the grid must stay regular for windowing).
    """
    df = pd.read_excel(path)
    df = df.set_index("Horodatage_debut").sort_index()
    grid = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(grid)
    df.index.name = "timestamp"
    return df


def existence_period(s: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last hour with production > 0 for a plant.

    Before commissioning (or after decommissioning) the production is 0 by
    definition; a NaN inside the existence period is a genuine gap.
    """
    active = s[s > 0]
    return active.index.min(), active.index.max()


def impute_series(s: pd.Series, max_gap: int = 3) -> pd.Series:
    """Causal two-step imputation.

    1) Linear interpolation of short gaps (<= `max_gap` hours).
    2) Long gaps: mean of the same hour over the 7 PREVIOUS days
       (rolling hourly profile — uses the past only).
    3) Remainder (start of the series): 0.
    """
    s = s.copy()
    # 1) short gaps: only short, bracketed holes are interpolated
    nan_mask = s.isna()
    block_id = (nan_mask != nan_mask.shift()).cumsum()
    block_size = nan_mask.groupby(block_id).transform("sum")
    short = nan_mask & (block_size <= max_gap)
    interp = s.interpolate(method="linear", limit_area="inside")
    s[short] = interp[short]
    # 2) long gaps: hourly profile of the 7 previous days (causal)
    if s.isna().any():
        frame = s.to_frame("v")
        frame["hour"] = frame.index.hour
        profile = (frame.groupby("hour")["v"]
                   .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean()))
        s = s.fillna(profile)
    # 3) remainder
    return s.fillna(0.0)


def build_target(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Build PV_total (MW) from the 11 plants, with full traceability.

    For each plant:
      a) clipping threshold = 1.2 × 99.9 % quantile computed on TRAIN only (R2)
         → a value above it (e.g. 958 MW at Mbour) is an entry error: NaN;
      b) outside the existence period: 0;
      c) gaps inside the existence period: causal imputation.
    Returns the target and a per-plant diagnostic table.
    """
    train = df.index < VAL_START
    diagnostics, series = [], {}
    for col in PV_COLUMNS:
        s = df[col].astype(float).clip(lower=0)
        start, end = existence_period(s)
        threshold = 1.2 * s[train].quantile(0.999) if s[train].notna().sum() > 100 \
            else 1.2 * s.quantile(0.999)
        n_outliers = int((s > threshold).sum())
        s[s > threshold] = np.nan
        exists = (s.index >= start) & (s.index <= end)
        n_gaps = int(s[exists].isna().sum())
        s[~exists] = 0.0
        s[exists] = impute_series(s[exists])
        series[col] = s
        diagnostics.append(dict(plant=col, commissioning=start, last_record=end,
                                threshold_MW=round(threshold, 2), clipped_values=n_outliers,
                                imputed_gaps=n_gaps,
                                mean_MW=round(s.mean(), 2), max_MW=round(s.max(), 2)))
    target = pd.DataFrame(series).sum(axis=1).rename(TARGET)
    return target, pd.DataFrame(diagnostics)


# =============================================================================
# 2. COVARIATES
# =============================================================================

def cyclic_covariates(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sine/cosine encodings: continuity 23 h → 0 h, 31/12 → 01/01, Sun → Mon.

    Hour (period 24), day of year (period 365.25) and day of week (period 7).
    The day of week may capture an operational effect (load shedding,
    maintenance) — it has no physical effect on irradiance, so its measured
    importance serves as a negative control in the XAI analysis.
    """
    h = index.hour.values + 0.5            # middle of the hour [t, t+1)
    doy = index.dayofyear.values - 1 + h / 24
    dow = index.dayofweek.values
    return pd.DataFrame({
        "h_sin": np.sin(2 * np.pi * h / 24), "h_cos": np.cos(2 * np.pi * h / 24),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
    }, index=index)


def solar_covariates(index: pd.DatetimeIndex, offset_h: float = 0.0) -> pd.DataFrame:
    """Solar geometry and clear-sky irradiance (pvlib), computed at the middle
    of each hour and shifted by `offset_h` hours to align the log time stamps
    with solar time (offset estimated on Train).

    - cos_zenith    : cosine of the apparent zenith angle, set to 0 at night;
    - ghi_clear_sky : clear-sky global horizontal irradiance (Haurwitz, W/m²).
    These variables are DETERMINISTIC: available without error at any horizon.
    """
    import pvlib
    t = (index + pd.Timedelta(minutes=30) + pd.Timedelta(hours=offset_h)).tz_localize("UTC")
    pos = pvlib.solarposition.get_solarposition(t, LATITUDE, LONGITUDE, ALTITUDE)
    ghi = pvlib.clearsky.haurwitz(pos["apparent_zenith"])["ghi"].values
    cz = np.clip(np.cos(np.deg2rad(pos["apparent_zenith"].values)), 0, None)
    return pd.DataFrame({"cos_zenith": cz, "ghi_clear_sky": ghi}, index=index)


def estimate_offset(target: pd.Series, candidates=np.arange(-2, 2.01, 0.25)) -> float:
    """Time offset (h) maximising the correlation between the mean daily
    profile of PV_total and that of clear-sky irradiance — estimated on TRAIN
    only. It absorbs the log's time-stamp convention (start/end of hour) and
    the gap between legal time and solar time."""
    train = target[target.index < VAL_START]
    pv_profile = train.groupby(train.index.hour).mean().values
    best, best_corr = 0.0, -1
    hours = pd.date_range("2020-01-01", "2020-12-31 23:00", freq="h")
    for d in candidates:
        cs = solar_covariates(hours, d)["ghi_clear_sky"]
        cs_profile = cs.groupby(cs.index.hour).mean().values
        c = np.corrcoef(pv_profile, cs_profile)[0, 1]
        if c > best_corr:
            best, best_corr = float(d), c
    return best


def grid_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Grid operating context, used as a "weather proxy".

    - Thermal          : sum of thermal units (responds to net demand, hence
                         indirectly to past cloudiness);
    - Taiba_wind       : wind production (wind regime, advection);
    - Import_Manantali : OMVS hydro import.
    These channels are only used over the PAST window (unknown in the future).
    """
    thermal = df[THERMAL_COLUMNS].sum(axis=1, min_count=1)
    wind = df["Eolien Taiba"].copy()
    wind_start, _ = existence_period(wind)
    wind[wind.index < wind_start] = 0.0
    out = pd.DataFrame({
        "Thermal": thermal,
        "Taiba_wind": wind.clip(lower=0),
        "Import_Manantali": df["Import_Manantali_Felou_Somelec"].clip(lower=0),
    }, index=df.index)
    # clipping of entry errors (e.g. Thermal = 17,316 MW): threshold
    # 1.2 × q99.9 computed on Train (R2), then causal imputation
    train = out.index < VAL_START
    for c in out:
        threshold = 1.2 * out.loc[train, c].quantile(0.999)
        out.loc[out[c] > threshold, c] = np.nan
        out[c] = impute_series(out[c])
    return out


# =============================================================================
# 3. DATASET: assembly, split, normalisation
# =============================================================================

@dataclass
class Dataset:
    """Container of the dataset, ready for windowing."""
    table: pd.DataFrame            # physical values (MW, unitless)
    values: np.ndarray             # normalised values float32 [T, C]
    mean: np.ndarray               # normalisation parameters (Train)
    std: np.ndarray
    i_val: int                     # first index of Val
    i_test: int                    # first index of Test
    solar_offset_h: float
    target_diagnostics: pd.DataFrame

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.table.index

    def to_mw(self, y_norm: np.ndarray) -> np.ndarray:
        """De-normalise values of the target channel back to MW."""
        return y_norm * self.std[0] + self.mean[0]


def build_dataset(path: Path | str = DATA_PATH, cache: bool = True) -> Dataset:
    """Complete preparation pipeline (cached on disk)."""
    cache_file = CACHE_DIR / "dataset.pkl"
    if cache and cache_file.exists():
        return pd.read_pickle(cache_file)

    df = load_log(path)
    target, diagnostics = build_target(df)
    offset = estimate_offset(target)
    table = pd.concat([target, grid_covariates(df),
                       cyclic_covariates(df.index),
                       solar_covariates(df.index, offset)], axis=1)[CHANNELS]

    # --- R4: no individual PV column nor any "leaking" column as input
    assert not set(PV_COLUMNS + EXCLUDED_COLUMNS) & set(table.columns)
    assert table.notna().all().all(), "NaN values remain after imputation"

    i_val = int(np.searchsorted(table.index, VAL_START))
    i_test = int(np.searchsorted(table.index, TEST_START))
    # --- R1: strict order of the segments
    assert 0 < i_val < i_test < len(table)
    assert table.index[i_val - 1] < VAL_START <= table.index[i_val]

    # --- R2: normalisation fitted on Train only
    raw = table.values.astype(np.float64)
    mean = raw[:i_val].mean(axis=0)
    std = raw[:i_val].std(axis=0)
    std[std < 1e-8] = 1.0
    values = ((raw - mean) / std).astype(np.float32)

    ds = Dataset(table, values, mean, std, i_val, i_test, offset, diagnostics)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(ds, cache_file)
    return ds


# =============================================================================
# 4. WINDOWING
# =============================================================================
#
# Convention: a forecast "origin" t means that the last observed hour is t-1.
# The model receives X[t-L : t] (all channels, past) and, for the X variant,
# the deterministic covariates F[t : t+H] (known future). It predicts the
# target Y[t : t+H]. No target or grid value later than t-1 enters the model.

def origins(start: int, end: int, L: int, H: int, step: int = 1) -> np.ndarray:
    """Admissible origins whose past and future fit within [start-L, end)."""
    first = max(start, L)
    last = end - H          # inclusive
    return np.arange(first, last + 1, step, dtype=np.int64)


class WindowDataset(torch.utils.data.Dataset):
    """Samples (past, deterministic future, target) for a set of origins."""

    def __init__(self, values: np.ndarray, orig: np.ndarray, L: int, H: int):
        self.v = torch.from_numpy(values)
        self.orig, self.L, self.H = orig, L, H
        self.i_det = [CHANNELS.index(c) for c in DETERMINISTIC_CHANNELS]

    def __len__(self):
        return len(self.orig)

    def __getitem__(self, k):
        t = int(self.orig[k])
        past = self.v[t - self.L:t]                        # [L, C]
        future = self.v[t:t + self.H][:, self.i_det]       # [H, C_det]
        target = self.v[t:t + self.H, 0]                   # [H]
        return past, future, target


# =============================================================================
# 5. PatchTST MODEL
# =============================================================================

class RevIN(nn.Module):
    """Reversible instance normalisation (Kim et al., ICLR 2022).

    Each window is standardised channel by channel; the forecast is brought
    back to the scale of the input window of the target channel. Mitigates
    non-stationarity (growth of the installed capacity).

    RevIN is applied to the STOCHASTIC channels only (target and grid). The
    deterministic covariates keep their global normalisation: over a window
    of a few days the day of year barely changes, and standardising it would
    erase the useful information (the position in the year), replacing it
    with an artificial ramp.
    """

    def __init__(self, n_channels: int, n_stochastic: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.k = n_stochastic               # channels 0..k-1 normalised per window
        self.gamma = nn.Parameter(torch.ones(n_stochastic))
        self.beta = nn.Parameter(torch.zeros(n_stochastic))

    def normalize(self, x):                 # x: [B, L, C]
        xs = x[:, :, :self.k]
        self.mu = xs.mean(dim=1, keepdim=True).detach()
        self.sigma = torch.sqrt(xs.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        xs = (xs - self.mu) / self.sigma * self.gamma + self.beta
        return torch.cat([xs, x[:, :, self.k:]], dim=-1)

    def denormalize_target(self, y):        # y: [B, H] (channel 0)
        y = (y - self.beta[0]) / (self.gamma[0] + self.eps)
        return y * self.sigma[:, 0, 0:1] + self.mu[:, 0, 0:1]


@dataclass
class PatchTSTConfig:
    """Hyper-parameters of a configuration (the 4 factors of projet.md + options)."""
    P: int = 24                  # patch length (temporal granularity)
    L: int = 168                 # history (look-back)
    H: int = 24                  # forecast horizon
    channel_mode: str = "CA"     # "CI" (independence) or "CA" (cross-channel attention)
    known_future: bool = False   # conditioning on future deterministic covariates
    physical_gate: bool = False  # output set to zero when the sun is down
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    # training
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 25
    patience: int = 4
    train_step: int = 1          # every hourly origin is a candidate…
    epoch_fraction: float = 0.2  # …but 20 % are drawn at random each epoch (CPU cost)
    val_step: int = 6

    @property
    def stride(self) -> int:
        """Stride between patches: half the patch (50 % overlap, Nie et al. 2023)."""
        return max(1, self.P // 2)

    @property
    def n_patches(self) -> int:
        return (self.L - self.P) // self.stride + 1

    def name(self) -> str:
        suffix = ("X" if self.known_future else "") + ("g" if self.physical_gate else "")
        return f"PatchTST-{self.channel_mode}{suffix}_P{self.P}_L{self.L}_H{self.H}"


class PatchTST(nn.Module):
    """Multivariate PatchTST with three channel-handling variants.

    Computation chain (B = batch, C = channels, N = patches, D = d_model):

      x [B, L, C] ──RevIN──► patching [B, C, N, P]
         ──linear projection P→D + position──► [B·C, N, D]
         ──Transformer encoder SHARED across channels (temporal attention)──►
      ┌─ CI: only the target channel is kept → [B, N·D] → linear head → H
      │      (exogenous channels therefore do NOT influence the forecast:
      │       this is equivalent to a univariate PatchTST with shared weights)
      └─ CA: CROSS-CHANNEL attention at each patch position [B·N, C, D]
             (the target channel "queries" the exogenous channels), then the
             target channel → [B, N·D] → linear head → H
      Option X: step-wise correction from the FUTURE deterministic covariates
             (hour, season, solar geometry of the forecast time).
      Option g: physical gate — forecast forced to 0 when future cos_zenith = 0.
    """

    def __init__(self, cfg: PatchTSTConfig, n_channels: int = len(CHANNELS),
                 n_det: int = len(DETERMINISTIC_CHANNELS)):
        super().__init__()
        self.cfg = cfg
        self.C = n_channels if cfg.channel_mode == "CA" else 1
        D, N = cfg.d_model, cfg.n_patches
        self.revin = RevIN(self.C, 1 + len(GRID_CHANNELS) if cfg.channel_mode == "CA" else 1)
        # --- patch embedding (shared across channels) + learned position
        self.projection = nn.Linear(cfg.P, D)
        self.position = nn.Parameter(torch.randn(1, N, D) * 0.02)
        # --- channel identity embedding (only useful in CA)
        self.channel_id = nn.Parameter(torch.randn(1, self.C, 1, D) * 0.02)
        self.dropout = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(D, cfg.n_heads, cfg.d_ff, cfg.dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.temporal_encoder = nn.TransformerEncoder(layer, cfg.n_layers,
                                                      enable_nested_tensor=False)
        if cfg.channel_mode == "CA":
            self.channel_attention = nn.MultiheadAttention(D, cfg.n_heads, dropout=cfg.dropout,
                                                           batch_first=True)
            self.channel_norm = nn.LayerNorm(D)
        # --- direct multi-horizon forecasting head
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(cfg.dropout), nn.Linear(N * D, cfg.H))
        # --- option X: correction conditioned on the deterministic future
        if cfg.known_future:
            self.context = nn.Linear(N * D, 32)
            self.correction = nn.Sequential(nn.Linear(32 + n_det + 1, 64), nn.GELU(),
                                            nn.Linear(64, 1))
        self.i_cz = DETERMINISTIC_CHANNELS.index("cos_zenith")
        self.last_channel_weights = None    # for the XAI analysis

    def forward(self, past, future=None, raw_future_cz=None):
        cfg = self.cfg
        x = past if cfg.channel_mode == "CA" else past[:, :, :1]      # [B, L, C]
        x = self.revin.normalize(x)
        B = x.shape[0]
        # patching: [B, C, N, P]
        patches = x.permute(0, 2, 1).unfold(-1, cfg.P, cfg.stride)
        z = self.projection(patches) + self.position.unsqueeze(1)     # [B, C, N, D]
        z = z + self.channel_id
        z = self.dropout(z)
        N, D = z.shape[2], z.shape[3]
        z = self.temporal_encoder(z.reshape(B * self.C, N, D)).reshape(B, self.C, N, D)
        if cfg.channel_mode == "CA":
            # cross-channel attention at each patch position
            zc = z.permute(0, 2, 1, 3).reshape(B * N, self.C, D)      # [B·N, C, D]
            query = zc[:, :1]                                         # target channel
            out, weights = self.channel_attention(query, zc, zc, need_weights=True)
            self.last_channel_weights = weights.detach().reshape(B, N, self.C)
            zt = self.channel_norm(query + out).reshape(B, N, D)
        else:
            zt = z[:, 0]                                              # [B, N, D]
        y = self.head(zt)                                             # [B, H]
        if cfg.known_future:
            ctx = self.context(zt.flatten(1))                         # [B, 32]
            ctx = ctx.unsqueeze(1).expand(-1, cfg.H, -1)
            base = y.unsqueeze(-1)                                    # base forecast
            y = y + self.correction(torch.cat([ctx, future, base], dim=-1)).squeeze(-1)
        y = self.revin.denormalize_target(y)
        if cfg.physical_gate and raw_future_cz is not None:
            y = y * (raw_future_cz > 0).float()
        return y


# =============================================================================
# 6. TRAINING & PREDICTION
# =============================================================================

def set_seed(seed: int):
    """Reproducibility (numpy, torch, deterministic algorithms)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _raw_cos_zenith(ds: Dataset, future_norm: torch.Tensor) -> torch.Tensor:
    """Future cos_zenith brought back to physical scale (for the physical gate)."""
    i = CHANNELS.index("cos_zenith")
    j = DETERMINISTIC_CHANNELS.index("cos_zenith")
    return future_norm[..., j] * ds.std[i] + ds.mean[i]


def mse_loss(ds, y_pred, y_true):
    """Mean squared error in normalised units."""
    return torch.mean((y_pred - y_true) ** 2)


def train_model(cfg: PatchTSTConfig, ds: Dataset, seed: int, verbose: bool = False):
    """Train a model on Train with early stopping on Val (Test never seen).

    Adam optimiser, learning rate reduced on plateau, gradient clipping,
    restoration of the best weights.
    """
    set_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    o_train = origins(0, ds.i_val, cfg.L, cfg.H, cfg.train_step)
    o_val = origins(ds.i_val, ds.i_test, cfg.L, cfg.H, cfg.val_step)
    # R1: no training target spills over Val, and no Val target over Test
    assert o_train.max() + cfg.H <= ds.i_val and o_val.max() + cfg.H <= ds.i_test
    g = torch.Generator().manual_seed(seed)
    # Random sub-sampling of the origins at each epoch: every hourly phase
    # (0 h … 23 h) stays represented, unlike a fixed step (a 3-hour step would
    # only show 8 phases out of 24 to the model).
    sampler = torch.utils.data.RandomSampler(
        range(len(o_train)), replacement=False,
        num_samples=int(cfg.epoch_fraction * len(o_train)), generator=g)
    dl_train = torch.utils.data.DataLoader(WindowDataset(ds.values, o_train, cfg.L, cfg.H),
                                           batch_size=cfg.batch_size, sampler=sampler)
    # NB: random drawing only concerns the TRAINING windows within an epoch
    # (stochastic gradient descent); the temporal split Train < Val < Test
    # remains strict (R1).
    dl_val = torch.utils.data.DataLoader(WindowDataset(ds.values, o_val, cfg.L, cfg.H),
                                         batch_size=512, shuffle=False)
    model = PatchTST(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_weights, wait = math.inf, None, 0
    history = []
    t0 = time.time()
    for epoch in range(cfg.max_epochs):
        model.train()
        losses = []
        for past, future, target in dl_train:
            opt.zero_grad()
            pred = model(past, future, _raw_cos_zenith(ds, future))
            loss = mse_loss(ds, pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            lv = [mse_loss(ds, model(p, f, _raw_cos_zenith(ds, f)), c).item() * len(c)
                  for p, f, c in dl_val]
        val_loss = sum(lv) / len(o_val)
        sched.step(val_loss)
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), val_loss=val_loss))
        if verbose:
            print(f"  epoch {epoch+1:2d}  train={np.mean(losses):.4f}  val={val_loss:.4f}")
        if val_loss < best - 1e-5:
            best, best_weights, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_weights)
    model.eval()
    return model, pd.DataFrame(history), time.time() - t0


@torch.no_grad()
def predict(model: PatchTST, ds: Dataset, orig: np.ndarray,
            values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts and observations (MW) for a list of origins: [n, H]."""
    cfg = model.cfg
    v = ds.values if values is None else values
    dl = torch.utils.data.DataLoader(WindowDataset(v, orig, cfg.L, cfg.H),
                                     batch_size=512, shuffle=False)
    preds = []
    for past, future, _ in dl:
        preds.append(model(past, future, _raw_cos_zenith(ds, future)).numpy())
    y_pred = ds.to_mw(np.concatenate(preds))
    # observations are read from the ORIGINAL (noise-free) data
    y_true = np.stack([ds.table[TARGET].values[t:t + cfg.H] for t in orig])
    return np.clip(y_pred, 0, None), y_true


# =============================================================================
# 7. METRICS & BASELINES
# =============================================================================

def metrics(y_true: np.ndarray, y_pred: np.ndarray, ref_mean: float | None = None,
            mask: np.ndarray | None = None) -> dict:
    """RMSE, MAE, nRMSE = RMSE / mean(production), R², MBE (in MW).

    `mask` restricts the computation (e.g. daylight hours).
    """
    yt, yp = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    if mask is not None:
        m = np.asarray(mask).ravel()
        yt, yp = yt[m], yp[m]
    e = yp - yt
    rmse = float(np.sqrt(np.mean(e ** 2)))
    ref = float(np.mean(yt)) if ref_mean is None else ref_mean
    return dict(RMSE=rmse, MAE=float(np.mean(np.abs(e))), nRMSE=rmse / ref,
                R2=float(1 - np.sum(e ** 2) / np.sum((yt - yt.mean()) ** 2)),
                MBE=float(np.mean(e)))


def daily_persistence(ds: Dataset, orig: np.ndarray, H: int) -> np.ndarray:
    """Daily persistence: ŷ(t+h) = y(same hour, last observed day).

    For h ≤ 24: y(t+h-24); beyond that, the last observed day is repeated.
    """
    y = ds.table[TARGET].values
    h = np.arange(H)
    return np.stack([y[t - 24 + (h % 24)] for t in orig])


def seven_day_mean(ds: Dataset, orig: np.ndarray, H: int, days: int = 7) -> np.ndarray:
    """Moving average: ŷ(t+h) = mean of the last `days` days at the same hour."""
    y = ds.table[TARGET].values
    h = np.arange(H)
    return np.stack([np.mean([y[t - 24 * k + (h % 24)] for k in range(1, days + 1)], axis=0)
                     for t in orig])


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test (squared loss) with the Harvey-Leybourne-Newbold
    correction. H0: equal accuracy. Stat < 0 ⇒ model 1 is better.

    e1, e2: errors [n, H]; losses are aggregated per origin (mean of squares
    over the horizon) to obtain a loss series, then the Newey-West long-run
    variance with h-1 lags is used.
    """
    from scipy import stats
    d = (np.asarray(e1) ** 2).mean(axis=-1) - (np.asarray(e2) ** 2).mean(axis=-1)
    n = len(d)
    d_bar = d.mean()
    gamma = [np.sum((d[k:] - d_bar) * (d[:n - k] - d_bar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = d_bar / math.sqrt(max(var, 1e-12))
    stat *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)      # HLN correction
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


# =============================================================================
# 8. EVALUATION PROTOCOL OF A CONFIGURATION
# =============================================================================

def perturb_exogenous(ds: Dataset, level: float = 0.05, seed: int = 123) -> np.ndarray:
    """Robustness test: Gaussian noise of level 5 % (σ = 5 % of the Train
    standard deviation of each channel) added to ALL exogenous channels (past
    grid and deterministic covariates) — the target is not perturbed. With no
    weather in the log, the covariates play the role of the "weather" of
    projet.md."""
    rng = np.random.default_rng(seed)
    v = ds.values.copy()
    v[:, 1:] += rng.normal(0, level, size=v[:, 1:].shape).astype(np.float32)
    return v


def daily_rmse(y_true, y_pred, orig, ds: Dataset) -> pd.Series:
    """RMSE per calendar day of the forecast origin."""
    days = ds.index[orig].normalize()
    e2 = ((y_pred - y_true) ** 2).mean(axis=1)
    return pd.Series(e2, index=days).groupby(level=0).mean().pipe(np.sqrt)


def measure_efficiency(model: PatchTST, ds: Dataset, repetitions: int = 50) -> dict:
    """Inference time of ONE sample (ms, median) and model size (MB)."""
    cfg = model.cfg
    wds = WindowDataset(ds.values, np.array([ds.i_test]), cfg.L, cfg.H)
    past, future, _ = wds[0]
    past, future = past[None], future[None]
    cz = _raw_cos_zenith(ds, future)
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(5):
            model(past, future, cz)
        times = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            model(past, future, cz)
            times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return dict(inference_ms=float(np.median(times)), size_MB=buf.tell() / 2 ** 20,
                n_parameters=int(sum(p.numel() for p in model.parameters())))


def evaluate_configuration(cfg: PatchTSTConfig, ds: Dataset, seeds=SEEDS,
                           rolling_days: int = 30, force: bool = False,
                           verbose: bool = True) -> dict:
    """Evaluate a configuration against the 4 criteria of projet.md.

    For each seed: training, forecasts over the whole Test set (every hourly
    origin), 5 % noise test, 30-day rolling forecast without re-training.
    Results (and weights) are cached: delete the folder cache/<name> to
    re-train.
    """
    folder = CACHE_DIR / cfg.name()
    file = folder / "results.json"
    if file.exists() and not force:
        return json.loads(file.read_text())
    folder.mkdir(parents=True, exist_ok=True)

    o_test = origins(ds.i_test, len(ds.values), cfg.L, cfg.H, 1)
    rolling_end = ds.i_test + 24 * rolling_days
    o_roll = o_test[o_test < rolling_end]
    v_noise = perturb_exogenous(ds)
    per_seed = []
    for seed in seeds:
        model, hist, duration = train_model(cfg, ds, seed)
        y_pred, y_true = predict(model, ds, o_test)
        y_noise, _ = predict(model, ds, o_test, v_noise)
        day = (y_true > 0) | (y_pred > 0)
        m = metrics(y_true, y_pred)
        m_day = metrics(y_true, y_pred, ref_mean=float(y_true.mean()), mask=day)
        rmse_noise = metrics(y_true, y_noise)["RMSE"]
        # 30-day rolling: forecasts whose origin lies within the first 30 days
        k = np.isin(o_test, o_roll)
        rd = daily_rmse(y_true[k], y_pred[k], o_roll, ds)
        slope = float(np.polyfit(np.arange(len(rd)), rd.values, 1)[0])
        eff = measure_efficiency(model, ds)
        per_seed.append(dict(
            seed=seed, duration_s=duration, epochs=len(hist), **m,
            RMSE_day=m_day["RMSE"], nRMSE_day=m_day["nRMSE"],
            RMSE_noise=rmse_noise, noise_degradation=rmse_noise / m["RMSE"] - 1,
            drift_D30_D1=float(rd.iloc[-1] / rd.iloc[0]),
            drift_7d=float(rd.iloc[-7:].mean() / rd.iloc[:7].mean()),
            slope_MW_per_day=slope, daily_rmse=rd.round(4).tolist(), **eff))
        np.save(folder / f"pred_seed{seed}.npy", y_pred.astype(np.float32))
        torch.save(model.state_dict(), folder / f"weights_seed{seed}.pt")
        hist.to_csv(folder / f"history_seed{seed}.csv", index=False)
        if verbose:
            print(f"  {cfg.name()}  seed {seed}: RMSE={m['RMSE']:.3f} MW  "
                  f"nRMSE={m['nRMSE']:.3f}  noise={per_seed[-1]['noise_degradation']:+.1%}  "
                  f"drift7d={per_seed[-1]['drift_7d']:.2f}  ({duration:.0f}s, {len(hist)} ep.)")
    np.save(folder / "true.npy", y_true.astype(np.float32))
    np.save(folder / "origins.npy", o_test)
    res = dict(config=asdict(cfg), name=cfg.name(), per_seed=per_seed,
               summary=summarize_seeds(per_seed))
    file.write_text(json.dumps(res, indent=1, default=str))
    return res


def summarize_seeds(per_seed: list[dict]) -> dict:
    """Means and standard deviations (ddof=1) over the seeds."""
    df = pd.DataFrame(per_seed)
    keys = ["RMSE", "MAE", "nRMSE", "R2", "MBE", "RMSE_day", "nRMSE_day", "noise_degradation",
            "drift_D30_D1", "drift_7d", "slope_MW_per_day", "inference_ms", "size_MB",
            "n_parameters", "duration_s", "epochs"]
    s = {f"{k}_mean": float(df[k].mean()) for k in keys}
    s.update({f"{k}_std": float(df[k].std(ddof=1)) for k in ["RMSE", "MAE", "nRMSE", "R2"]})
    return s


# =============================================================================
# 9. DECISION SCORE (projet.md)
# =============================================================================

def decision_score(summary: dict) -> dict:
    """Score = 0.4·(1/nRMSE) + 0.3·(1/std) + 0.3·(1/drift) + efficiency bonus.

    - nRMSE : mean over 5 seeds (all hours);
    - std   : standard deviation of the RMSE (MW) over 5 seeds;
    - drift : smoothed ratio RMSE(D24–D30)/RMSE(D1–D7) — the raw D30/D1 ratio
              depends on the cloudiness of two isolated days and is reported
              for information only;
    - bonus : +0.1 if inference < 100 ms, +0.1 if weights < 50 MB;
    - rejection if the degradation under 5 % noise exceeds 15 %.
    """
    std = max(summary["RMSE_std"], 1e-3)
    bonus = 0.1 * (summary["inference_ms_mean"] < 100) + 0.1 * (summary["size_MB_mean"] < 50)
    score = (0.4 / summary["nRMSE_mean"] + 0.3 / std + 0.3 / summary["drift_7d_mean"] + bonus)
    return dict(score=float(score), bonus=float(bonus),
                rejected_noise=bool(summary["noise_degradation_mean"] > 0.15))


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """Comparison table (one row per configuration) with raw and normalised scores.

    The normalised score maps each criterion to [0, 1] (min-max across the
    compared configurations) before the 40/30/30 weighting, so that a
    larger-scale term (1/std) does not dominate. For the drift, only a
    DEGRADATION is penalised (drift > 1): an RMSE decrease over the month
    reflects the weather (onset of the dry season), not a merit of the model.
    """
    rows = []
    for r in results:
        s = r["summary"]
        sc = decision_score(s)
        rows.append(dict(configuration=r["name"], RMSE=s["RMSE_mean"], RMSE_std=s["RMSE_std"],
                         MAE=s["MAE_mean"], nRMSE=s["nRMSE_mean"], nRMSE_day=s["nRMSE_day_mean"],
                         R2=s["R2_mean"], MBE=s["MBE_mean"],
                         noise_degradation=s["noise_degradation_mean"],
                         drift_D30_D1=s["drift_D30_D1_mean"], drift_7d=s["drift_7d_mean"],
                         inference_ms=s["inference_ms_mean"], size_MB=s["size_MB_mean"],
                         n_parameters=int(s["n_parameters_mean"]), **sc))
    df = pd.DataFrame(rows)

    def mm(x, inverse=False):
        x = x.astype(float)
        r = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0 + 1
        return 1 - r if inverse else r
    df["normalised_score"] = (0.4 * mm(df["nRMSE"], True) + 0.3 * mm(df["RMSE_std"], True)
                              + 0.3 * mm(df["drift_7d"].clip(lower=1.0), True) + df["bonus"] / 2)
    return df
