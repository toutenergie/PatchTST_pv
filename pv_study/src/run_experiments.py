# -*- coding: utf-8 -*-
"""
run_experiments.py — Sequential execution of the 4 configurations (one factor at a time).

Each step carries over the best value of the previous step (normalised
decision score; configurations rejected by the noise test are excluded).
Results are cached by patchtst_pv.evaluate_configuration: the script can be
interrupted and restarted without loss.

    Config 1 : P ∈ {12, 24, 48}                         (L=168, CA-X, H=24)
    Config 2 : L ∈ {96, 168, 336}                       (P*)
    Config 3 : channels ∈ {CI, CI-X, CA, CA-X, CA-Xg}   (P*, L*)
    Config 4 : H ∈ {1, 6, 12, 24, 168}                  (P*, L*, channels*)
"""
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patchtst_pv as m  # noqa: E402

ds = m.build_dataset()
log = m.CACHE_DIR / "selection.json"
selection = json.loads(log.read_text()) if log.exists() else {}


def step(name: str, configs: list) -> m.PatchTSTConfig:
    """Evaluate a list of configurations and return the best one."""
    print(f"\n=== {name} ===", flush=True)
    results = [m.evaluate_configuration(c, ds) for c in configs]
    table = m.comparison_table(results)
    m.TABLES_DIR.mkdir(exist_ok=True)
    table.to_csv(m.TABLES_DIR / f"{name}.csv", index=False)
    # admissible = not rejected by the 5 % noise test (if at least one remains)
    admissible = table[~table["rejected_noise"]] if (~table["rejected_noise"]).any() else table
    best = admissible.sort_values("normalised_score", ascending=False).iloc[0]["configuration"]
    print(table[["configuration", "RMSE", "RMSE_std", "nRMSE", "noise_degradation", "drift_7d",
                 "score", "normalised_score"]].round(4).to_string(), flush=True)
    print(f"--> selected: {best}", flush=True)
    selection[name] = best
    log.write_text(json.dumps(selection, indent=1))
    return configs[[c.name() for c in configs].index(best)]


if __name__ == "__main__":
    base = m.PatchTSTConfig(P=24, L=168, H=24, channel_mode="CA", known_future=True)

    c1 = step("config1_patch", [dataclasses.replace(base, P=p) for p in (12, 24, 48)])
    c2 = step("config2_history", [dataclasses.replace(c1, L=l) for l in (96, 168, 336)])
    c3 = step("config3_channels", [
        dataclasses.replace(c2, channel_mode="CI", known_future=False, physical_gate=False),
        dataclasses.replace(c2, channel_mode="CI", known_future=True, physical_gate=False),
        dataclasses.replace(c2, channel_mode="CA", known_future=False, physical_gate=False),
        dataclasses.replace(c2, channel_mode="CA", known_future=True, physical_gate=False),
        dataclasses.replace(c2, channel_mode="CA", known_future=True, physical_gate=True),
    ])
    c4 = step("config4_horizon", [dataclasses.replace(c3, H=h) for h in (1, 6, 12, 24, 168)])
    print("\nDONE — selection:", json.dumps(selection, indent=1), flush=True)
