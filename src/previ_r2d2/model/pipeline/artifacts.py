"""Assemblage des métriques (results.json) et écriture des artefacts
d'entraînement -- port fidèle de train_meta.py:600-680 (Previ_v2, section
assemblage résultats). log_interpretability a déjà perdu son paramètre
output_dir vestigial (pièce métriques déjà portée) -- appelé sans ce
paramètre ici."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from previ_r2d2.model.architectures.stacking import kge_components
from previ_r2d2.model.pipeline.metrics import kge_by_regime, kge_by_season, kge_per_step, log_interpretability


def nan_safe(v):
    """Convertit NaN/None en None, arrondit sinon -- pour la sérialisation JSON."""
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4)


def deep_nan_safe(obj):
    """Applique nan_safe récursivement sur un dict/list imbriqué."""
    if isinstance(obj, float):
        return nan_safe(obj)
    if isinstance(obj, dict):
        return {k: deep_nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_nan_safe(v) for v in obj]
    return obj


def build_results(
    dossier: str,
    horizon: int,
    meta_type: str,
    yt_v: np.ndarray,
    pl_v: np.ndarray,
    pt_v: np.ndarray,
    stk_v: np.ndarray,
    q_now_v: np.ndarray,
    df_sub: pd.DataFrame,
    y_test: np.ndarray,
    pred_lgbm_multi_test: np.ndarray,
    pred_lstm_test: np.ndarray,
    pred_stacking_multi: np.ndarray,
    meta,
    horizon_steps: int,
    meteo_feature_cols: list[str],
) -> dict:
    """Assemble le dict results.json complet (métriques globales, par pas, par régime/saison, interprétabilité)."""
    metrics = {
        "LightGBM seul": kge_components(yt_v, pl_v),
        "BiLSTM seul": kge_components(yt_v, pt_v),
        "Stacking": kge_components(yt_v, stk_v),
    }

    pred_stacking_log = np.log1p(np.clip(pred_stacking_multi, 0, None))
    lgbm_by_step = kge_per_step(y_test, pred_lgbm_multi_test, horizon_steps)
    lstm_by_step = kge_per_step(y_test, pred_lstm_test, horizon_steps)
    stack_by_step = kge_per_step(y_test, pred_stacking_log, horizon_steps)

    q33 = float(np.nanpercentile(yt_v, 33))
    q90 = float(np.nanpercentile(yt_v, 90))
    regime_results = kge_by_regime(yt_v, {"LightGBM seul": pl_v, "BiLSTM seul": pt_v, "Stacking": stk_v}, q33, q90)

    dates_sub = pd.DatetimeIndex(df_sub.index)
    season_results = kge_by_season(yt_v, {"LightGBM seul": pl_v, "BiLSTM seul": pt_v, "Stacking": stk_v}, dates_sub)

    interp = log_interpretability(meta, yt_v, pl_v, pt_v, stk_v, q_now_v, horizon_steps, meteo_feature_cols)

    return {
        "centrale": dossier,
        "horizon": horizon,
        "meta_type": meta_type,
        "kge_lgbm": nan_safe(metrics["LightGBM seul"]["kge"]),
        "kge_lstm": nan_safe(metrics["BiLSTM seul"]["kge"]),
        "kge_stacking": nan_safe(metrics["Stacking"]["kge"]),
        "components": {name: {k: nan_safe(v) for k, v in m.items()} for name, m in metrics.items()},
        "kge_by_step": {
            "lgbm": [nan_safe(s["kge"]) for s in lgbm_by_step],
            "lstm": [nan_safe(s["kge"]) for s in lstm_by_step],
            "stack": [nan_safe(s["kge"]) for s in stack_by_step],
            "rmse_m3s_stack": [nan_safe(s["rmse_m3s"]) for s in stack_by_step],
        },
        "kge_by_regime": {name: {reg: {k: nan_safe(v) for k, v in d.items()} for reg, d in rr.items()} for name, rr in regime_results.items()},
        "kge_by_season": {name: {saison: {k: nan_safe(v) for k, v in d.items()} for saison, d in sr.items()} for name, sr in season_results.items()},
        "quantiles_test": {"q33_m3s": nan_safe(q33), "q90_m3s": nan_safe(q90)},
        "interpretability": deep_nan_safe(interp),
    }


def write_artifacts(
    results: dict,
    meta_config: dict,
    yt_v: np.ndarray,
    pl_v: np.ndarray,
    pt_v: np.ndarray,
    stk_v: np.ndarray,
    df_sub: pd.DataFrame,
    dossier: str,
    horizon: int,
    weights_dir: Path,
    outputs_dir: Path,
) -> None:
    """Écrit results.json/meta_config.json (weights_dir) et le CSV de prédictions test (outputs_dir)."""
    with open(weights_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    with open(weights_dir / "meta_config.json", "w") as fh:
        json.dump(meta_config, fh, indent=2)

    pd.DataFrame({
        "datetime": df_sub.index,
        "q_obs_m3s": yt_v,
        "q_lgbm_m3s": pl_v,
        "q_lstm_m3s": pt_v,
        "q_stacking_m3s": stk_v,
    }).to_csv(outputs_dir / f"hybrid_test_{dossier}_{horizon}h.csv", index=False)
