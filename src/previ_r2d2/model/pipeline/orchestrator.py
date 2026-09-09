"""Orchestrateur d'entraînement -- port fidèle de run_one/run_train_meta
(train_meta.py, Previ_v2) en une fonction pure appelant dans l'ordre les
briques déjà portées. Pas de plots (SHAP/attention/comparaison), pas de
MLflow/DVCLive (mlflow_run_id=None) -- portée réduite actée."""

from __future__ import annotations


from pathlib import Path

import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.model.architectures.bilstm.model import BiLSTMHydro
from previ_r2d2.model.architectures.bilstm.sequences import build_sequences, get_seq_cols
from previ_r2d2.model.architectures.lightgbm.features import build_features
from previ_r2d2.model.architectures.lightgbm.predict import predict_lgbm_full
from previ_r2d2.model.architectures.stacking import meteo_cols, slice_lgbm_multistep
from previ_r2d2.model.features.amont import shift_amont_columns
from previ_r2d2.model.pipeline.artifacts import build_results, write_artifacts
from previ_r2d2.model.pipeline.bv_config import bv_params_from_bv_json, transit_amont_from_bv_json
from previ_r2d2.model.pipeline.data_loading import load_df, resample_to_daily, split_train_test
from previ_r2d2.model.pipeline.oof_cache import (
    load_or_compute_lgbm_final,
    load_or_compute_oof_lgbm,
    load_or_compute_oof_lstm,
)
from previ_r2d2.model.pipeline.plots import generate_training_plots
from previ_r2d2.model.pipeline.predict import predict_test_set
from previ_r2d2.model.pipeline.stacking_fit import fit_stacking
from previ_r2d2.model.seeding import set_seeds

HORIZON_CFG = {
    8: {"horizon_steps": 8, "timestep": "hourly", "steps_per_day": 24},
    48: {"horizon_steps": 2, "timestep": "1D", "steps_per_day": 1},
    72: {"horizon_steps": 3, "timestep": "1D", "steps_per_day": 1},
}
MULT_POIDS = 4  # constante fixe Previ_v2 (lightgbm_model.py:289), jamais surchargée


def run_training(
    dossier: str,
    horizon: int,
    exutoire: dict,
    bv_json: dict,
    meta_type: str = "ridge",
    epochs: int = 40,
    n_trials_lgbm: int = 30,
    n_trials_final: int = 50,
    force_lgbm: bool = False,
    force_lstm: bool = False,
    weights_dir: Path | None = None,
) -> dict:
    """Orchestre un entraînement complet (une centrale, un horizon) : chargement, OOF, fit final, fit Stacking, évaluation test, artefacts persistés."""
    cfg = HORIZON_CFG[horizon]
    horizon_steps, timestep, steps_per_day = cfg["horizon_steps"], cfg["timestep"], cfg["steps_per_day"]
    n_splits_eff = 2 if timestep == "1D" else 3

    bv_params = bv_params_from_bv_json(bv_json)
    transit_amont = transit_amont_from_bv_json(bv_json)

    # Avant toute construction de modèle : sinon l'initialisation des poids
    # BiLSTM et le mélange des batches rendent deux runs identiques
    # incomparables (~0.14 KGE d'écart mesuré).
    seed = set_seeds()

    weights_dir = weights_dir or (config.ROOT / "weights" / "hybrid" / dossier / f"h{horizon}")
    outputs_dir = config.ROOT / "outputs" / "hybrid" / dossier / f"h{horizon}"
    weights_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(dossier)
    if timestep == "1D":
        df = resample_to_daily(df)
    df_train, df_test = split_train_test(df)

    X_train, y_train, _ = build_features(df_train, exutoire, bv_params, steps_per_day, horizon_steps, transit_amont)
    oof_lgbm = load_or_compute_oof_lgbm(
        X_train, y_train, horizon_steps, MULT_POIDS, timestep, weights_dir, n_splits_eff,
        n_trials=n_trials_lgbm, force=force_lgbm,
    )
    lgbm_result = load_or_compute_lgbm_final(
        X_train, y_train, horizon_steps, MULT_POIDS, timestep, weights_dir, n_trials=n_trials_final, force=force_lgbm,
    )

    df_train_seq = shift_amont_columns(df_train, transit_amont, horizon_steps)
    df_test_seq = shift_amont_columns(df_test, transit_amont, horizon_steps)

    seq_cols, seq_len = get_seq_cols(transit_amont, df_train_seq, horizon_steps)
    X_seq_train, y_seq_train, lstm_idx_train, t_last_train = build_sequences(df_train_seq, seq_len, horizon_steps, seq_cols)
    X_seq_test, y_test, lstm_idx_test, t_last_test = build_sequences(df_test_seq, seq_len, horizon_steps, seq_cols)

    n_features = X_seq_train.shape[2]
    bilstm = BiLSTMHydro(n_features=n_features, horizon=horizon_steps)
    oof_lstm = load_or_compute_oof_lstm(
        bilstm, X_seq_train, y_seq_train, horizon_steps, weights_dir, n_splits_eff, epochs,
        dates=df_train_seq.index[lstm_idx_train], force=force_lstm,
    )

    meteo_feature_cols = meteo_cols(df_train)
    oof_lgbm_multi = slice_lgbm_multistep(oof_lgbm, t_last_train, horizon_steps)
    stacking_result = fit_stacking(
        df_train, oof_lgbm_multi, oof_lstm, t_last_train, y_seq_train, horizon_steps,
        meteo_feature_cols, weights_dir, meta_type=meta_type,
    )

    df_full_ctx = pd.concat([df_train, df_test])
    pred_lgbm_all = predict_lgbm_full(
        lgbm_result, df_full_ctx, exutoire, bv_params, steps_per_day, horizon_steps, transit_amont,
    )

    pred_lstm_test = bilstm.predict(X_seq_test)
    yt_v, pl_v, pt_v, stk_v, df_sub, q_now_v, pred_lgbm_multi_test, pred_lstm_test, pred_stacking_multi, valid = predict_test_set(
        pred_lstm_test, y_test, lstm_idx_test, t_last_test, len(df_train),
        pred_lgbm_all, df_full_ctx, stacking_result["meta"], stacking_result["meta_scaler"],
        horizon_steps, meteo_feature_cols, df_test,
    )

    results = build_results(
        dossier, horizon, meta_type, yt_v, pl_v, pt_v, stk_v, q_now_v, df_sub,
        y_test, pred_lgbm_multi_test, pred_lstm_test, pred_stacking_multi,
        stacking_result["meta"], horizon_steps, meteo_feature_cols,
    )

    # Diagnostic honnête fit/val/test (jamais utilisé pour une sélection auto ici,
    # meta_type est figé par l'appelant) -- permet de comparer deux entraînements.
    # Fenêtre d'évaluation : split_train_test est POSITIONNEL (20% de fin), donc
    # deux entraînements sur des CSV de longueurs différentes ne mesurent pas la
    # même période. Sans cette trace, comparer deux results.json n'a pas de sens.
    results["seed"] = seed
    results["evaluation_window"] = {
        "data_rows": len(df),
        "n_train": len(df_train),
        "n_test": len(df_test),
        "test_start": str(df_test.index.min()),
        "test_end": str(df_test.index.max()),
    }

    results["training_curves"] = {
        "lgbm": lgbm_result.get("training_curve"),
        "meta": stacking_result.get("training_curve"),
    }
    results["meta_alpha"] = getattr(stacking_result["meta"], "alpha", None)

    meta_config = {
        "centrale": dossier,
        "horizon": horizon,
        "horizon_steps": horizon_steps,
        "timestep": timestep,
        "meta_type": meta_type,
        "seq_cols": seq_cols,
        "seq_len": seq_len,
        "n_splits": n_splits_eff,
        "q90_train": stacking_result["q90_train"],
        "meteo_feature_cols": meteo_feature_cols,
        "mlflow_run_id": None,
    }

    write_artifacts(results, meta_config, yt_v, pl_v, pt_v, stk_v, df_sub, dossier, horizon, weights_dir, outputs_dir)

    generate_training_plots(
        results, lgbm_result, df_full_ctx, exutoire, bv_params, steps_per_day, horizon_steps, transit_amont,
        bilstm, X_seq_test, y_test, lstm_idx_test, df_test, seq_len, df_sub,
        yt_v, pl_v, pt_v, stk_v, valid, y_test[valid], pred_stacking_multi[valid], lstm_idx_test[valid],
        dossier, horizon, weights_dir, outputs_dir,
    )

    # Doit rester après write_artifacts : _eval_context contient des ndarrays/
    # DataFrames non sérialisables JSON, write_artifacts json.dump(results) plus
    # haut planterait si ce bloc passait avant.
    results["_eval_context"] = {
        "df_full_ctx": df_full_ctx,
        "df_test": df_test,
        "X_seq_test": X_seq_test,
        "y_test": y_test,
        "lstm_idx_test": lstm_idx_test,
        "t_last_test": t_last_test,
        "n_train": len(df_train),
        "meteo_feature_cols": meteo_feature_cols,
        "bv_params": bv_params,
        "transit_amont": transit_amont,
        "exutoire": exutoire,
        "steps_per_day": steps_per_day,
        "horizon_steps": horizon_steps,
    }

    return results
