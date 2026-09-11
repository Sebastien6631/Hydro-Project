"""Entraînement LightGBM (meta) -- port fidèle de objective_reg/fit_oof
(Previ_v2 lightgbm_model.py:829-1239), plus select_top_features (bloc
partagé dédupliqué de fit_oof/fit_final). Pas de persistance disque : X_valid/
y_valid/pmax (objective_reg) sont vestigiaux côté Previ_v2 (jamais lus dans
le corps), retirés de la signature. Le log "crue exceptionnelle" (vestigial,
ne pilotait rien) est retiré ; des logs de progression utiles sont ajoutés
à la place (demande utilisateur)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit

from projet_hydro.model.architectures.lightgbm.metrics import debit_weights, debit_quantiles, kge_loss
from projet_hydro.model.pipeline.split import train_val_test_indices

logger = logging.getLogger(__name__)

OOF_LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "verbose": -1,
    "objective": "regression",
}


def select_top_features(X: pd.DataFrame, y_log: np.ndarray) -> list[str]:
    """Sélectionne les top features (importances LightGBM + forced_always + slow[:3])."""
    quick = lgb.LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y_log)
    importances = pd.Series(quick.feature_importances_, index=X.columns)
    top50 = importances.sort_values(ascending=False).head(50).index.tolist()

    forced_always = ["debit_lag_24h", "debit_lag_36h", "max_debit_vu"]
    forced = [c for c in forced_always if c in X.columns and c not in top50]
    slow_pats = [r"^Reserve_Nappe", r"^Saturation_Index", r"^Reserve_Moyen_Terme"]
    slow = [f for f in top50 if any(re.match(p, f) for p in slow_pats)]
    other = [f for f in top50 if f not in slow]
    return [c for c in (other + slow[:3] + forced) if c in X.columns]


def objective_reg(
    trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, horizon: int, mult_poids: float, timestep: str
) -> float:
    """Score Optuna (KGE moyen sur TimeSeriesSplit) pour un jeu d'hyperparamètres LightGBM."""
    n_splits = 3 if timestep in ("12h", "1D") else 6
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    scores = []

    objective_choice = trial.suggest_categorical("objective", ["tweedie"])
    metric_choice = trial.suggest_categorical("metric", ["huber", "l1", "l2"])
    ff_low, ff_high = 0.10, 0.35

    param = {
        "objective": objective_choice,
        "alpha": 0.5,
        "metric": metric_choice,
        "verbosity": -1,
        "boosting": trial.suggest_categorical("boosting", ["gbdt"]),
        "n_estimators": trial.suggest_int("n_estimators", 1000, 5000),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, max(20, len(X_train) // (n_splits * 4))),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 50.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 50.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", ff_low, ff_high),
        "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", ff_low, ff_high),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "path_smooth": trial.suggest_float("path_smooth", 1, 40.0, log=True),
        "extra_trees": True,
        "n_jobs": 8,
    }
    if objective_choice == "tweedie":
        param["tweedie_variance_power"] = 1.99

    y_ref_full = np.expm1(y_train)

    for train_index, test_index in tscv.split(X_train):
        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        weights_cv = debit_weights(np.expm1(y_train_cv), mult_poids, y_ref=y_ref_full)

        gbm = lgb.LGBMRegressor(**param)
        gbm.fit(
            X_train_cv,
            y_train_cv,
            sample_weight=weights_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        preds = gbm.predict(X_val_cv)
        y_val_orig = np.expm1(y_val_cv.values)
        preds_orig = np.expm1(preds)
        scores.append(kge_loss(y_val_orig, preds_orig))

    mean_score = float(np.mean(scores))
    logger.info("objective_reg: score moyen (1-KGE)=%.4f sur %d folds", mean_score, n_splits)
    return mean_score


def fit_oof(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    mult_poids: float,
    timestep: str,
    n_splits: int = 5,
    n_trials: int = 30,
    checkpoint_path: Path | None = None,
) -> np.ndarray:
    """OOF (log-space) via TimeSeriesSplit ; NaN aux positions jamais validées.

    `checkpoint_path`, si fourni : sauvegarde une reprise par fold (oof
    partiel + top_cols/best_params déjà figés, pour rester identiques d'un
    fold à l'autre après reprise) après chaque fold réellement entraîné --
    si le processus est tué en cours de route, un appel ultérieur avec le
    même `checkpoint_path` saute les folds déjà faits au lieu de tout
    recommencer. Le fichier est supprimé automatiquement une fois tous les
    folds terminés."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y_arr = y.to_numpy()
    y_log = np.log1p(y_arr)
    y_log_series = pd.Series(y_log, index=X.index)

    checkpoint = None
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = joblib.load(checkpoint_path)
        logger.info(
            "fit_oof: reprise depuis checkpoint (%d/%d folds déjà faits)",
            len(checkpoint["completed_folds"]), n_splits,
        )

    top_cols = checkpoint["top_cols"] if checkpoint is not None else select_top_features(X, y_log)
    X_top = X[top_cols].reset_index(drop=True)

    if checkpoint is not None:
        best_params = checkpoint["best_params"]
    elif n_trials > 0:
        logger.info("fit_oof: tuning Optuna (%d trials)", n_trials)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda trial: objective_reg(trial, X_top, y_log_series, horizon, mult_poids, timestep),
            n_trials=n_trials,
        )
        best_params = study.best_params
        logger.info("fit_oof: tuning terminé, meilleur score=%.4f", study.best_value)
    else:
        best_params = dict(OOF_LGBM_PARAMS)

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    if checkpoint is not None:
        oof = checkpoint["oof"]
        completed_folds = set(checkpoint["completed_folds"])
    else:
        oof = np.full(len(X_top), np.nan)
        completed_folds = set()

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_top)):
        if fold_idx in completed_folds:
            logger.info("fit_oof: fold %d/%d déjà fait (checkpoint), sauté", fold_idx + 1, n_splits)
            continue
        sw = debit_weights(y_arr[train_idx], mult_poids)
        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            X_top.iloc[train_idx],
            y_log[train_idx],
            sample_weight=sw,
            eval_set=[(X_top.iloc[val_idx], y_log[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        preds_log = model.predict(X_top.iloc[val_idx])
        oof[val_idx] = preds_log
        fold_kge = 1.0 - kge_loss(y_arr[val_idx], np.expm1(preds_log).clip(0))
        logger.info("fit_oof: fold %d/%d KGE=%.4f", fold_idx + 1, n_splits, fold_kge)

        completed_folds.add(fold_idx)
        if checkpoint_path is not None:
            joblib.dump(
                {
                    "oof": oof,
                    "top_cols": top_cols,
                    "best_params": best_params,
                    "completed_folds": sorted(completed_folds),
                },
                checkpoint_path,
            )

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()

    return oof


def fit_final(X: pd.DataFrame, y: pd.Series, horizon: int, mult_poids: float, timestep: str, n_trials: int = 50) -> dict:
    """Entraînement final : sélectionne les top features, tune via objective_reg (mêmes hyperparamètres que fit_oof), fit sur tout X."""
    y_arr = y.to_numpy()
    _, q_start, q90, q99 = debit_quantiles(y_arr)
    y_log = np.log1p(y_arr)
    y_log_series = pd.Series(y_log, index=X.index)

    top_features = select_top_features(X, y_log)
    X_top = X[top_features].reset_index(drop=True)
    sw = debit_weights(y_arr, mult_poids)

    if n_trials > 0:
        logger.info("fit_final: tuning Optuna (%d trials)", n_trials)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda trial: objective_reg(trial, X_top, y_log_series, horizon, mult_poids, timestep),
            n_trials=n_trials,
        )
        best_params = study.best_params
        logger.info("fit_final: tuning terminé, meilleur score=%.4f", study.best_value)
    else:
        best_params = dict(OOF_LGBM_PARAMS)

    # Découpe chronologique 80/10/10 : le fit final n'avait AUCUN early stopping
    # (contrairement à fit_oof), donc aucun garde-fou contre le sur-apprentissage.
    # Céder 20% des lignes sans rien en retour aurait été une perte sèche : val
    # sert désormais à l'early stopping, test à un KGE honnête (diagnostic seul).
    fit_s, val_s, test_s = train_val_test_indices(len(X_top))
    has_val = val_s.stop > val_s.start
    sw_fit = sw[fit_s]

    model = lgb.LGBMRegressor(**best_params)
    if has_val:
        # eval_sample_weight explicite : sans lui LightGBM réutilise pour la série
        # "fit" le Dataset déjà pondéré, mais construit "val" à weight=None -- les
        # deux courbes seraient sur des échelles incohérentes, masquant justement
        # le sur-apprentissage que l'early stopping doit détecter.
        sw_val = debit_weights(y_arr[val_s], mult_poids, y_ref=y_arr[fit_s])
        model.fit(
            X_top.iloc[fit_s], y_log[fit_s], sample_weight=sw_fit,
            eval_set=[(X_top.iloc[fit_s], y_log[fit_s]), (X_top.iloc[val_s], y_log[val_s])],
            eval_sample_weight=[sw_fit, sw_val],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        model.fit(X_top.iloc[fit_s], y_log[fit_s], sample_weight=sw_fit)

    def _kge(sl):
        if sl.stop <= sl.start:
            return float("nan")
        yt, yp = np.expm1(y_log[sl]), np.expm1(model.predict(X_top.iloc[sl])).clip(0)
        return float(1.0 - kge_loss(yt, yp))  # kge_loss = 1 - KGE (cf. objective_reg)

    training_curve = {"kge_fit": _kge(fit_s), "kge_val": _kge(val_s), "kge_test": _kge(test_s)}
    logger.info(
        "fit_final: entraînement final terminé (%d features, fit=%d/val=%d/test=%d) "
        "| kge_fit=%.4f kge_val=%.4f kge_test=%.4f",
        len(top_features), fit_s.stop - fit_s.start, val_s.stop - val_s.start,
        test_s.stop - test_s.start, training_curve["kge_fit"], training_curve["kge_val"],
        training_curve["kge_test"],
    )

    return {
        "model": model,
        "top_features": top_features,
        "q_start": q_start,
        "q90": q90,
        "q99": q99,
        "n_train": len(X),
        "training_curve": training_curve,
    }
