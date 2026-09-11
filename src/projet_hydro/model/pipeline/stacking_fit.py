"""Fit du meta-learner Stacking -- port fidèle de la séquence de
train_meta.py:693-719 (Previ_v2), qui contourne StackingHydroModel.fit()
(confirmée morte en prod, sous-projet Stacking). Pas de cache : refit
toujours frais (fitter Ridge/petit LGBM sur des OOF déjà calculées est
rapide, contrairement à fit_oof LightGBM/BiLSTM -- cf. pièce C).

Le fit s'appuie sur la découpe chronologique fit/val/test partagée
(`pipeline/split.py`) : le scaler et le modèle ne voient que la tranche fit,
`val` sert à calibrer (alpha Ridge / early stopping LGBM par pas) et `test`
n'est jamais vu avant l'évaluation finale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from projet_hydro.model.architectures.stacking import build_meta_features, kge_components
from projet_hydro.model.pipeline.split import train_val_test_indices

logger = logging.getLogger(__name__)

DEFAULT_RIDGE_META_PARAMS = {"alpha": 1.0}
DEFAULT_LGBM_META_PARAMS = {
    "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "n_jobs": -1, "verbose": -1,
}
META_EARLY_STOPPING_ROUNDS = 50


class PerStepLGBMMeta:
    """H LGBMRegressor indépendants, un par pas d'horizon, chacun avec son propre
    early stopping.

    Remplace `MultiOutputRegressor(LGBMRegressor)`, qui transmet le même `eval_set`
    (toutes les colonnes de `meta_y`) à chacun de ses H sous-modèles alors que
    chacun n'en attend qu'une seule -- incompatible tel quel, donc jamais câblé :
    le meta-learner LGBM fittait sans aucun garde-fou contre le sur-apprentissage.
    """

    def __init__(self, **params):
        self.params = params or dict(DEFAULT_LGBM_META_PARAMS)
        self.models_: list[lgb.LGBMRegressor] = []

    def fit(self, X_fit, y_fit, X_val=None, y_val=None):
        self.models_ = []
        for k in range(y_fit.shape[1]):
            model = lgb.LGBMRegressor(**self.params)
            if X_val is not None and len(X_val) > 0:
                model.fit(
                    X_fit, y_fit[:, k],
                    eval_set=[(X_fit, y_fit[:, k]), (X_val, y_val[:, k])],
                    callbacks=[lgb.early_stopping(META_EARLY_STOPPING_ROUNDS, verbose=False)],
                )
            else:
                model.fit(X_fit, y_fit[:, k])
            self.models_.append(model)
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models_])


def _kge_m3s(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """KGE en m3/s (expm1) sur des cibles log1p -- NaN si la tranche est dégénérée."""
    yt = np.expm1(np.asarray(y_true_log, dtype=float)).ravel()
    yp = np.expm1(np.asarray(y_pred_log, dtype=float)).clip(0).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 2:
        return float("nan")
    yt_m, yp_m = yt[mask], yp[mask]
    # Seuil RELATIF, pas `== 0` : expm1 sur une cible constante laisse un std
    # résiduel d'arrondi (~1e-15) qui passait le test d'égalité stricte et
    # produisait un KGE fini de l'ordre de -1e6 -- assez pour piloter une
    # calibration et fuiter dans results.json.
    if np.std(yt_m) <= 1e-9 * max(1.0, abs(float(np.mean(yt_m)))):
        return float("nan")
    kge = kge_components(yt_m, yp_m)["kge"]
    return kge if np.isfinite(kge) else float("nan")


def fit_stacking(
    df_full: pd.DataFrame,
    oof_lgbm_multi: np.ndarray,
    oof_lstm_vals: np.ndarray,
    t_last: np.ndarray,
    y_train: np.ndarray,
    horizon: int,
    meteo_feature_cols: list[str],
    output_dir: Path,
    meta_type: str = "ridge",
) -> dict:
    """Construit les features meta, fit le meta-learner (Ridge/LGBM), sauvegarde meta.pkl/meta_scaler.pkl."""
    meta_X, meta_y, valid = build_meta_features(
        df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon, meteo_feature_cols
    )
    q90_train = float(np.percentile(y_train[~np.isnan(y_train)], 90))

    fit_s, val_s, test_s = train_val_test_indices(len(meta_X))
    has_val = val_s.stop > val_s.start

    if meta_type == "lgbm":
        # PerStepLGBMMeta a BESOIN d'une tranche val (early stopping par pas) :
        # ce chemin garde donc la découpe. Non mesuré par l'ablation du
        # 2026-09-09, qui ne portait que sur ridge (le défaut).
        meta_scaler = StandardScaler().fit(meta_X[fit_s])
        meta_X_s = meta_scaler.transform(meta_X)
        meta = PerStepLGBMMeta(**DEFAULT_LGBM_META_PARAMS)
        meta.fit(meta_X_s[fit_s], meta_y[fit_s],
                 meta_X_s[val_s] if has_val else None, meta_y[val_s] if has_val else None)
        diagnostic = meta
    else:
        # Ridge : le modèle DÉPLOYÉ est fitté sur 100% avec l'alpha historique.
        # Calibrer alpha sur une tranche val contiguë unique a été mesuré NUISIBLE
        # (ablation à modèles de base identiques : -0.023 KGE sur apas_G1_G4,
        # alpha retenu instable de 0.1 à 30.0 d'un run à l'autre). Cf. skill.
        meta_scaler = StandardScaler().fit(meta_X)
        meta_X_s = meta_scaler.transform(meta_X)
        meta = Ridge(**DEFAULT_RIDGE_META_PARAMS).fit(meta_X_s, meta_y)
        # Modèle JETABLE, fitté sur la seule tranche fit, uniquement pour que
        # kge_val/kge_test restent des mesures hors échantillon. Jamais sauvegardé.
        diagnostic = Ridge(**DEFAULT_RIDGE_META_PARAMS).fit(meta_X_s[fit_s], meta_y[fit_s])

    kge_fit = _kge_m3s(meta_y[fit_s], diagnostic.predict(meta_X_s[fit_s]))
    kge_val = _kge_m3s(meta_y[val_s], diagnostic.predict(meta_X_s[val_s])) if has_val else float("nan")
    kge_test = _kge_m3s(meta_y[test_s], diagnostic.predict(meta_X_s[test_s])) if test_s.stop > test_s.start else float("nan")

    logger.info(
        "Meta-learner [%s] -- fit deploye sur %d samples, %d features, H=%d pas "
        "| diagnostic hors echantillon : kge_fit=%.4f kge_val=%.4f kge_test=%.4f",
        meta_type, len(meta_X), meta_X.shape[1], horizon, kge_fit, kge_val, kge_test,
    )

    joblib.dump(meta, output_dir / "meta.pkl")
    joblib.dump(meta_scaler, output_dir / "meta_scaler.pkl")
    logger.info("Meta-learner [%s] entraîné et sauvegardé -> %s", meta_type, output_dir / "meta.pkl")

    return {
        "meta": meta,
        "meta_scaler": meta_scaler,
        "meta_X": meta_X,
        "meta_y": meta_y,
        "valid": valid,
        "q90_train": q90_train,
        "training_curve": {"kge_fit": kge_fit, "kge_val": kge_val, "kge_test": kge_test},
    }
