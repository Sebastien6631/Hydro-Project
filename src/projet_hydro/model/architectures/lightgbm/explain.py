"""Interprétabilité LightGBM (SHAP) -- port fidèle de _explain_lgbm
(train_meta.py, Previ_v2) et de plot_shap_bar (mlflow_tracker.py,
Previ_v2), qui réutilise le même shap_values que le beeswarm."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from projet_hydro.model.architectures.lightgbm.features import build_features


def build_shap_explanation(
    lgbm_result: dict,
    df: pd.DataFrame,
    exutoire: dict,
    bv_params: dict,
    steps_per_day: int,
    horizon: int,
    transit_amont: dict,
) -> tuple[np.ndarray, pd.DataFrame] | tuple[None, None]:
    """Calcule les shap_values (TreeExplainer) sur les top_features ; (None, None) si pas de modèle."""
    model = lgbm_result["model"]
    top_features = lgbm_result["top_features"]
    if model is None or not top_features:
        return None, None
    X, _, _ = build_features(df, exutoire, bv_params, steps_per_day, horizon, transit_amont)
    avail = [f for f in top_features if f in X.columns]
    if not avail:
        return None, None
    X_top = X[avail]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_top.values)
    return shap_values, X_top


def plot_shap_beeswarm(shap_values: np.ndarray, X_top: pd.DataFrame, output_path) -> None:
    """Beeswarm SHAP -- impact et direction des top-20 features."""
    shap.summary_plot(shap_values, X_top.values, feature_names=list(X_top.columns), max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_shap_bar(shap_values: np.ndarray, X_top: pd.DataFrame, output_path) -> None:
    """Bar chart SHAP -- importance moyenne absolue classée."""
    shap.summary_plot(shap_values, X_top.values, feature_names=list(X_top.columns), max_display=20, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
