"""Métriques pour l'orchestration entraînement -- port fidèle de
kge_per_step/kge_by_regime/kge_by_season/log_interpretability
(train_meta.py:127-318, Previ_v2). Réutilise kge_components (sous-projet
Stacking) plutôt que de dupliquer _kge_components (formule identique,
seule différence : un masque NaN interne redondant puisque tous les sites
d'appel filtrent déjà les NaN avant d'appeler la fonction)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from projet_hydro.model.architectures.stacking import kge_components

logger = logging.getLogger(__name__)


def kge_per_step(y_true_log: np.ndarray, y_pred_log: np.ndarray, horizon_steps: int) -> list[dict]:
    """KGE en log1p + RMSE en m3/s pour chaque pas t+1..t+H."""
    results = []
    for h in range(horizon_steps):
        yt = y_true_log[:, h]
        yp = y_pred_log[:, h]
        mask = ~np.isnan(yt) & ~np.isnan(yp)
        if mask.sum() < 10:
            results.append({"kge": np.nan, "r": np.nan, "alpha": np.nan, "beta": np.nan, "rmse_m3s": np.nan})
            continue
        d = kge_components(yt[mask], yp[mask])
        d["rmse_m3s"] = float(np.sqrt(np.mean((np.expm1(yp[mask]) - np.expm1(yt[mask])) ** 2)))
        results.append(d)
    return results


def kge_by_regime(y_true_m3s: np.ndarray, preds: dict, q33: float, q90: float) -> dict:
    """KGE + beta par régime de débit (etiage/normal/crue)."""
    regimes = {
        "etiage": y_true_m3s <= q33,
        "normal": (y_true_m3s > q33) & (y_true_m3s <= q90),
        "crue": y_true_m3s > q90,
    }
    results = {}
    for name, yp in preds.items():
        results[name] = {}
        for reg, mask in regimes.items():
            m = mask & ~np.isnan(yp) & ~np.isnan(y_true_m3s)
            if m.sum() < 10:
                results[name][reg] = {"kge": np.nan, "beta": np.nan, "n": int(m.sum())}
                continue
            d = kge_components(y_true_m3s[m], yp[m])
            results[name][reg] = {"kge": d["kge"], "beta": d["beta"], "n": int(m.sum())}
    return results


def kge_by_season(y_true_m3s: np.ndarray, preds: dict, dates: pd.DatetimeIndex) -> dict:
    """KGE + beta par saison calendaire (hiver DJF, printemps MAM, ete JJA, automne SON)."""
    month = dates.month
    seasons = {
        "hiver": np.isin(month, [12, 1, 2]),
        "printemps": np.isin(month, [3, 4, 5]),
        "ete": np.isin(month, [6, 7, 8]),
        "automne": np.isin(month, [9, 10, 11]),
    }
    results = {}
    for name, yp in preds.items():
        results[name] = {}
        for saison, mask in seasons.items():
            m = mask & ~np.isnan(yp) & ~np.isnan(y_true_m3s)
            if m.sum() < 10:
                results[name][saison] = {"kge": np.nan, "beta": np.nan, "n": int(m.sum())}
                continue
            d = kge_components(y_true_m3s[m], yp[m])
            results[name][saison] = {"kge": d["kge"], "beta": d["beta"], "n": int(m.sum())}
    return results


def log_interpretability(
    meta,
    y_true_m3s: np.ndarray,
    pl_v: np.ndarray,
    pt_v: np.ndarray,
    stk_v: np.ndarray,
    q_now_m3s: np.ndarray,
    horizon_steps: int,
    meteo_feature_cols: list[str],
) -> dict:
    """Analyse d'interprétabilité : coefficients Ridge, skill vs persistance, distribution des erreurs, qualité en crue."""
    report = {}

    if hasattr(meta, "coef_"):
        coef = meta.coef_
        if coef.ndim == 1:
            coef = coef[np.newaxis, :]
        H = horizon_steps
        n_meta = len(meteo_feature_cols)
        groups = {
            "lgbm": (0, H),
            "lstm": (H, 2 * H),
            "ancre": (2 * H, 2 * H + 4),
            "contexte": (2 * H + 4, 2 * H + 6),
            "meteo": (2 * H + 6, 2 * H + 6 + n_meta),
        }
        logger.info("=== Coefficients Ridge (moyenne |coef| par groupe de features) ===")
        coef_summary = {}
        for grp, (s, e) in groups.items():
            if s >= coef.shape[1] or e > coef.shape[1]:
                continue
            c = coef[:, s:e]
            mean_abs = float(np.mean(np.abs(c)))
            max_abs = float(np.max(np.abs(c)))
            sign = "+" if float(np.mean(c)) >= 0 else "-"
            logger.info("%-12s %11.4f %11.4f %15s", grp, mean_abs, max_abs, sign)
            coef_summary[grp] = {"mean_abs": round(mean_abs, 4), "max_abs": round(max_abs, 4), "sign": sign}
        w_lgbm = coef_summary.get("lgbm", {}).get("mean_abs", 0)
        w_lstm = coef_summary.get("lstm", {}).get("mean_abs", 0)
        total = w_lgbm + w_lstm + 1e-9
        logger.info("Confiance relative : LGB=%.1f%% BiLSTM=%.1f%%", w_lgbm / total * 100, w_lstm / total * 100)
        report["ridge_coef"] = coef_summary

    mask_p = ~np.isnan(q_now_m3s) & ~np.isnan(y_true_m3s)
    if mask_p.sum() > 10:
        yt = y_true_m3s[mask_p]
        qp = q_now_m3s[mask_p]
        mse_persist = float(np.mean((yt - qp) ** 2))
        kge_persist = kge_components(yt, qp)["kge"]
        logger.info("=== Skill score vs persistance naïve (q_t+H = q_now) ===")
        logger.info("Persistance : KGE=%.3f | MSE=%.4f (m3/s)2", kge_persist, mse_persist)
        skill_scores = {"persistance": {"kge": round(kge_persist, 4), "mse": round(mse_persist, 4)}}
        for name, yp in [("LightGBM seul", pl_v), ("BiLSTM seul", pt_v), ("Stacking", stk_v)]:
            m = mask_p & ~np.isnan(yp)
            if m.sum() < 10:
                continue
            mse_mod = float(np.mean((yt[m[mask_p]] - yp[m]) ** 2))
            skill = float(1 - mse_mod / (mse_persist + 1e-9))
            logger.info("%-18s %8.4f %7.3f", name, mse_mod, skill)
            skill_scores[name] = {"mse": round(mse_mod, 4), "skill_vs_persistence": round(skill, 4)}
        report["skill_vs_persistence"] = skill_scores

    logger.info("=== Distribution des erreurs absolues (m3/s) ===")
    error_dist = {}
    for name, yp in [("LightGBM seul", pl_v), ("BiLSTM seul", pt_v), ("Stacking", stk_v)]:
        m = ~np.isnan(yp) & ~np.isnan(y_true_m3s)
        if m.sum() < 10:
            continue
        ae = np.abs(y_true_m3s[m] - yp[m])
        p25, p50, p75, p90, p99 = [float(np.percentile(ae, p)) for p in [25, 50, 75, 90, 99]]
        logger.info("%-18s %6.2f %6.2f %6.2f %6.2f %6.2f", name, p25, p50, p75, p90, p99)
        error_dist[name] = {
            "p25": round(p25, 3), "p50": round(p50, 3), "p75": round(p75, 3), "p90": round(p90, 3), "p99": round(p99, 3),
        }
    report["error_distribution"] = error_dist

    q90_val = float(np.nanpercentile(y_true_m3s, 90))
    crue_mask = (y_true_m3s > q90_val) & ~np.isnan(y_true_m3s)
    if crue_mask.sum() > 10:
        logger.info("=== Qualité sur les crues (Q > Q90=%.1f m3/s, n=%d) ===", q90_val, crue_mask.sum())
        crue_quality = {}
        for name, yp in [("LightGBM seul", pl_v), ("BiLSTM seul", pt_v), ("Stacking", stk_v)]:
            m = crue_mask & ~np.isnan(yp)
            if m.sum() < 10:
                continue
            yt_c = y_true_m3s[m]
            yp_c = yp[m]
            kge_c = kge_components(yt_c, yp_c)["kge"]
            biais = float((np.mean(yp_c) - np.mean(yt_c)) / np.mean(yt_c) * 100)
            surpct = float(np.mean(yp_c > yt_c) * 100)
            souspct = float(np.mean(yp_c < yt_c) * 100)
            logger.info("%-18s %6.3f %10.1f%% %10.1f%% %11.1f%%", name, kge_c, biais, surpct, souspct)
            crue_quality[name] = {
                "kge": round(kge_c, 4), "biais_pct": round(biais, 2),
                "sur_estimation_pct": round(surpct, 2), "sous_estimation_pct": round(souspct, 2),
            }
        report["crue_quality"] = crue_quality

    return report
