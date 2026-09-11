"""Orchestrateur de prédiction -- port de predict_one (predict_future_meta.py,
Previ_v2), réduit au périmètre acté : débit station + conversion turbine,
pas de puissance/kW ni d'archivage previsions_db (différés)."""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
import torch

from projet_hydro.common import config
from projet_hydro.model.architectures.bilstm.model import BiLSTMHydro
from projet_hydro.model.architectures.bilstm.sequences import build_last_window, get_seq_cols
from projet_hydro.model.device import resolve_device
from projet_hydro.model.architectures.lightgbm.predict import predict_lgbm_future
from projet_hydro.model.architectures.stacking import build_meta_features_live, build_meteo_lag, safe_slope
from projet_hydro.model.features.amont import shift_amont_columns
from projet_hydro.model.pipeline.bv_config import (
    bv_params_from_bv_json,
    transit_amont_from_bv_json,
    transit_centrale_from_bv_json,
)
from projet_hydro.model.pipeline.hydraulic import compute_hydraulic_point
from projet_hydro.model.pipeline.orchestrator import HORIZON_CFG
from projet_hydro.model.pipeline.predict_window import find_record, load_prediction_window

logger = logging.getLogger(__name__)

SAISONS_MOIS = {
    "hiver": [12, 1, 2],
    "printemps": [3, 4, 5],
    "ete": [6, 7, 8],
    "automne": [9, 10, 11],
}


def season_for_month(month: int) -> str:
    """Saison française (hiver/printemps/ete/automne) pour un mois donné."""
    for saison, mois in SAISONS_MOIS.items():
        if month in mois:
            return saison
    raise ValueError(f"Mois invalide : {month}")


def to_display_timezone(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convertit UTC -> Europe/Paris pour l'affichage."""
    return dates.tz_localize("UTC").tz_convert("Europe/Paris").tz_localize(None)


def priorite_groupe_1(rec: dict) -> int | None:
    """Priorité du groupe le plus prioritaire (valeur minimale) du raccordement, None si aucun groupe."""
    groupes = rec.get("groupes", [])
    if not groupes:
        return None
    return min(g["priorite"] for g in groupes)


def build_prevision_points(out_df: pd.DataFrame, rec: dict) -> list[dict]:
    """Construit la liste 'points' de prevision.json directement depuis out_df (série continue observé+prédit, déjà tronquée à partir de l'heure de lancement) -- lead_hours = position dans la série."""
    points = []
    for k, row in enumerate(out_df.itertuples()):
        q_entrant = float(row.q_entrant_m3s)
        hydro = compute_hydraulic_point(q_entrant, rec, pd.Timestamp(row.datetime))
        points.append({
            "lead_hours": k,
            "target_datetime": pd.Timestamp(row.datetime).replace(microsecond=0).isoformat(),
            "debit": {
                "entrant": round(q_entrant, 3),
                "turbinable": hydro["debit_turbinable"],
                "reserve": hydro["debit_reserve"],
            },
            "puissance": hydro["puissance"],
            "pmax_dyn": hydro["pmax_dyn"],
            "chute_estimee": hydro["chute_estimee"],
            "rendement_estime": hydro["rendement_estime"],
        })
    return points


def write_prevision_json(centrales_dir, dossier, rec, now, now_ts, points, puissance_reel, meta_config):
    """Écrit centrales/{dossier}/prevision.json (horizon 8 uniquement)."""
    data = {
        "centrale": dossier,
        "generation-date": now.strftime("%Y-%m-%d-T%H:%M:%S"),
        "data-date": now_ts.strftime("%Y-%m-%d-T%H:%M:%S"),
        "is-current": True,
        "priorite": priorite_groupe_1(rec),
        "type": rec.get("type"),
        "puissance-reel": puissance_reel,
        "run": {
            "model_type": "meta",
            "model_version": meta_config.get("mlflow_run_id") or "unknown",
            "points": points,
        },
    }
    path = centrales_dir / dossier / "prevision.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_trained_models(weights_dir, horizon_steps: int, n_features: int):
    """Charge BiLSTM (poids + scalers), lgbm_final.pkl, meta.pkl, meta_scaler.pkl depuis weights_dir."""
    # map_location="cpu" d'abord (un checkpoint entraîné sur GPU doit rester
    # chargeable sur une machine sans GPU), PUIS déplacement vers le device
    # effectif -- _forward_batched suit le device du modèle.
    device = resolve_device()
    bilstm = BiLSTMHydro(n_features=n_features, horizon=horizon_steps)
    bilstm.load_state_dict(torch.load(weights_dir / "bilstm.pt", map_location="cpu"))
    bilstm.to(device)
    bilstm.eval()
    scaler_files = sorted(weights_dir.glob("bilstm_scaler_*.pkl"))
    bilstm.scalers = [joblib.load(p) for p in scaler_files]

    lgbm_result = joblib.load(weights_dir / "lgbm_final.pkl")
    meta = joblib.load(weights_dir / "meta.pkl")
    meta_scaler = joblib.load(weights_dir / "meta_scaler.pkl")
    return bilstm, lgbm_result, meta, meta_scaler


def run_prediction(
    dossier: str, horizon: int, exutoire: dict, bv_json: dict, now: pd.Timestamp,
    source: str = "live",
) -> dict:
    """Prédit les horizon prochains pas pour une centrale, convertit en débit turbine, écrit un CSV."""
    cfg = HORIZON_CFG[horizon]
    horizon_steps, timestep, steps_per_day = cfg["horizon_steps"], cfg["timestep"], cfg["steps_per_day"]

    weights_dir = config.MODELS_DIR / dossier / f"h{horizon}"
    outputs_dir = config.ROOT / "outputs" / "hybrid" / dossier / f"h{horizon}" / "predictions"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    meta_config_path = weights_dir / "meta_config.json"
    if not meta_config_path.exists():
        raise FileNotFoundError(
            f"{dossier} h{horizon} : aucun modèle promu en production ({meta_config_path} introuvable) -- "
            "premier entraînement/promotion pas encore effectué pour cette centrale/horizon."
        )
    with open(meta_config_path) as fh:
        meta_config = json.load(fh)
    seq_cols = meta_config["seq_cols"]
    q90_train = meta_config["q90_train"]
    meteo_feature_cols = meta_config["meteo_feature_cols"]

    bv_params = bv_params_from_bv_json(bv_json)
    transit_amont = transit_amont_from_bv_json(bv_json)
    rec = find_record(dossier)

    logger.info("%s h%s -- lancement à %s", dossier, horizon, now)

    df_window = load_prediction_window(dossier, horizon_steps, timestep, now, source=source)
    logger.info("%s h%s -- fenêtre chargée : %s -> %s (%d lignes)", dossier, horizon, df_window.index.min(), df_window.index.max(), len(df_window))
    df_window_seq = shift_amont_columns(df_window, transit_amont, horizon_steps)

    _, seq_len = get_seq_cols(transit_amont, df_window_seq, horizon_steps)
    n_features = len(seq_cols) + 5  # + hour_sin/cos, doy_sin/cos, is_fut

    bilstm, lgbm_result, meta, meta_scaler = load_trained_models(weights_dir, horizon_steps, n_features)

    pred_lgbm_future, future_dates = predict_lgbm_future(
        lgbm_result, df_window, exutoire, bv_params, steps_per_day, horizon_steps, transit_amont,
    )

    X_last = build_last_window(df_window_seq, seq_len, horizon_steps, seq_cols)
    pred_lstm_future = bilstm.predict(X_last)[0]

    obs_mask = df_window["debit_m3s"].notna().values
    now_pos = int(np.where(obs_mask)[0][-1])
    last_obs_debit = float(df_window["debit_m3s"].iloc[now_pos])
    q_now_log = float(np.log1p(max(0.0, last_obs_debit)))
    now_ts = pd.Timestamp(df_window.index[now_pos])
    month_now = now_ts.month
    logger.info("%s h%s -- dernière observation à %s (débit=%.3f m3/s)", dossier, horizon, now_ts, last_obs_debit)

    trend_series = df_window["debit_m3s"].rolling(72, min_periods=12).apply(safe_slope, raw=True)
    trend_at_now = trend_series.iloc[now_pos]
    trend = float(trend_at_now) if not np.isnan(trend_at_now) else 0.0

    meteo_arr = build_meteo_lag(df_window, meteo_feature_cols, horizon_steps)
    meteo_now = meteo_arr[now_pos]

    meta_X = build_meta_features_live(pred_lgbm_future, pred_lstm_future, q_now_log, month_now, q90_train, trend, meteo_now)
    meta_X_s = meta_scaler.transform(meta_X)
    pred_stacking_log = meta.predict(meta_X_s)[0]

    lgbm_m3s = np.expm1(np.nan_to_num(pred_lgbm_future, nan=0.0)).clip(0)
    lstm_m3s = np.expm1(pred_lstm_future).clip(0)
    stacking_m3s = np.expm1(pred_stacking_log).clip(0)

    facteur_debit = float(rec.get("facteur_debit", 1.0))
    # Le débit est mesuré à une station Hub'Eau en amont : le temps de transit
    # jusqu'à la turbine doit être ajouté.
    transit_centrale = transit_centrale_from_bv_json(bv_json)
    decalage_h = round(transit_centrale.get(season_for_month(month_now), 0))
    logger.info("%s h%s -- decalage_h=%dh (saison=%s)", dossier, horizon, decalage_h, season_for_month(month_now))

    step = pd.Timedelta(hours=1)
    future_dates_turbine = pd.DatetimeIndex([
        now_ts + pd.Timedelta(hours=decalage_h) + (k + 1) * step for k in range(horizon_steps)
    ])
    q_entrant_m3s = stacking_m3s * facteur_debit

    # Affichage seul (le pipeline interne reste en UTC) : cf. to_display_timezone.
    datetime_out = to_display_timezone(future_dates_turbine)

    # Historique observé concaténé aux prédictions (comme Previ_v2, merged_debit
    # + type_col "observe"/"prediction") -- même conversion turbine (facteur_debit
    # + decalage_h) que les points prédits, pour un CSV continu utilisable à
    # partir de n'importe quelle heure, pas seulement le futur.
    lookback_periods = 48
    hist_debit = df_window.loc[df_window.index <= now_ts, "debit_m3s"].dropna().tail(lookback_periods)
    hist_dates_turbine = pd.DatetimeIndex(hist_debit.index + pd.Timedelta(hours=decalage_h))
    hist_dates_display = to_display_timezone(hist_dates_turbine)

    hist_df = pd.DataFrame({
        "datetime": hist_dates_display,
        "q_lgbm_m3s": np.nan,
        "q_lstm_m3s": np.nan,
        "q_stacking_m3s": np.nan,
        "q_entrant_m3s": hist_debit.values * facteur_debit,
        "source": "observe",
    })
    pred_df = pd.DataFrame({
        "datetime": datetime_out,
        "q_lgbm_m3s": lgbm_m3s,
        "q_lstm_m3s": lstm_m3s,
        "q_stacking_m3s": stacking_m3s,
        "q_entrant_m3s": q_entrant_m3s,
        "source": "prediction",
    })
    out_df = pd.concat([hist_df, pred_df], ignore_index=True)
    csv_path = outputs_dir / f"hybrid_pred_{dossier}_{now.strftime('%Y%m%d_%Hh')}_{horizon}h.csv"
    out_df.to_csv(csv_path, index=False)
    logger.info("%s h%s -- out_df construit : %d lignes (%s -> %s), CSV écrit -> %s",
                dossier, horizon, len(out_df), out_df["datetime"].iloc[0], out_df["datetime"].iloc[-1], csv_path)

    # Le CSV garde tout l'historique (48/5 dernières périodes) pour lecture
    # humaine, mais le JSON ne doit démarrer qu'à l'heure de lancement (now) --
    # comme Previ_v2 : merged_debit[merged_debit.index >= launch_ts]. Tronquer
    # ici suffit à reproduire la correction du décalage de transit (le calcul
    # "q_now_turbine" spécial n'est plus nécessaire : la première ligne de la
    # série tronquée à partir de `now` porte déjà le décalage appliqué).
    out_df_json = out_df[out_df["datetime"] >= now].reset_index(drop=True)
    logger.info("%s h%s -- out_df tronqué à partir de now=%s : %d lignes pour le JSON",
                dossier, horizon, now, len(out_df_json))

    q_now_entrant = float(out_df_json["q_entrant_m3s"].iloc[0])
    now_row_datetime = pd.Timestamp(out_df_json["datetime"].iloc[0])
    puissance_reel_full = compute_hydraulic_point(q_now_entrant, rec, now_row_datetime)
    puissance_reel = {
        "puissance": puissance_reel_full["puissance"], "pmax_dyn": puissance_reel_full["pmax_dyn"],
        "chute_estimee": puissance_reel_full["chute_estimee"], "rendement_estime": puissance_reel_full["rendement_estime"],
    }
    logger.info("%s h%s -- puissance-reel : q_entrant=%.3f m3/s, puissance=%.1f kW",
                dossier, horizon, q_now_entrant, puissance_reel["puissance"])

    # data-date dérive de now_ts (l'horodatage de la dernière observation
    # réelle, genuinement stocké en UTC) -- passe par to_display_timezone.
    # `now` (pd.Timestamp.now(), déjà l'heure murale locale du système) ne
    # doit PAS repasser par to_display_timezone -- ce n'est pas une donnée
    # UTC, la convertir une seconde fois la décale à tort de +2h (bug
    # constaté en usage réel : generation-date affichait 18h alors que
    # l'heure de lancement réelle était 16h).
    display_now_ts = to_display_timezone(pd.DatetimeIndex([now_ts]))[0]

    points = build_prevision_points(out_df_json, rec)
    write_prevision_json(config.CENTRALES_DIR, dossier, rec, now, display_now_ts, points, puissance_reel, meta_config)
    logger.info("%s h%s -- prevision.json écrit (%d points)", dossier, horizon, len(points))

    return {
        "centrale": dossier,
        "horizon": horizon,
        "now": now_ts.isoformat(),
        "csv_path": str(csv_path),
        "q_stacking_m3s": stacking_m3s.tolist(),
        "q_entrant_m3s": q_entrant_m3s.tolist(),
    }
