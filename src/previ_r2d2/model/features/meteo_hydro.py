"""Features météo/hydrologie -- port fidèle de la partie météo/hydrologie de
`feature()` (Previ_v2 lightgbm_model.py:489-708), boucle par station météo.
Pas de persistance : fonctions de calcul pur, appelées à la volée par
l'entraînement/la prédiction (sous-projets ultérieurs)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from previ_r2d2.preprocessing.bv.transit import haversine_km
from previ_r2d2.model.features.et0 import compute_et0
from previ_r2d2.model.features.snow import partition_precipitation, snow_melt

APPORT_NET_SEUIL = 0.1
VITESSE_ECOULEMENT_KMH = 3
ALTITUDE_HAUTE_SEUIL_M = 1300
GEL_SEUIL = 0.2
RAIN_SHOCK_OFFSET = 0.1
CHOC_HYDRAULIQUE_OFFSET = 0.1
RATIO_EPS = 1e-6
SURPRESSION_OFFSET = 0.1


def station_count(df: pd.DataFrame) -> int:
    suffixes = [c.split("_S")[-1] for c in df.columns if "_S" in c]
    return max((int(s) for s in suffixes), default=0)


def first_valid(df: pd.DataFrame, column: str) -> float:
    """Première valeur non-NaN de `column` ; lève si la colonne est entièrement NaN."""
    series = df[column].dropna()
    if series.empty:
        raise ValueError(f"{column} entièrement NaN -- station absente des données")
    return series.iloc[0]


def windows_for_distance(distance_km: float) -> list[int]:
    if distance_km <= 50:
        return [1, 3, 6, 12, 24, 36, 48, 72]
    if distance_km <= 100:
        return [6, 12, 24, 36, 48, 72]
    if distance_km <= 1000:
        return [12, 24, 36, 48, 72, 168]
    return [24, 36, 48, 72, 168]


def compute_meteo_hydro_features(
    df: pd.DataFrame,
    exutoire: dict,
    bv_params: dict,
    steps_per_day: int,
    weather_lags: list[int] | None = None,
) -> pd.DataFrame:
    """Calcule les features météo/hydrologie par station, boucle station par station."""
    spd = steps_per_day
    altitude_bv = bv_params["altitude_bv"]
    surface_km2 = bv_params["surface_km2"]
    k_base = bv_params["k_base"]
    exposition = bv_params["exposition"]
    kc_unit = bv_params["kc_unit"]

    df = df.copy()
    nb_stations = station_count(df)
    all_new_frames = []
    all_debits_theoriques = []

    for station_idx in range(1, nb_stations + 1):
        s = f"_S{station_idx}"
        new_cols: dict[str, pd.Series] = {}
        d3 = max(1, spd // 8)  # ≈3h en pas
        d6 = max(1, spd // 4)  # ≈6h en pas

        lat = first_valid(df, f"latitude{s}")
        lon = first_valid(df, f"longitude{s}")
        distance = haversine_km(exutoire["lat"], exutoire["lon"], lat, lon)
        transfert_h = max(1, int(distance / VITESSE_ECOULEMENT_KMH))
        windows = windows_for_distance(distance)

        etp = compute_et0(df, s, kc_unit)
        pluie_liquide, neige, pluie_sol, precip_inc = partition_precipitation(df, s, altitude_bv)
        fonte_series, stock_neige_list, t_moyen = snow_melt(
            df, s, neige, pluie_sol, altitude_bv, k_base, exposition
        )

        apport_net_raw = (pluie_sol + fonte_series - etp).clip(lower=0)
        apport_net_s = apport_net_raw.where(apport_net_raw > APPORT_NET_SEUIL, 0)
        new_cols[f"Pluie_apres_infiltration{s}"] = pluie_sol
        new_cols[f"Neige{s}"] = neige
        new_cols[f"Apport_Liquide_Net{s}"] = apport_net_s

        t_lisse_48h = t_moyen.rolling(2 * spd, min_periods=1).mean()
        t_lisse_24h = t_moyen.rolling(spd, min_periods=1).mean()
        new_cols[f"T_moyen_lisse_48h{s}"] = t_lisse_48h
        new_cols[f"T_moyen_lisse_24h{s}"] = t_lisse_24h
        new_cols[f"T_moyen_diff6_48h{s}"] = t_lisse_48h.diff(d6)
        new_cols[f"T_moyen_diff6_24h{s}"] = t_lisse_24h.diff(d6)

        if altitude_bv > ALTITUDE_HAUTE_SEUIL_M:
            stock_neige_series = pd.Series(stock_neige_list, index=df.index)
            facteur_gel_s = np.clip((t_lisse_24h + 3) / 2, 0.05, 1.0) ** 3
            apport_net_s = apport_net_s * facteur_gel_s
            new_cols[f"Log_Stock_Neige{s}"] = np.log1p(stock_neige_list)
            new_cols[f"Facteur_Gel{s}"] = facteur_gel_s
            new_cols[f"Apport_Liquide_Net{s}"] = apport_net_s
            new_cols[f"Vitesse_Refroidissement_6h{s}"] = t_moyen.diff(d6).clip(upper=0)
            new_cols[f"Interaction_fonte_temperature{s}"] = stock_neige_series * t_moyen
            new_cols[f"Pluie_sur_neige{s}"] = pluie_sol * stock_neige_series
            new_cols[f"Vitesse_Rechauffement_6h{s}"] = t_moyen.diff(d6).clip(lower=0)
            fonte_cumul_24h = fonte_series.rolling(spd, min_periods=1).sum()
            new_cols[f"Fonte_Cumul_24h{s}"] = fonte_cumul_24h
            new_cols[f"Fonte_Cumul_72h{s}"] = fonte_series.rolling(3 * spd, min_periods=1).sum()
            new_cols[f"Fonte_Cumul_168h{s}"] = fonte_series.rolling(7 * spd, min_periods=1).sum()
            new_cols[f"Fonte_Gradient_24h{s}"] = fonte_cumul_24h.diff(spd)
            if station_idx == 1:
                new_cols["Gel"] = (facteur_gel_s < GEL_SEUIL).astype(int)

        deficit_horaire = (etp - pluie_sol).clip(lower=0)
        new_cols[f"Indice_Précipitations_Antécédentes{s}"] = pluie_sol.ewm(span=15 * spd).mean()
        new_cols[f"Deficit_7j{s}"] = deficit_horaire.rolling(7 * spd, min_periods=1).sum()
        new_cols[f"Deficit_30j{s}"] = deficit_horaire.rolling(30 * spd, min_periods=1).sum()

        # Double underscore avant le suffixe (ex. Intensite_Pluie_1h__S1) fidèle au nom Previ_v2, pas une erreur.
        new_cols[f"Intensite_Pluie_1h_{s}"] = precip_inc
        new_cols[f"Intensite_Pluie_3h_{s}"] = precip_inc.rolling(d3).sum()
        new_cols[f"Rain_Shock_{s}"] = precip_inc / (precip_inc.rolling(spd).mean() + RAIN_SHOCK_OFFSET)

        if weather_lags:
            for lag in weather_lags:
                new_cols[f"precipitation_lag_{lag}h{s}"] = precip_inc.shift(lag)
                new_cols[f"temperature_lag_{lag}h{s}"] = df[f"temperature{s}"].shift(lag)
                new_cols[f"precipitation_cumul_{lag}h{s}"] = precip_inc.rolling(lag).sum()

        apport_shifted = apport_net_s.shift(transfert_h)
        debit_theorique = (apport_shifted * surface_km2) / 3.6
        gradient_3h = debit_theorique.diff(d3)
        new_cols[f"Apport_Liquide_Net_shifted{s}"] = apport_shifted
        new_cols[f"Debit_Theorique_Physique{s}"] = debit_theorique
        new_cols[f"Gradient_Apport_1h_{s}"] = debit_theorique.diff(1)
        new_cols[f"Gradient_Apport_3h_{s}"] = gradient_3h
        new_cols[f"Acceleration_Apport_6h_{s}"] = gradient_3h.diff(d3)
        new_cols[f"Acceleration_Apport{s}"] = apport_shifted.diff(d3)
        new_cols[f"Choc_Hydraulique_{s}"] = apport_shifted.diff(d6) / (apport_shifted.shift(d6) + CHOC_HYDRAULIQUE_OFFSET)

        apport_lisse_12h = apport_shifted.rolling(max(1, spd // 2), min_periods=1).mean()
        debit_m3s_24h_s = None
        for rol in windows:
            rol_steps = max(1, rol * spd // 24)
            apport_lisse = apport_shifted.rolling(rol_steps, min_periods=1).mean()
            debit_m3s_rol = (apport_lisse * surface_km2) / 3.6
            new_cols[f"Apport_Liquide_Sum_{rol}{s}"] = apport_net_s.rolling(rol_steps).sum()
            new_cols[f"Apport_Lisse_shift_{rol}h{s}"] = apport_lisse
            new_cols[f"Debit_m3s_{rol}h{s}"] = debit_m3s_rol
            new_cols[f"Apport_Liquide_Net_passé_{rol}_shift_{s}"] = apport_shifted.shift(rol_steps)
            if rol == 24:
                debit_m3s_24h_s = debit_m3s_rol

        reserve_court_terme = apport_shifted.ewm(span=3 * spd).mean()
        reserve_nappe = apport_shifted.ewm(span=30 * spd).mean()
        new_cols[f"Reserve_Flash_{s}"] = apport_shifted.ewm(span=max(1, spd // 2)).mean()
        new_cols[f"Reserve_Court_Terme{s}"] = reserve_court_terme
        new_cols[f"Reserve_Moyen_Terme{s}"] = apport_shifted.ewm(span=7 * spd).mean()
        new_cols[f"Reserve_Nappe{s}"] = reserve_nappe
        new_cols[f"Ratio_Rapide_Base{s}"] = reserve_court_terme / (reserve_nappe + RATIO_EPS)

        moyenne_30j = apport_net_s.rolling(30 * spd, min_periods=1).mean()
        saturation_idx = moyenne_30j / (moyenne_30j.expanding().mean() + RATIO_EPS)
        new_cols[f"Saturation_Index{s}"] = saturation_idx
        new_cols[f"Gradient_saturation_Index_168h{s}"] = saturation_idx.diff(7 * spd)
        new_cols[f"Gradient_saturation_Index_48h{s}"] = saturation_idx.diff(2 * spd)
        new_cols[f"Gradient_saturation_Index_24{s}"] = saturation_idx.diff(spd)
        new_cols[f"Gradient_saturation_Index_6h{s}"] = saturation_idx.diff(d6)

        reactivite_12h = apport_lisse_12h / (new_cols[f"Apport_Lisse_shift_{windows[-1]}h{s}"] + RATIO_EPS)
        debit_ref = debit_m3s_24h_s if debit_m3s_24h_s is not None else new_cols[f"Debit_m3s_{windows[0]}h{s}"]
        debit_mean_6h = debit_ref.rolling(d6, min_periods=1).mean()
        new_cols[f"Reactivite_12h_{s}"] = reactivite_12h
        new_cols[f"Debit_Mean_6h{s}"] = debit_mean_6h
        new_cols[f"Debit_Mean_24h{s}"] = debit_ref.rolling(spd, min_periods=1).mean()
        new_cols[f"Debit_Mean_7j{s}"] = debit_ref.rolling(7 * spd, min_periods=spd).mean()
        new_cols[f"Debit_Mean_30j{s}"] = debit_ref.rolling(30 * spd, min_periods=spd).mean()
        new_cols[f"Surpression_Relative_{s}"] = debit_theorique / (debit_mean_6h + SURPRESSION_OFFSET)

        new_cols[f"Ratio_Fonte{s}"] = fonte_series / (apport_net_s + RATIO_EPS)

        all_debits_theoriques.append(debit_theorique)

        all_new_frames.append(pd.DataFrame(new_cols, index=df.index))

    df = pd.concat([df] + all_new_frames, axis=1)

    df["Apport_Bassin_Moyen"] = pd.concat(all_debits_theoriques, axis=1).mean(axis=1)
    day_of_year = df.index.dayofyear
    df["saison_cos"] = np.cos(2 * np.pi * (day_of_year - 172) / 365)
    df["saison_sin"] = np.sin(2 * np.pi * (day_of_year - 172) / 365)

    return df
