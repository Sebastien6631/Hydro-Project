"""Calcul physique du débit entrant (Qentrant) -- port de load_debit_automate_v2.py.

`compute_qentrant` renvoie un DataFrame multi-colonnes (une colonne par méthode
d'estimation disponible), pas une série unique -- la sélection de la colonne
finale ("_ouv" > "_P" par priorité) est faite par l'appelant (cron/scripts/maj-automate.py),
pas ici, pour rester un port fidèle de la logique Previ_v2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.81


def calcul_polynome_vectorise(series: pd.Series, param_segments: list[dict]) -> pd.Series:
    result = np.zeros(len(series))
    for param in param_segments:
        s_min, s_max = param["seuils"]
        mask = (series >= s_min) & (series <= s_max)
        valeurs_segment = sum(a * (series[mask] ** i) for i, a in enumerate(param["A"]))
        result[mask] = valeurs_segment
    return pd.Series(result, index=series.index)


def calcul_hauteur(df: pd.DataFrame, dico: dict, puissance: pd.Series) -> pd.Series:
    categorie = dico["categorie"]
    if categorie == "basse chute":
        return (df["sonde_amont"] - df["sonde_restitution"] + dico["constante_correction_Hn"]) / 100

    if categorie == "haute chute":
        estime_qturb = calcul_polynome_vectorise(puissance, dico["segment_Q_fct_P"])
        zb_zc = (
            dico["constante_correction_Hn"]
            if dico["type"] == "action"
            else -df["sonde_restitution"] + dico["constante_correction_Hn"]
        )
        return (
            (df["pression_conduite"] * 100000) / (G * 1000)
            + ((estime_qturb / (np.pi * (dico["diam_conduite"] / 2) ** 2)) ** 2) / (2 * G)
            + (zb_zc / 100)
        )

    raise ValueError("Catégorie de l'aménagement mal définie")


def calcul_qturb(df: pd.DataFrame, dico: dict, puissance: pd.Series, chute_nette: pd.Series) -> pd.DataFrame:
    """Jusqu'à 4 estimations candidates du débit turbiné, une par méthode
    configurée -- chaque méthode absente (clé de config manquante, ou colonne
    capteur absente pour "Q_fct_ouv") reste une série de zéros, pas une erreur."""
    zeros = pd.Series(np.zeros(len(df)), index=df.index)
    cols: dict[str, pd.Series] = {}

    cols["Q_fct_P"] = (
        calcul_polynome_vectorise(puissance, dico["segment_Q_fct_P"])
        if "segment_Q_fct_P" in dico else zeros.copy()
    )

    if "segment_Q_fct_charge_pression" in dico:
        try:
            charge = (df["pression_conduite"] * 1e5) / (G * 1000) + dico.get("sonde_aval_grille", 0) / 1000
            cols["Q_fct_charge"] = calcul_polynome_vectorise(charge, dico["segment_Q_fct_charge_pression"])
        except Exception:
            cols["Q_fct_charge"] = zeros.copy()
    else:
        cols["Q_fct_charge"] = zeros.copy()

    if "segment_Rgroupe_fct_ouv_grp" in dico:
        try:
            r = calcul_polynome_vectorise(df["ouv_groupe"], dico["segment_Rgroupe_fct_ouv_grp"]) / 100
            mask = (r > 0) & (chute_nette > 0)
            q = zeros.copy()
            q[mask] = puissance[mask] / (r[mask] * chute_nette[mask] * G)
            cols["Q_fct_P_Hn_R"] = q
        except Exception:
            cols["Q_fct_P_Hn_R"] = zeros.copy()
    else:
        cols["Q_fct_P_Hn_R"] = zeros.copy()

    if "segment_Q_fct_ouv_grp" in dico:
        try:
            type_turbine = dico["type"]
            if type_turbine == "action":
                q = zeros.copy()
                for i in range(1, 6):
                    col = f"Ouv_inject_{i}"
                    if col in df.columns:
                        q = q + calcul_polynome_vectorise(df[col], dico["segment_Q_fct_ouv_grp"])
                cols["Q_fct_ouv"] = q
            elif type_turbine in ("réaction", "reaction"):
                cols["Q_fct_ouv"] = (
                    calcul_polynome_vectorise(df["ouv_groupe"], dico["segment_Q_fct_ouv_grp"])
                    if "ouv_groupe" in df.columns else zeros.copy()
                )
            else:
                raise ValueError("Type de turbine mal défini")
        except Exception:
            cols["Q_fct_ouv"] = zeros.copy()
    else:
        cols["Q_fct_ouv"] = zeros.copy()

    return pd.concat(cols, axis=1)


def calcul_deversoir_q_serie(df: pd.DataFrame, dico: dict, i: int) -> pd.Series:
    sonde = df[f"sonde_{i}"]
    if dico["Qreserve_fixe"] == "oui":
        if dico["type"] == "fixe":
            return calcul_polynome_vectorise(sonde, dico["param_segments"])
        if dico["type"] == "mobile":
            ouverture = df[dico["capteur_ouv"]]
            elev_crete = calcul_polynome_vectorise(ouverture, dico["param_segments"])
            lame_eau = (sonde - elev_crete).clip(lower=0)
            return dico["coef_debitance"] * dico["largeur"] * np.sqrt(2 * G) * ((lame_eau * 0.001) ** 1.5)
        raise ValueError("Type de déversoir mal défini")

    if dico["Qreserve_fixe"] == "non":
        (sm, sd), (em, ed) = dico["periode_ete"]
        month, day = df.index.month, df.index.day
        is_summer = (
            ((month > sm) | ((month == sm) & (day >= sd)))
            & ((month < em) | ((month == em) & (day <= ed)))
        )
        seg_hiver = dico.get("param_segments_2", dico["param_segments_1"])
        result = pd.Series(0.0, index=df.index)
        if is_summer.any():
            result[is_summer] = calcul_polynome_vectorise(sonde[is_summer], dico["param_segments_1"])
        if (~is_summer).any():
            result[~is_summer] = calcul_polynome_vectorise(sonde[~is_summer], seg_hiver)
        return result

    raise ValueError("Erreur sur la définition de Qreserve_fixe")


def compute_qrestitue_total(df: pd.DataFrame, config: dict, consigne_regulation: dict) -> pd.Series:
    qrestitue_total = pd.Series(np.zeros(len(df)), index=df.index)
    i = 1
    while f"Config_deversoir_{i}" in config:
        df[f"sonde_{i}"] = df[f"sonde_{i}"] - float(consigne_regulation.get(f"sonde_{i}", 0))
        segment = config[f"Config_deversoir_{i}"]
        debit = calcul_deversoir_q_serie(df, segment, i).round(3)
        qrestitue_total = qrestitue_total + debit
        i += 1
    return qrestitue_total.round(3)


def compute_qentrant(
    df: pd.DataFrame,
    config_dicts: dict,
    mapping: dict,
    puissance: pd.DataFrame,
    consigne_regulation: dict,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Port de compute_qentrant (load_debit_automate_v2.py). `puissance` doit
    avoir une colonne `power_output` (déjà produite par maj-puissance.py)."""
    for col, targets in mapping.items():
        for dict_name, key in targets:
            if dict_name in config_dicts and col in df.columns:
                df.rename(columns={col: key}, inplace=True)

    qrestitue_total = compute_qrestitue_total(df, config_dicts, consigne_regulation)

    groupes = []
    i = 1
    while f"Config_G{i}" in config_dicts:
        segment = config_dicts[f"Config_G{i}"]
        try:
            hn = calcul_hauteur(df, segment, puissance["power_output"])
        except Exception:
            hn = pd.Series(np.zeros(len(df)), index=df.index)
        qturb = calcul_qturb(df, segment, puissance["power_output"], hn)
        groupes.append(qturb)
        i += 1

    qturbine_total = groupes[0].copy()
    for grp in groupes[1:]:
        qturbine_total = qturbine_total + grp

    qturbine_total = qturbine_total.loc[:, (qturbine_total.fillna(0) != 0).any(axis=0)].clip(lower=0)
    qentrant = qturbine_total.add(qrestitue_total, axis=0).clip(lower=0).round(3)

    return qrestitue_total, qturbine_total, qentrant
