"""Calcul puissance/chute/rendement/débit turbinable-réserve à partir du
débit entrant -- port simplifié de compute_Pdisponible/compute_hydraulic_estimates
(Previ_v2, load_debit_automate_v2.py/debit_to_puissance.py) : répartition
prioritaire complète du débit entre groupes (comme Previ_v2), mais
chute_estimee/rendement_estime rapportés uniquement pour le groupe
prioritaire plutôt qu'un détail par groupe (décision utilisateur -- le
JSON de sortie reste plat). puissance/pmax_dyn restent des totaux de
site (somme sur tous les groupes)."""

from __future__ import annotations

import pandas as pd

GRAVITE = 9.81


def eval_polynome(coeffs: list[float], x: float) -> float:
    """Évalue un polynôme (coefficients du degré 0 au degré n)."""
    return sum(a * x ** i for i, a in enumerate(coeffs))


def eval_chute(groupe: dict, q: float) -> float:
    """Chute (m) en fonction du débit -- polynôme simple ou segmenté par seuils."""
    polynome = groupe["chute_disponible_polynome"]
    seuils = groupe.get("chute_disponible_seuils")
    if seuils is None:
        return max(eval_polynome(polynome, q), 0.0)
    for (s_min, s_max), coeffs in zip(seuils, polynome):
        if s_min <= q <= s_max:
            return max(eval_polynome(coeffs, q), 0.0)
    return max(eval_polynome(polynome[-1], q), 0.0)


def eval_rendement(groupe: dict, q: float) -> float:
    """Rendement (%) en fonction du débit turbiné -- toujours un polynôme simple (jamais segmenté)."""
    return max(eval_polynome(groupe["rendement"], q), 0.0)


def compute_q_non_turbinable(rec: dict, when: pd.Timestamp) -> float:
    """Débit réservé/non turbinable -- constant, ou saisonnier été/hiver si les 2 champs sont présents."""
    if "q_non_turbinable_ete" in rec and "q_non_turbinable_hiver" in rec:
        (sm, sd), (em, ed) = rec["periode_ete"]
        month_day = (when.month, when.day)
        is_summer = (month_day >= (sm, sd)) and (month_day <= (em, ed))
        return float(rec["q_non_turbinable_ete"] if is_summer else rec["q_non_turbinable_hiver"])
    return float(rec.get("debit_non_turbinable", 0.0))


def dispatch_groupes(q_disponible: float, groupes: list[dict]) -> list[dict]:
    """Répartit le débit disponible entre groupes par priorité croissante (1 = premier servi) ; ajoute 'debit_turbine' à chaque groupe."""
    groupes_tries = sorted(groupes, key=lambda g: g["priorite"])
    q_restant = q_disponible
    result = []
    for g in groupes_tries:
        debit_max = g["debit_max"]
        debit_arm = g["debit_armement_turbine"]
        q_groupe = min(q_restant, debit_max)
        if q_groupe < debit_arm:
            q_groupe = 0.0
        q_restant = max(q_restant - q_groupe, 0.0)
        result.append({**g, "debit_turbine": q_groupe})
    return result


def compute_hydraulic_point(q_entrant: float, rec: dict, when: pd.Timestamp) -> dict:
    """Calcule débit turbinable/réserve, puissance/pmax_dyn (site, somme des groupes), chute/rendement (groupe prioritaire) pour un débit entrant donné."""
    groupes = rec.get("groupes", [])
    q_non_turb = compute_q_non_turbinable(rec, when)
    q_disponible = max(q_entrant - q_non_turb, 0.0)

    if not groupes:
        return {
            "debit_turbinable": round(q_disponible, 3),
            "debit_reserve": round(min(q_entrant, q_non_turb), 3),
            "puissance": 0.0, "pmax_dyn": 0.0, "chute_estimee": 0.0, "rendement_estime": 0.0,
        }

    dispatched = dispatch_groupes(q_disponible, groupes)

    puissance = 0.0
    pmax_dyn = 0.0
    for g in dispatched:
        chute_at_q = eval_chute(g, q_entrant)
        rendement_turbine = eval_rendement(g, g["debit_turbine"])
        puissance += g["debit_turbine"] * rendement_turbine / 100.0 * chute_at_q * GRAVITE
        rendement_max = eval_rendement(g, g["debit_max"])
        pmax_dyn += g["debit_max"] * rendement_max / 100.0 * chute_at_q * GRAVITE

    priorite_1 = min(dispatched, key=lambda g: g["priorite"])
    chute_estimee = eval_chute(priorite_1, q_entrant)
    rendement_estime = eval_rendement(priorite_1, priorite_1["debit_turbine"])
    debit_turbinable = sum(g["debit_turbine"] for g in dispatched)
    debit_reserve = min(q_entrant, q_non_turb)

    return {
        "debit_turbinable": round(debit_turbinable, 3),
        "debit_reserve": round(debit_reserve, 3),
        "puissance": round(puissance, 1),
        "pmax_dyn": round(pmax_dyn, 1),
        "chute_estimee": round(chute_estimee, 3),
        "rendement_estime": round(rendement_estime, 2),
    }
