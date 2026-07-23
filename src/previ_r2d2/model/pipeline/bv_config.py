"""Assembleurs de config depuis bv.json -- glue code (pas un port direct de
Previ_v2, qui sourçait ces valeurs depuis Centrale_Kfonte_BV.csv +
puissance_config.yaml, fichiers qui n'existent pas côté previ-R2-D2).
bv.json est la source de vérité unique ici. Pas de persistance : fonctions
de calcul pur, l'appelant charge le JSON."""

from __future__ import annotations

SEASON_MAP = {"DJF": "hiver", "MAM": "printemps", "JJA": "ete", "SON": "automne"}


def bv_params_from_bv_json(bv_json: dict) -> dict:
    """Mappe bv.json (imbriqué) vers bv_params (plat), attendu par les fonctions de feature engineering."""
    return {
        "altitude_bv": bv_json["bassin_versant"]["altitude_moyenne_m"],
        "surface_km2": bv_json["bassin_versant"]["surface_km2"],
        "k_base": bv_json["parametres_calage"]["K_base"],
        "exposition": bv_json["parametres_calage"]["exposition"],
        "kc_unit": bv_json["parametres_calage"]["kc_unit"],
    }


def transit_amont_from_bv_json(bv_json: dict) -> dict[str, dict[str, int]]:
    """Traduit les stations amont de bv.json en dict transit_amont/transit_cfg (même format pour les deux)."""
    amont_stations = [s for s in bv_json.get("stations_hydrometriques", []) if s.get("role") == "amont"]
    single = len(amont_stations) == 1

    result = {}
    for station in amont_stations:
        column = "debit_amont" if single else f"debit_amont_{station['code']}"
        transit_h = station.get("transit_vers_reference_h", {})
        result[column] = {SEASON_MAP[k]: v for k, v in transit_h.items() if k in SEASON_MAP}
    return result


def transit_centrale_from_bv_json(bv_json: dict) -> dict[str, int]:
    """Traduit bv_json["transit_vers_centrale_h"] (clés DJF/MAM/JJA/SON) en clés françaises (SEASON_MAP)."""
    transit_h = bv_json.get("transit_vers_centrale_h", {})
    return {SEASON_MAP[k]: v for k, v in transit_h.items() if k in SEASON_MAP}
