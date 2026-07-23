"""Validation de la complétude d'un config-raccordement.json avant de lancer
puissance/débit/bv pour un nouveau raccordement -- si un champ nécessaire à
la suite (entraînement/prédiction) manque, `missing_fields` en fait la liste
pour le digest quotidien (cf. cron/scripts/onboarding-check.py)."""

from __future__ import annotations

from pathlib import Path

from previ_r2d2.common import config
from previ_r2d2.preprocessing.puissance.puissance_store import (
    PuissanceMatchError,
    find_source_folder,
    load_puissance_mapping,
)

REQUIRED_RACC_FIELDS = ["facteur_debit"]
REQUIRED_GROUPE_FIELDS = ["priorite", "debit_armement_turbine", "debit_max", "rendement"]
VALID_FLEX_STRATEGIES = ("DEFAULT",)


def missing_fields(rec: dict, mapping_path: Path | None = None) -> list[str]:
    """Liste des champs manquants dans `rec` nécessaires pour puissance/débit/bv/
    entraînement. Liste vide = raccordement complet, prêt pour la suite."""
    missing: list[str] = []

    flex_strategy = rec.get("flex_strategy")
    if flex_strategy not in VALID_FLEX_STRATEGIES:
        missing.append("flex_strategy")
    elif flex_strategy == "DEFAULT":
        if not rec.get("station_vigicrue_reference"):
            missing.append("station_vigicrue_reference")
        if not rec.get("stations_vigicrue_amont"):
            missing.append("stations_vigicrue_amont")

    for field in REQUIRED_RACC_FIELDS:
        if rec.get(field) is None:
            missing.append(field)

    groupes = rec.get("groupes") or []
    if not groupes:
        missing.append("groupes")
    for groupe in groupes:
        nom = groupe.get("nom_groupe") or "?"
        for field in REQUIRED_GROUPE_FIELDS:
            if groupe.get(field) is None:
                missing.append(f"groupes[{nom}].{field}")
        if groupe.get("chute_disponible_polynome") is None and groupe.get("chute_disponible_seuils") is None:
            missing.append(f"groupes[{nom}].chute_disponible_polynome|seuils")

    mapping_path = mapping_path if mapping_path is not None else (config.ROOT / "config" / "puissance_mapping.yaml")
    mapping = load_puissance_mapping(mapping_path) if mapping_path.exists() else {}
    try:
        find_source_folder(rec, mapping=mapping)
    except PuissanceMatchError as exc:
        missing.append(f"puissance_mapping ({exc})")

    return missing
