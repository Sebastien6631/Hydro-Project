"""Transformation des données memorandum `/select` en config par raccordement.

L'endpoint `/hydrogrid/memorandum/select` renvoie l'arbre memorandum réduit aux
clés demandées (amenagement → centrales → raccordements → groupes), hiérarchie
conservée et sous-objets/listes vides omis.

Ce module aplatit cet arbre en une liste de raccordements et construit, pour
chacun, le nom de dossier et l'enregistrement à sérialiser.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Clés demandées à l'endpoint /select (ordre indifférent).
SELECT_KEYS = [
    "nom_amenagement",
    "nom_centrale",
    "nom_groupe",
    "flexibilite_rte",
    "flexibilite_pool",
    "flexibilite_type",
    "flex_strategy",
    "station_vigicrue_reference",
    "stations_vigicrue_amont",
    "adresse_lat",
    "adresse_lng",
    "adresse_alt",
    "uuid",
    "debit_reserve",
    "debit_non_turbinable",
    "facteur_debit",
    "priorite",
    "debit_armement_turbine",
    "debit_max",
    "rendement",
    "chute_disponible_polynome",
    "chute_disponible_seuils",
]

# `periode_ete` n'est jamais envoyé par OneGate : previ-R2-D2 le calcule à
# partir de `flex_strategy`, uniquement quand `debit_non_turbinable` (niveau
# centrale) est la forme saisonnière (liste [été, hiver]).
PERIODE_ETE_HAUTE_CHUTE = [[7, 1], [10, 31]]
PERIODE_ETE_DEFAUT = [[6, 1], [10, 31]]


def compute_periode_ete(flex_strategy: str | None) -> list[list[int]]:
    """1er juillet -> 31 octobre en haute chute, 1er juin -> 31 octobre sinon."""
    return PERIODE_ETE_HAUTE_CHUTE if flex_strategy == "HAUTE_CHUTE" else PERIODE_ETE_DEFAUT

# Champs puissance_config.yaml portés au niveau de chaque groupe (mêmes
# conditions : absents si non fournis par OneGate).
GROUPE_EXTRA_KEYS = [
    "priorite",
    "debit_armement_turbine",
    "debit_max",
    "rendement",
    "chute_disponible_polynome",
    "chute_disponible_seuils",
]

_WHITESPACE = re.compile(r"\s+")


def _is_true(value: Any) -> bool:
    """Vrai pour le booléen True ou la chaîne "true" (la BDD renvoie "true")."""
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _split_stations(value: Any) -> list[str]:
    """Liste de codes station depuis une chaîne "code1,code2" (ou une liste)."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(s).strip() for s in value if str(s).strip()]
    return [s.strip() for s in str(value).split(",") if s.strip()]


def slugify_dossier(nom_centrale: str | None, noms_groupes: list[str]) -> str:
    """Nom de dossier = nom_centrale + noms des groupes, espaces -> `_`.

    Ex. ("Centrale amont", ["G1", "G2"]) -> "Centrale_amont_G1_G2".
    Les séparateurs de chemin (`/`, `\\`) sont neutralisés pour la sécurité.
    """
    base = " ".join([nom_centrale or "", *noms_groupes]).strip()
    base = base.replace("/", "_").replace("\\", "_")
    return _WHITESPACE.sub("_", base) or "sans_nom"


def decode_onegate_value(value: Any) -> Any:
    """OneGate stocke tout en texte (BDD) : les nombres et listes arrivent en
    chaînes JSON-encodées (ex. "2", "[80.07]"), comme le "true" de
    `flexibilite_rte` déjà décodé par `_is_true`. Chaîne vide -> `None`
    (absent) ; chaîne non JSON -> renvoyée telle quelle (texte réel)."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return value


def decode_chute_disponible(value: str) -> Any:
    """Repli pour `chute_disponible_polynome`/`chute_disponible_seuils` quand
    `decode_onegate_value` échoue (valeur toujours une chaîne) : OneGate omet
    parfois les crochets extérieurs pour plusieurs segments (ex. "[a],[b]"
    au lieu de "[[a],[b]]", vu sur Melles/Nancy/Touzac), et utilise ".inf"
    (syntaxe YAML, vu sur Melles). `json.loads` accepte nativement le token
    `Infinity`, mais ce n'est pas du JSON standard (json.dumps l'écrirait
    tel quel, invalide pour un parseur strict) -- converti en chaîne
    "Inf"/"-Inf" plutôt qu'en flottant infini. Renvoie la chaîne brute si
    le nouvel essai échoue."""
    fixed = value.replace("-.inf", '"-Inf"').replace(".inf", '"Inf"')
    try:
        return json.loads(f"[{fixed}]")
    except (json.JSONDecodeError, ValueError):
        return value


def pick_present(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Sous-dict de `source` restreint à `keys`, décodé (`decode_onegate_value`)
    et en omettant les clés absentes ou vides (pas de `None` trompeur -- même
    convention que `bv.json`)."""
    result = {}
    for key in keys:
        if key not in source:
            continue
        value = decode_onegate_value(source[key])
        if value is not None:
            result[key] = value
    return result


def flatten_flexibilite(tree: Any, *, rte_only: bool = True) -> list[dict[str, Any]]:
    """Aplatit l'arbre memorandum en une liste de raccordements.

    Une entrée par raccordement. Si `rte_only`, ne conserve que les
    raccordements dont `flexibilite_rte` est vrai.
    """
    records: list[dict[str, Any]] = []

    for amenagement in tree or []:
        nom_amenagement = amenagement.get("nom_amenagement")

        for centrale in amenagement.get("centrales", []) or []:
            nom_centrale = centrale.get("nom_centrale")

            for racc in centrale.get("raccordements", []) or []:
                rte = _is_true(racc.get("flexibilite_rte"))
                if rte_only and not rte:
                    continue

                groupes = racc.get("groupes", []) or []
                noms_groupes = [
                    g.get("nom_groupe") for g in groupes if g.get("nom_groupe")
                ]

                # Nom lisible, à la façon du front : "Centrale amont — G1, G2".
                nom = nom_centrale or ""
                if noms_groupes:
                    nom = f"{nom_centrale} — {', '.join(noms_groupes)}"

                groupes_list = []
                for g in groupes:
                    groupe_entry = {
                        "uuid": g.get("uuid"),
                        "nom_groupe": g.get("nom_groupe"),
                        **pick_present(g, GROUPE_EXTRA_KEYS),
                    }
                    for key in ("chute_disponible_polynome", "chute_disponible_seuils"):
                        if isinstance(groupe_entry.get(key), str):
                            groupe_entry[key] = decode_chute_disponible(groupe_entry[key])
                    groupes_list.append(groupe_entry)

                record = {
                    "amenagement": nom_amenagement,
                    "amenagement_uuid": amenagement.get("uuid"),
                    "centrale": nom_centrale,
                    "centrale_uuid": centrale.get("uuid"),
                    "adresse_lat": centrale.get("adresse_lat"),
                    "adresse_lng": centrale.get("adresse_lng"),
                    "adresse_alt": centrale.get("adresse_alt"),
                    "raccordement_uuid": racc.get("uuid"),
                    "nom": nom,
                    "dossier": slugify_dossier(nom_centrale, noms_groupes),
                    "rte": rte,
                    "pool": racc.get("flexibilite_pool"),
                    "type": racc.get("flexibilite_type"),
                    "flex_strategy": racc.get("flex_strategy"),
                    "station_vigicrue_reference": racc.get("station_vigicrue_reference"),
                    "stations_vigicrue_amont": _split_stations(
                        racc.get("stations_vigicrue_amont")
                    ),
                    "groupes": groupes_list,
                }

                debit_reserve = decode_onegate_value(centrale.get("debit_reserve"))
                if debit_reserve is not None:
                    record["debit_reserve"] = debit_reserve

                debit_non_turbinable = decode_onegate_value(centrale.get("debit_non_turbinable"))
                if isinstance(debit_non_turbinable, str) and "," in debit_non_turbinable:
                    # OneGate envoie parfois "valeur_ete,valeur_hiver" sans
                    # crochets (ex. Melles) -- json.loads échoue, on retente
                    # en découpant par virgule comme _split_stations.
                    try:
                        debit_non_turbinable = [
                            float(v.strip()) for v in debit_non_turbinable.split(",")
                        ]
                    except ValueError:
                        pass
                if isinstance(debit_non_turbinable, list) and len(debit_non_turbinable) == 2:
                    record["q_non_turbinable_ete"], record["q_non_turbinable_hiver"] = debit_non_turbinable
                    record["periode_ete"] = compute_periode_ete(racc.get("flex_strategy"))
                elif debit_non_turbinable is not None:
                    record["debit_non_turbinable"] = debit_non_turbinable

                facteur_debit = decode_onegate_value(racc.get("facteur_debit"))
                if facteur_debit is not None:
                    record["facteur_debit"] = facteur_debit

                records.append(record)

    return records


def deduplicate_dossiers(records: list[dict[str, Any]]) -> None:
    """Assure l'unicité du champ `dossier` (collisions -> suffixe uuid court)."""
    seen: set[str] = set()
    for rec in records:
        name = rec["dossier"]
        if name in seen:
            suffix = (rec.get("raccordement_uuid") or "")[:7]
            name = f"{name}_{suffix}" if suffix else f"{name}_{len(seen)}"
            rec["dossier"] = name
        seen.add(name)
