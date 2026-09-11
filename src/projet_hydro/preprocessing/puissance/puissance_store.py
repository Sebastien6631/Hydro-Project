"""Résolution du dossier hydrospot_stream correspondant à un raccordement.

Le dossier projet_hydro (ex. `apas_G1_G4`) est mis en correspondance avec un
dossier hydrospot_stream (ex. `Castillon_Apas_G1`) par mot-clé centrale +
numéro de groupe, ou via `config/puissance_mapping.yaml` en repli explicite.

Utilisé par `preprocessing/onboarding/validation.py` pour valider qu'un
raccordement a bien une source de puissance résolvable -- l'import et la
fusion réelle des fichiers hydrospot_stream (`export_puissance_csv`, NAS
bloqué hors serveur) ont été retirés, cf. spec simplification 2026-07-23 ;
`puissance.csv`/`puissance_horaire.csv` des 3 centrales gardées restent des
données statiques déjà présentes localement.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import yaml

from projet_hydro.common import config


class PuissanceMatchError(RuntimeError):
    """Levée quand le dossier hydrospot_stream ne peut pas être déterminé sans ambiguïté."""


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower()


def load_puissance_mapping(path: Path) -> dict:
    """Charge `config/puissance_mapping.yaml` (dossier -> nom de dossier hydrospot_stream).

    Recours explicite pour les cas où l'heuristique mot-clé+groupe ne peut pas
    fonctionner : plusieurs groupes projet_hydro partagent un seul dossier
    hydrospot_stream (ex. la_bastide_G1_G2_G3 -> LabastideSalat_Village_G1),
    ou le mot-clé du dossier ne matche pas (ex. touzac_g2_G2 -> Touzac_Baque_G2).
    """
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def find_source_folder(rec: dict, mapping: dict | None = None) -> str:
    """Trouve le dossier hydrospot_stream correspondant à ce raccordement.

    Priorité à `mapping` (dossier -> nom de dossier hydrospot_stream, cf.
    `load_puissance_mapping`) si le dossier y figure. Sinon, heuristique :
    le nom du dossier source doit contenir le mot-clé centrale (insensible
    casse/accents) ET se terminer par un des numéros de groupe.
    Lève `PuissanceMatchError` si 0 ou plusieurs candidats (ou si le dossier
    mappé n'existe pas).
    """
    dossier = rec.get("dossier")
    mapping = mapping or {}
    if dossier in mapping:
        folder = mapping[dossier]
        if not (config.PUISSANCE_SOURCE_ROOT / folder).is_dir():
            raise PuissanceMatchError(
                f"{dossier} : dossier mappé {folder!r} introuvable dans "
                f"{config.PUISSANCE_SOURCE_ROOT}"
            )
        return folder

    keyword = _normalize(rec.get("centrale") or rec.get("amenagement") or "")
    groupes = [
        g.get("nom_groupe") for g in rec.get("groupes", []) if g.get("nom_groupe")
    ]
    if not keyword or not groupes:
        raise PuissanceMatchError(
            f"{rec.get('dossier')} : mot-clé centrale ou groupes manquants"
        )

    candidates = []
    for folder in config.PUISSANCE_SOURCE_ROOT.iterdir():
        if not folder.is_dir():
            continue
        name_norm = _normalize(folder.name)
        if keyword not in name_norm:
            continue
        if any(re.search(rf"_{re.escape(g.lower())}$", name_norm) for g in groupes):
            candidates.append(folder.name)

    if len(candidates) != 1:
        raise PuissanceMatchError(
            f"{rec.get('dossier')} : {len(candidates)} correspondance(s) "
            f"hydrospot_stream pour mot-clé={keyword!r} groupes={groupes} "
            f"({candidates})"
        )
    return candidates[0]
