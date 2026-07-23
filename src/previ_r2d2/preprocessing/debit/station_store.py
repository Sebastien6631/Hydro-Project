"""Stockage des CSV de débit, avec dédup par code station.

Un même code station (ex. un amont partagé par deux centrales) n'est stocké
qu'une fois. Un index (`_index.json`) retient, pour chaque code, le chemin
du fichier réel qui le porte. Les autres dossiers qui référencent ce code
reçoivent un symlink local vers ce fichier réel.

`config.NAS_DATA_ROOT` == `config.CENTRALES_DIR` dans ce projet (un seul
niveau de stockage -- pas de NAS distinct hors serveur de production).
"""

from __future__ import annotations

import json
from pathlib import Path

from previ_r2d2.common import config


def index_path() -> Path:
    """Chemin de `_index.json`, relu à chaque appel (jamais figé à l'import)."""
    return config.NAS_DATA_ROOT / "_index.json"


def filename_for(station: str, role: str) -> str:
    """Nom de fichier selon le rôle : `<station>.csv` (reference) ou `amont_<station>.csv`."""
    return f"{station}.csv" if role == "reference" else f"amont_{station}.csv"


def _load_index() -> dict[str, str]:
    if index_path().exists():
        return json.loads(index_path().read_text(encoding="utf-8"))
    return {}


def _save_index(index: dict[str, str]) -> None:
    index_path().parent.mkdir(parents=True, exist_ok=True)
    index_path().write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_nas_path(station: str, dossier: str, role: str) -> tuple[Path, str]:
    """Chemin NAS réel pour ce (station, dossier, role) + statut.

    Renvoie (chemin_réel, statut) :
      - "new"      -> aucun fichier réel n'existe encore pour ce code station ;
                       à l'appelant de l'importer à ce chemin.
      - "existing" -> ce dossier est déjà le porteur réel du fichier ; mise à
                       jour incrémentale possible.
      - "linked"   -> un AUTRE dossier porte déjà ce code station ; un symlink
                       NAS a été créé vers son fichier réel, rien à importer
                       ici (l'import/MAJ se fait via le dossier porteur).
    """
    filename = filename_for(station, role)
    own_path = config.NAS_DATA_ROOT / dossier / filename

    index = _load_index()
    real_rel = index.get(station)

    if real_rel is None:
        # Première fois qu'on voit ce code station : ce dossier en devient le porteur réel.
        index[station] = f"{dossier}/{filename}"
        _save_index(index)
        return own_path, "new"

    real_path = config.NAS_DATA_ROOT / real_rel
    if real_path == own_path:
        return own_path, "new" if not own_path.exists() else "existing"

    # Un autre dossier porte déjà ce code -> symlink NAS vers son fichier réel.
    own_path.parent.mkdir(parents=True, exist_ok=True)
    if not (own_path.is_symlink() and own_path.resolve() == real_path.resolve()):
        if own_path.is_symlink() or own_path.exists():
            own_path.unlink()
        own_path.symlink_to(real_path)
    return real_path, "linked"
