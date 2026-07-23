"""Archivage horaire des JSON de prévision -- avant que predict_orchestrator
n'écrase prevision.json/enchere.json avec la nouvelle heure, copie l'existant
vers NAS_ARCHIVE_ROOT/<dossier>/<AAAA>/<MM>/<JJ>/<nom>_<horodatage>.json (un
fichier par heure archivée, historique complet).

Le fichier existant a été produit lors du run précédent, une heure avant
l'archivage courant (ex : archivage à 10h -> fichier produit à 9h). Le nom
archivé porte donc l'heure de production (now - 1h), pas l'heure d'archivage,
afin que l'historique reflète fidèlement quand chaque prévision a été émise.

Contrat d'appel : l'appelant doit passer le `now` COURANT, tel quel -- la
même valeur que celle donnée ensuite à `run_prediction(dossier, horizon, ...,
now)` pour générer la nouvelle heure. Ne jamais pré-soustraire une heure
avant d'appeler cette fonction : l'ajustement -1h est déjà fait ici, le
refaire côté appelant décalerait le nom archivé de 2h au lieu d'1h.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


def archive_previous_json(dossier: str, centrales_dir: Path, archive_root: Path, now: pd.Timestamp) -> list[Path]:
    """Copie prevision.json/enchere.json (s'ils existent déjà) de
    centrales/<dossier>/ vers l'archive NAS. Retourne les chemins archivés
    (liste vide si aucun fichier n'existait encore -- 1er run d'une centrale)."""
    src_dir = centrales_dir / dossier
    produced_at = now - pd.Timedelta(hours=1)
    dest_dir = archive_root / dossier / produced_at.strftime("%Y") / produced_at.strftime("%m") / produced_at.strftime("%d")

    archived = []
    for name in ("prevision.json", "enchere.json"):
        src = src_dir / name
        if not src.exists():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{src.stem}_{produced_at.strftime('%Y%m%d_%Hh')}.json"
        shutil.copy2(src, dest)
        archived.append(dest)
    return archived
