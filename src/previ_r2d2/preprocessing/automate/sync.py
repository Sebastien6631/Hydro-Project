"""Sync rsync/SSH unifié des capteurs automate (port de sync.py + sync_melles.py Previ_v2).

Contrairement à `sync.py` (Previ_v2, Bonneval), les corrections (multiply/offset)
ne sont appliquées qu'aux fichiers NOUVELLEMENT renommés dans ce run -- pas à tout
l'historique -- pour éviter de réappliquer une correction non idempotente à chaque
run (bug identifié dans `sync.py` : `dest.rglob("*.txt")` sans filtrage sur les
fichiers nouveaux, cf. spec).
"""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path


def apply_correction(filepath: Path, multiply: float, offset: float) -> bool:
    """Applique `multiply`/`offset` sur un fichier `HH:MM:SS  valeur`.

    Ignore les fichiers au format original (contient `;`). Renvoie True si corrigé.
    """
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines or ";" in lines[0]:
        return False

    corrected = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                val = float(parts[1]) * multiply + offset
                val_str = str(int(val)) if val == int(val) else f"{val:.3f}".rstrip("0").rstrip(".")
                corrected.append(f"{parts[0]}  {val_str}")
            except ValueError:
                corrected.append(line)
        else:
            corrected.append(line)

    filepath.write_text("\n".join(corrected) + "\n", encoding="utf-8")
    return True


def sync_variable(
    var: str,
    source_host: str,
    source_base: str,
    dest_dir: Path,
    corrections: dict | None,
    source_var: str | None = None,
) -> tuple[int, list[Path]]:
    """Rsync `source_var` (nom de dossier côté source -- souvent différent de `var`,
    ex. Melles : `PosInj1` distant -> `ML2_POS_STAB_INJEC1_G1` local, cf.
    `config_sync.yaml`/`sync_melles.py` côté Previ_v2 ; par défaut identique à `var`,
    comme Bonneval où source et destination partagent le même nom) depuis
    `source_host:source_base/` vers `dest_dir`, renomme les fichiers sans extension
    en `.txt`, corrige UNIQUEMENT les nouveaux.

    Renvoie (nombre de fichiers renommés, liste des fichiers corrigés dans ce run).
    """
    source_var = source_var or var
    dest_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["rsync", "-az", "--ignore-existing", "--timeout=30",
         f"{source_host}:{source_base}/{source_var}/", f"{dest_dir}/"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rsync {source_var} (code {result.returncode}) : {result.stderr.strip()}")

    # Cible les dossiers jour récents (aujourd'hui + hier) 
    newly_renamed = []
    today = datetime.date.today()
    for day in (today - datetime.timedelta(days=1), today):
        day_dir = dest_dir / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
        if not day_dir.is_dir():
            continue
        for f in day_dir.iterdir():
            if f.is_file() and f.suffix == "":
                new_path = f.with_suffix(".txt")
                f.rename(new_path)
                newly_renamed.append(new_path)

    corrected_files = []
    if corrections:
        for f in newly_renamed:
            if apply_correction(f, corrections["multiply"], corrections["offset"]):
                corrected_files.append(f)

    return len(newly_renamed), corrected_files
