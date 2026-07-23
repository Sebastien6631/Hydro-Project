"""Mirroring FTP récursif des prévisions météo NWP.

Port de `download_ftp_dir` (Previ_v2, `cron/scripts/autodownload_weather_data.py`) :
un dossier distant peut contenir des sous-dossiers et des fichiers, indissociables
autrement qu'en tentant `ftp.cwd(item)` (succès = dossier, `error_perm` = fichier).
"""

from __future__ import annotations

import ftplib
import os


def download_ftp_dir(ftp: ftplib.FTP, remote_dir: str, local_dir: str) -> tuple[int, list[str]]:
    """Télécharge récursivement `remote_dir` -> `local_dir`.

    Renvoie (nombre de fichiers téléchargés, liste des erreurs) -- ne lève jamais,
    une erreur sur un fichier/sous-dossier n'interrompt pas les autres.
    """
    downloaded = 0
    errors: list[str] = []

    try:
        ftp.cwd(remote_dir)
    except Exception as exc:
        errors.append(f"Impossible d'accéder au dossier {remote_dir} : {exc}")
        return downloaded, errors

    os.makedirs(local_dir, exist_ok=True)

    try:
        items = ftp.nlst()
    except ftplib.error_perm as resp:
        if not str(resp).startswith("550"):
            errors.append(f"Liste des fichiers impossible sur {remote_dir} : {resp}")
        return downloaded, errors

    for item in items:
        local_path = os.path.join(local_dir, item)
        try:
            ftp.cwd(item)
            ftp.cwd("..")
        except ftplib.error_perm:
            try:
                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {item}", f.write)
                downloaded += 1
            except Exception as exc:
                errors.append(f"Échec téléchargement {item} : {exc}")
        else:
            sub_downloaded, sub_errors = download_ftp_dir(
                ftp, f"{remote_dir}/{item}", local_path
            )
            downloaded += sub_downloaded
            errors.extend(sub_errors)

    return downloaded, errors
