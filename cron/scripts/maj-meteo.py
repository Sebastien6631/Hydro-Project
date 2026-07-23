#!/usr/bin/env python3
"""maj-meteo — télécharge les prévisions météo NWP du jour depuis le FTP.

Mirrore `/8_Custom/1_Weather/Forecast/Parameters/EC_OP/{YYYY}/{MM}/{DD}/` (FTP)
vers `<NAS_METEO>/{YYYY}/{MM}/{DD}/`. Envoie UN SEUL e-mail de recap si la
connexion échoue, si des fichiers échouent au téléchargement, ou si le nombre
de fichiers présents localement est sous le seuil minimum attendu.

Usage :
    python cron/scripts/maj-meteo.py
"""

from __future__ import annotations

import datetime
import ftplib
import sys

from previ_r2d2.common import config, mailer
from previ_r2d2.preprocessing.meteo.nwp_ftp import download_ftp_dir

REMOTE_ROOT = "/8_Custom/1_Weather/Forecast/Parameters/EC_OP"
MIN_FILES_EXPECTED = 24  # un run NWP complet contient ~145 fichiers


def run() -> int:
    today = datetime.date.today()
    year, month, day = today.strftime("%Y"), today.strftime("%m"), today.strftime("%d")
    remote_dir = f"{REMOTE_ROOT}/{year}/{month}/{day}"
    local_dir = config.NAS_METEO / year / month / day
    local_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    errors: list[str] = []
    try:
        with ftplib.FTP(config.FTP_HOST) as ftp:
            ftp.login(config.FTP_USER, config.FTP_PASS)
            downloaded, errors = download_ftp_dir(ftp, remote_dir, str(local_dir))
    except OSError as exc:
        errors.append(f"Erreur disque/I/O : {exc}")
    except Exception as exc:
        errors.append(f"Échec connexion FTP : {exc}")

    total_local = len(list(local_dir.glob("*.csv")))
    too_few = total_local < MIN_FILES_EXPECTED

    if errors or too_few:
        lines = [
            f"Date : {today} — Dossier distant : {remote_dir}",
            f"Fichiers présents localement : {total_local} (seuil minimum : {MIN_FILES_EXPECTED})",
            f"Fichiers téléchargés dans ce run : {downloaded}",
            "",
        ]
        if too_few:
            lines.append(
                f"Données incomplètes : seulement {total_local} fichier(s) disponibles "
                f"(minimum attendu : {MIN_FILES_EXPECTED}). Les prévisions seront dégradées."
            )
        if errors:
            lines.append(f"\n{len(errors)} erreur(s) lors du téléchargement :")
            lines.extend(f"  - {e}" for e in errors)
        subject = f"[previ-record] maj-meteo — {total_local} fichier(s), {len(errors)} erreur(s)"
        mailer.send_report(subject, "\n".join(lines))
    else:
        print(f"✓ {total_local} fichier(s) météo présents pour {today} ({downloaded} téléchargé(s) dans ce run).")

    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())
