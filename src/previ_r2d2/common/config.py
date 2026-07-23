"""Configuration centrale du projet.

Les valeurs (URL, jeton) sont lues, par ordre de priorité :
  1. depuis `secret_config.py` s'il existe (fichier local, NON versionné,
     à créer à partir de `secret_config.example.py`) ;
  2. sinon depuis les variables d'environnement ;
  3. sinon les valeurs par défaut ci-dessous.

Le fichier `secret_config.py` contient les secrets : il est ignoré par git et
ne doit jamais être commité.
"""

from __future__ import annotations

import os
from pathlib import Path

# Racine du projet (previ-R2-D2/, 3 niveaux au-dessus de ce fichier :
# src/previ_r2d2/common/config.py -> src/previ_r2d2/ -> src/ -> racine).
ROOT = Path(__file__).resolve().parents[3]

# Sortie OneGate + bv.json par dossier, et fichiers de référence partagés
# (config-general.json, bv_rules.json, centrales_calibration.json,
# shapefiles/, files/) sous centrales/REFERENCE/.
CENTRALES_DIR = ROOT / "centrales"
REFERENCE_DIR = CENTRALES_DIR / "REFERENCE"

# Charge le module de secrets local s'il est présent.
try:
    from . import secret_config as _secret  # type: ignore
except ImportError:
    _secret = None


def _get(name: str, default: str) -> str:
    """Valeur de `name` : secret_config.py > variable d'env > défaut."""
    if _secret is not None and hasattr(_secret, name):
        return getattr(_secret, name)
    return os.environ.get(name, default)


# --- Stockage des débits (NAS) --------------------------------------------
# Racine NAS où sont stockés les CSV réels (source de vérité).
# previ-R2-D2/data/<dossier>/<fichier>.csv n'est qu'un lien symbolique dessus.
NAS_DATA_ROOT = Path(_get("PREVI_NAS_DATA_ROOT", ""))

# --- Modèles entraînés (versionnés DVC + tag git) --------------------------
# models/<dossier>/h<horizon>/ = modèle en production, source de vérité lue
# par predict_orchestrator. weights/hybrid/<dossier>/h<horizon>/ (inchangé)
# reste la zone de travail/candidat (run.py, entraînement en cours).
MODELS_DIR = ROOT / "models"

# --- Archivage horaire des prévisions --------------------------------------
NAS_ARCHIVE_ROOT = NAS_DATA_ROOT / "ARCHIVE"

# Racine hydrospot_stream (source de puissance, déjà utilisée par Previ_v2).
PUISSANCE_SOURCE_ROOT = Path(_get("PREVI_PUISSANCE_SOURCE_ROOT", ""))

# --- Météo NWP (lecture de fichiers bruts) ---------------------------------
# Racine où seraient mirorrés les fichiers météo NWP bruts (acquisition FTP
# retirée -- hors périmètre du projet de cours, cf. nwp_ftp.py supprimé).
# `nwp_reader.read_points` renvoie un DataFrame vide si ce dossier est
# absent/vide -- jamais de crash, dégradation gracieuse en attendant le
# sous-projet "API météo publique".
NAS_METEO = Path(_get("PREVI_NAS_METEO", ""))

# --- Onboarding BV (délimitation de bassin versant) -----------------------
# MNT France entière (GeoTIFF, EPSG:4326 ou Lambert93) utilisé pour délimiter
# les bassins versants amont. Fichier volumineux, jamais versionné.
PREVI_MNT = Path(_get("PREVI_MNT", ""))
