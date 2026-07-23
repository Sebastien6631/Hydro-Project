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


# --- API OneGate ---------------------------------------------------------
# TODO(maxime): confirmer l'URL exacte une fois le bon lien récupéré.
#   https://<host>/cgi-bin/api/OneGate/app.py/hydrogrid/...
ONEGATE_BASE_URL = _get(
    "ONEGATE_BASE_URL",
    "https://hydrospot.prosohost.fr/cgi-bin/api/OneGate/app.py",
)

# Jeton d'accès (Authorization: Bearer <token>).
ONEGATE_TOKEN = _get("PREVI_BEARER_TOKEN", "") or _get("ONEGATE_TOKEN", "")

# Jeton de rafraîchissement (optionnel, pour renouveler l'accès).
ONEGATE_REFRESH_TOKEN = _get("REFRESCH", "") or _get("PREVI_REFRESH_TOKEN", "")

# Délai maximal (secondes) d'attente d'une réponse HTTP.
ONEGATE_TIMEOUT = int(_get("ONEGATE_TIMEOUT", "30"))

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

# Racine des ordres de consigne EDF (bridage de puissance), fichiers .true/.false
# de Previ_v2 (write_clean_data_v3_conf.json::ordres_dir, en dur côté Previ_v2 --
# jamais en dur ici). Absente -> aucun événement de consigne n'est trouvé, la
# fusion power_output retombe silencieusement sur la règle de priorité seule.
CONSIGNES_ROOT = Path(_get("PREVI_CONSIGNES_ROOT", ""))

# --- Météo NWP (FTP) -------------------------------------------------------
# Racine NAS où sont mirrorés les fichiers météo NWP (FTP quotidien).
NAS_METEO = Path(_get("PREVI_NAS_METEO", ""))

# Identifiants FTP du fournisseur météo (mirroring quotidien).
FTP_HOST = _get("PREVI_FTP_HOST", "")
FTP_USER = _get("PREVI_FTP_USER", "")
FTP_PASS = _get("PREVI_FTP_PASS", "")

# --- Onboarding BV (délimitation de bassin versant) -----------------------
# MNT France entière (GeoTIFF, EPSG:4326 ou Lambert93) utilisé pour délimiter
# les bassins versants amont. Fichier volumineux, jamais versionné.
PREVI_MNT = Path(_get("PREVI_MNT", ""))

# --- MLflow (tracking + artifacts file-based sur NAS, comme Previ_v2) -----
MLFLOW_URI = _get("PREVI_MLFLOW_URI", "") or None

# --- Notifications e-mail (alerte en cas d'erreur) -----------------------
MAIL_SMTP_HOST = _get("MAIL_SMTP_HOST", "")
MAIL_SMTP_PORT = int(_get("MAIL_SMTP_PORT", "587"))
MAIL_SMTP_USER = _get("MAIL_SMTP_USER", "")
MAIL_SMTP_PASSWORD = _get("MAIL_SMTP_PASSWORD", "")
MAIL_FROM = _get("MAIL_FROM", "") or MAIL_SMTP_USER
MAIL_USE_TLS = str(_get("MAIL_USE_TLS", "true")).strip().lower() in ("1", "true", "yes", "oui")

# Destinataires des alertes. Modifiable ici, ou surchargé via secret_config.py
# (MAIL_RECIPIENTS = liste), ou via l'env MAIL_RECIPIENTS (emails séparés par des virgules).
DEFAULT_MAIL_RECIPIENTS = ["sebastien.veyssiere@barthe-enr.fr"]


def _get_recipients() -> list[str]:
    if _secret is not None and hasattr(_secret, "MAIL_RECIPIENTS"):
        val = getattr(_secret, "MAIL_RECIPIENTS")
        if isinstance(val, (list, tuple)):
            return [str(e).strip() for e in val if str(e).strip()]
        return [e.strip() for e in str(val).split(",") if e.strip()]
    env = os.environ.get("MAIL_RECIPIENTS", "")
    if env:
        return [e.strip() for e in env.split(",") if e.strip()]
    return DEFAULT_MAIL_RECIPIENTS


MAIL_RECIPIENTS = _get_recipients()
