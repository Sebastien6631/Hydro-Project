from __future__ import annotations

import json
from pathlib import Path


def write_config_raccordement(centrales_dir: Path, dossier: str, centrale_uuid: str,
                               lat: float, lon: float, alt: float | None) -> Path:
    """Écrit un config-raccordement.json minimal pour les tests."""
    d = centrales_dir / dossier
    d.mkdir(parents=True, exist_ok=True)
    path = d / "config-raccordement.json"
    path.write_text(
        json.dumps({
            "dossier": dossier,
            "centrale_uuid": centrale_uuid,
            "adresse_lat": lat,
            "adresse_lng": lon,
            "adresse_alt": alt,
        }),
        encoding="utf-8",
    )
    return path
