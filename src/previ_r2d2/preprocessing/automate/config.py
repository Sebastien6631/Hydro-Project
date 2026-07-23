"""Chargement des configs automate (sync + physique) depuis centrales/REFERENCE/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_sync_config(path: Path) -> dict[str, Any]:
    """Charge `automate_sync.yaml` : dossier -> {source_host, source_base, ...}."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_physics_config(path: Path) -> dict[str, Any]:
    """Charge `automate_physics.yaml` : dossier -> {mapping, Config_G1, ...}.

    `.inf`/`-.inf` (syntaxe YAML) sont chargés nativement en flottants infinis
    par `yaml.safe_load` -- contrairement au JSON, aucun post-traitement n'est nécessaire.
    """
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
