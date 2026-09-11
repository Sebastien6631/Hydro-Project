"""Marqueurs DVC (cache: false) -- signalent juste qu'un stage a tourné, pour
que data_preparation ait une vraie flèche DVC vers debit/debit_automate/
puissance (dont la vraie sortie est externe à DVC, cf. always_changed)."""

from __future__ import annotations

import json
from datetime import datetime

from . import config

MARKERS_DIR = config.ROOT / "logs" / "dvc_markers"


def write(stage: str) -> None:
    MARKERS_DIR.mkdir(parents=True, exist_ok=True)
    path = MARKERS_DIR / f"{stage}.json"
    path.write_text(json.dumps({"last_run": datetime.now().isoformat()}), encoding="utf-8")
