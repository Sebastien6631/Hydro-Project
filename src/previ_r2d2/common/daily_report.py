"""Accumulation des bilans de scripts pour un envoi mail unique par jour.

`maj-data.py` (débit) et `maj-automate.py` (debit_automate), horaires, ainsi
que `maj-puissance.py` (quotidien), appellent `record()` au lieu d'envoyer
eux-mêmes un mail par exécution -- `cron/scripts/daily-sync-report.py` relit
`read_today()` une fois par jour et envoie LE seul mail (bilan debit +
debit_automate + puissance), via `mailer`.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import config

STATE_DIR = config.ROOT / "logs" / "daily_sync_state"


def _state_path(day: date | None = None) -> Path:
    return STATE_DIR / f"{(day or date.today()).isoformat()}.jsonl"


def record(script: str, subject: str, body: str, has_errors: bool) -> None:
    """Journalise le bilan d'une exécution de `script` (une ligne JSON par appel)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"script": script, "subject": subject, "body": body, "has_errors": has_errors}
    with _state_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_today() -> list[dict]:
    """Toutes les exécutions journalisées aujourd'hui, dans l'ordre d'écriture."""
    path = _state_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


SCRIPT_LABELS = {
    "maj-data": "Débit (Hub'Eau/eaufrance)",
    "maj-automate": "Débit automate (haute chute)",
    "maj-puissance": "Puissance (hydrospot_stream)",
    "onboarding-check": "Onboarding (validation raccordements)",
    "train": "Entraînement (premier/mensuel)",
    "predict-archive": "Prédiction + archivage horaire",
}


def build_email(entries: list[dict]) -> tuple[str, str]:
    """Compose (sujet, corps) du mail quotidien à partir des entrées de `read_today()`.

    Toujours envoyé (même sans erreur) -- confirme aussi que le sync tourne bien.
    Ne détaille le corps d'une exécution que si elle a des erreurs ; sinon juste
    son sujet (bilan compact), pour ne pas noyer 24 recaps horaires OK dans le mail.
    """
    by_script: dict[str, list[dict]] = {}
    for entry in entries:
        by_script.setdefault(entry["script"], []).append(entry)

    total_errors = sum(1 for e in entries if e["has_errors"])
    total_runs = len(entries)

    lines = [
        f"Bilan quotidien previ-R2-D2 — {total_runs} exécution(s), "
        f"{total_errors} avec erreur(s).",
        "",
    ]
    for script, label in SCRIPT_LABELS.items():
        runs = by_script.get(script, [])
        if not runs:
            lines.append(f"=== {label} : aucune exécution journalisée aujourd'hui ===")
            lines.append("")
            continue
        failing = [r for r in runs if r["has_errors"]]
        lines.append(f"=== {label} : {len(runs)} exécution(s), {len(failing)} avec erreur(s) ===")
        for run in failing:
            lines.append(f"--- {run['subject']} ---")
            lines.append(run["body"])
            lines.append("")
        if not failing:
            lines.append("(dernier bilan) " + runs[-1]["subject"])
            lines.append("")

    subject = f"[previ-record] Bilan quotidien sync — {total_errors} erreur(s) sur {total_runs} exécution(s)"
    return subject, "\n".join(lines)
