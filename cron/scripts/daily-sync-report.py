#!/usr/bin/env python3
"""daily-sync-report — envoie LE seul mail quotidien de bilan sync (débit +
debit_automate + puissance), à partir des exécutions journalisées par
`maj-data.py`, `maj-automate.py` (horaires) et `maj-puissance.py` (quotidien)
via `previ_r2d2.common.daily_report`.

Envoyé tous les jours, qu'il y ait eu une erreur ou non (confirme aussi que
tout tourne bien). À lancer une fois par jour, après la dernière exécution
horaire du jour (ex. 23:55).

Usage :
    python cron/scripts/daily-sync-report.py
"""

from __future__ import annotations

from previ_r2d2.common import daily_report, mailer


def run() -> int:
    entries = daily_report.read_today()
    subject, body = daily_report.build_email(entries)
    mailer.send_report(subject, body)
    return 1 if any(e["has_errors"] for e in entries) else 0


def main(argv: list[str] | None = None) -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
