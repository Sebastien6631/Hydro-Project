"""Envoi d'e-mails d'alerte via SMTP.

Méthode standard (smtplib + STARTTLS). Non bloquant côté appelant : si le SMTP
n'est pas configuré, `send()` lève `MailNotConfigured` que l'appelant peut
attraper pour ne pas planter.

NB : implémentation générique en attendant de réutiliser « la même méthode » que
le script de référence (à fournir).
"""

from __future__ import annotations

import smtplib
import sys
from email.message import EmailMessage

from . import config


class MailNotConfigured(RuntimeError):
    """Levée quand la configuration SMTP est incomplète."""


def is_configured() -> bool:
    """Vrai si l'envoi de mail est possible (host + expéditeur + destinataires)."""
    return bool(config.MAIL_SMTP_HOST and config.MAIL_FROM and config.MAIL_RECIPIENTS)


def send(subject: str, body: str, recipients: list[str] | None = None) -> None:
    """Envoie un e-mail texte. Lève `MailNotConfigured` si SMTP non paramétré."""
    recipients = recipients or config.MAIL_RECIPIENTS
    if not is_configured():
        raise MailNotConfigured(
            "SMTP non configuré (MAIL_SMTP_HOST / MAIL_FROM / MAIL_RECIPIENTS)."
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.MAIL_FROM
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    with smtplib.SMTP(config.MAIL_SMTP_HOST, config.MAIL_SMTP_PORT, timeout=30) as smtp:
        if config.MAIL_USE_TLS:
            smtp.starttls()
        if config.MAIL_SMTP_USER:
            smtp.login(config.MAIL_SMTP_USER, config.MAIL_SMTP_PASSWORD)
        smtp.send_message(msg)


def send_report(subject: str, body: str) -> bool:
    """Envoie un mail (recap ou alerte). Ne lève jamais : renvoie True si envoyé.

    Si le SMTP n'est pas configuré (ou échoue), prévient sur stderr et renvoie False.
    C'est le point d'entrée unique des scripts : un seul mail par exécution.
    """
    try:
        send(subject, body)
        print(f"✉️  Mail envoyé à : {', '.join(config.MAIL_RECIPIENTS)}")
        return True
    except MailNotConfigured as exc:
        print(f"⚠️  Mail non envoyé : {exc}", file=sys.stderr)
        return False
    except Exception as exc:  # échec SMTP
        print(f"⚠️  Échec de l'envoi du mail : {exc}", file=sys.stderr)
        return False


def notify_errors(context: str, errors: list[str]) -> bool:
    """Envoie UNE alerte groupant toutes les `errors` d'un script."""
    subject = f"[previ-record] {context} — {len(errors)} erreur(s)"
    body = f"{context}\n\nErreurs :\n" + "\n".join(f"- {e}" for e in errors)
    return send_report(subject, body)
