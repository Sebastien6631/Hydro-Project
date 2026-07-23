from __future__ import annotations

from previ_r2d2.common import daily_report


def test_record_then_read_today_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_report, "STATE_DIR", tmp_path)

    daily_report.record("maj-data", "sujet 1", "corps 1", has_errors=False)
    daily_report.record("maj-automate", "sujet 2", "corps 2", has_errors=True)

    entries = daily_report.read_today()

    assert [e["script"] for e in entries] == ["maj-data", "maj-automate"]
    assert entries[1]["has_errors"] is True


def test_read_today_returns_empty_list_when_no_state_file(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_report, "STATE_DIR", tmp_path)

    assert daily_report.read_today() == []


def test_build_email_reports_no_execution_for_missing_script():
    subject, body = daily_report.build_email([])

    assert "0 erreur(s) sur 0 exécution(s)" in subject
    assert "Débit (Hub'Eau/eaufrance) : aucune exécution journalisée aujourd'hui" in body
    assert "Puissance (hydrospot_stream) : aucune exécution journalisée aujourd'hui" in body


def test_build_email_shows_full_body_only_for_failing_runs():
    entries = [
        {"script": "maj-data", "subject": "maj-data OK", "body": "détail ok", "has_errors": False},
        {"script": "maj-automate", "subject": "maj-automate erreur", "body": "détail erreur", "has_errors": True},
    ]

    subject, body = daily_report.build_email(entries)

    assert "1 erreur(s) sur 2 exécution(s)" in subject
    assert "(dernier bilan) maj-data OK" in body
    assert "détail ok" not in body  # run OK : juste le sujet, pas le corps complet
    assert "détail erreur" in body  # run en erreur : corps complet inclus


def test_build_email_includes_onboarding_train_and_predict_archive_sections():
    """onboarding-check/train/predict-archive appellent bien daily_report.record
    (Tasks 3/9/11) -- mais rien ne garantit que build_email les affiche
    réellement dans le mail envoyé : ce test vérifie le rendu final, pas
    seulement l'enregistrement JSONL (read_today())."""
    entries = [
        {"script": "onboarding-check", "subject": "Onboarding — 1 raccordement(s)", "body": "nouveau_G1 : flex_strategy", "has_errors": True},
        {"script": "train", "subject": "Entraînement — apas_G1_G4 h8 PROMU v2", "body": "apas_G1_G4 h8 : PROMU v2", "has_errors": False},
        {"script": "predict-archive", "subject": "Prédiction+archivage — 5 réussie(s), 0 échec(s)", "body": "5 prédiction(s) horaire(s) réussie(s).", "has_errors": False},
    ]

    _, body = daily_report.build_email(entries)

    # onboarding-check est en erreur -> corps complet affiché
    assert "nouveau_G1" in body and "flex_strategy" in body
    # train/predict-archive sont OK -> juste le sujet (bilan compact), pas le corps complet
    assert "PROMU v2" in body  # présent dans le sujet ici
    assert "5 réussie(s), 0 échec(s)" in body