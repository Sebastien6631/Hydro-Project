from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from previ_r2d2.preprocessing.puissance import puissance_store


def test_read_source_coerces_date_to_datetime64_even_when_parse_fails(tmp_path):
    """Une ligne corrompue (doublon d'écriture, vu en prod sur campagne_G1_G2)
    fait échouer le parsing vectorisé de `parse_dates=["Date"]` ; pandas retombe
    alors sur un dtype non-datetime qui n'est PAS forcément "object" (ex. `str`
    sous pandas 3.x) — le repli doit re-coercer quel que soit ce dtype, sinon
    le merge suivant plante avec "merge on datetime64[us] and str columns"."""
    csv_path = tmp_path / "date_power.csv"
    csv_path.write_text(
        "Date;Puissance\n"
        "2026-05-30 21:56:00;100.0\n"
        "2026-05-32026-05-30 21:57:00;300.0\n"
        "2026-05-30 21:58:00;150.0\n",
        encoding="utf-8",
    )

    df = puissance_store._read_source(csv_path)

    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert len(df) == 2


def test_read_source_keeps_clean_file_as_datetime64(tmp_path):
    csv_path = tmp_path / "date_power.csv"
    csv_path.write_text(
        "Date;Puissance\n2026-05-30 21:56:00;100.0\n2026-05-30 21:57:00;150.0\n",
        encoding="utf-8",
    )

    df = puissance_store._read_source(csv_path)

    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert len(df) == 2


def test_find_source_folder_uses_explicit_mapping_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(puissance_store.config, "PUISSANCE_SOURCE_ROOT", tmp_path)
    (tmp_path / "Clairac_ruePlage_G1").mkdir()
    rec = {"dossier": "clairac_rd_G2", "centrale": "clairac_rd",
           "groupes": [{"nom_groupe": "G2"}]}

    folder = puissance_store.find_source_folder(rec, mapping={"clairac_rd_G2": "Clairac_ruePlage_G1"})

    assert folder == "Clairac_ruePlage_G1"


def test_find_source_folder_raises_when_mapped_folder_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(puissance_store.config, "PUISSANCE_SOURCE_ROOT", tmp_path)
    rec = {"dossier": "clairac_rd_G2", "centrale": "clairac_rd",
           "groupes": [{"nom_groupe": "G2"}]}

    with pytest.raises(puissance_store.PuissanceMatchError):
        puissance_store.find_source_folder(rec, mapping={"clairac_rd_G2": "Clairac_ruePlage_G1"})


def test_find_source_folder_falls_back_to_heuristic_when_dossier_not_mapped(tmp_path, monkeypatch):
    monkeypatch.setattr(puissance_store.config, "PUISSANCE_SOURCE_ROOT", tmp_path)
    (tmp_path / "Castillon_Apas_G1").mkdir()
    rec = {"dossier": "apas_G1_G4", "centrale": "apas",
           "groupes": [{"nom_groupe": "G1"}, {"nom_groupe": "G4"}]}

    folder = puissance_store.find_source_folder(rec, mapping={"clairac_rd_G2": "Clairac_ruePlage_G1"})

    assert folder == "Castillon_Apas_G1"


def test_load_puissance_mapping_reads_yaml(tmp_path):
    path = tmp_path / "puissance_mapping.yaml"
    path.write_text("clairac_rd_G2: Clairac_ruePlage_G1\n", encoding="utf-8")

    mapping = puissance_store.load_puissance_mapping(path)

    assert mapping == {"clairac_rd_G2": "Clairac_ruePlage_G1"}


def test_load_puissance_mapping_missing_file_returns_empty_dict(tmp_path):
    assert puissance_store.load_puissance_mapping(tmp_path / "absent.yaml") == {}


def test_export_puissance_csv_applies_consigne_interpolation_and_priority(tmp_path, monkeypatch):
    source_root = tmp_path / "hydrospot_stream"
    folder = source_root / "Castillon_Apas_G1"
    folder.mkdir(parents=True)
    (folder / "date_power.csv").write_text(
        "Date;Puissance\n"
        "2026-01-01 09:58:00;100.0\n"
        "2026-01-01 09:59:00;0.0\n"
        "2026-01-01 10:00:00;0.0\n"
        "2026-01-01 10:01:00;0.0\n"
        "2026-01-01 10:02:00;0.0\n"
        "2026-01-01 10:03:00;0.0\n"
        "2026-01-01 10:04:00;200.0\n",
        encoding="utf-8",
    )
    (folder / "date_power_fill_nan.csv").write_text(
        "Date;Puissance\n2026-01-01 09:58:00;100.0\n2026-01-01 10:04:00;200.0\n",
        encoding="utf-8",
    )
    (folder / "date_power_fill_nan_neg_price.csv").write_text(
        "Date;Puissance\n2026-01-01 09:58:00;0.0\n2026-01-01 10:04:00;0.0\n",
        encoding="utf-8",
    )

    day_dir = tmp_path / "consignes" / "2026" / "01" / "01"
    day_dir.mkdir(parents=True)
    (day_dir / "Castillon_Apas_G1_10_00.true").write_text(
        "true P 130.0 Consi 130.0 0", encoding="utf-8")
    (day_dir / "Castillon_Apas_G1_10_02.false").write_text("false 0", encoding="utf-8")

    monkeypatch.setattr(puissance_store.config, "PUISSANCE_SOURCE_ROOT", source_root)
    monkeypatch.setattr(puissance_store.config, "CONSIGNES_ROOT", tmp_path / "consignes")

    dest = tmp_path / "puissance.csv"
    n = puissance_store.export_puissance_csv("Castillon_Apas_G1", dest)

    df = pd.read_csv(dest, sep=";")
    assert n == 7
    row = df[df["Date"] == "2026-01-01 10:01"].iloc[0]
    assert row["MA_baisse"] == 150.0
    assert row["power_output"] == 150.0


def test_export_puissance_csv_falls_back_to_priority_when_no_consigne_files(tmp_path, monkeypatch):
    source_root = tmp_path / "hydrospot_stream"
    folder = source_root / "Castillon_Apas_G1"
    folder.mkdir(parents=True)
    (folder / "date_power.csv").write_text(
        "Date;Puissance\n2026-01-01 09:58:00;50.0\n", encoding="utf-8")
    (folder / "date_power_fill_nan.csv").write_text(
        "Date;Puissance\n2026-01-01 09:58:00;50.0\n", encoding="utf-8")
    (folder / "date_power_fill_nan_neg_price.csv").write_text(
        "Date;Puissance\n2026-01-01 09:58:00;0.0\n", encoding="utf-8")

    monkeypatch.setattr(puissance_store.config, "PUISSANCE_SOURCE_ROOT", source_root)
    monkeypatch.setattr(puissance_store.config, "CONSIGNES_ROOT", tmp_path / "no_consignes")

    dest = tmp_path / "puissance.csv"
    puissance_store.export_puissance_csv("Castillon_Apas_G1", dest)

    df = pd.read_csv(dest, sep=";")
    assert df.iloc[0]["MA_baisse"] == 50.0
    assert df.iloc[0]["power_output"] == 50.0


def test_export_puissance_csv_also_writes_hourly_cleaned_file(tmp_path, monkeypatch):
    source_root = tmp_path / "hydrospot_stream"
    folder = source_root / "Castillon_Apas_G1"
    folder.mkdir(parents=True)

    # 2 jours minute par minute, comme les fixtures de test_cleaning.py, pour
    # que les fenêtres glissantes (médiane 24h, écart-type 3h) de la
    # détection de chaos ne dégénèrent pas sur des données trop courtes.
    dates = pd.date_range("2026-01-01 00:00", periods=2 * 24 * 60, freq="1min")

    # Sans fichier de consigne (cf. CONSIGNES_ROOT ci-dessous), la priorité de
    # `export_puissance_csv` retombe sur Puissance_neg_price > 0 -> c'est donc
    # cette colonne qui doit se retrouver dans `power_output`. On y injecte un
    # pic de chaos, même gabarit que test_cleaning.py (baseline 100.0, pic à
    # 500.0 sur 20 minutes), pour exercer la vraie détection/interpolation de
    # `cleaning.py` à travers ce point d'entrée, pas seulement dans ses tests
    # unitaires isolés.
    neg_price_values = np.full(len(dates), 100.0)
    neg_price_values[500:520] = 500.0

    # Puissance et Puissance_fill_nan sont volontairement différentes et
    # constantes, et distinctes de Puissance_neg_price. Si le câblage se
    # trompait de colonne source (ex. `Puissance` au lieu de `power_output`
    # dans `merged[["Date", "power_output"]]`), le fichier horaire refléterait
    # 10.0 (ou 20.0) au lieu de ~100.0 et les assertions ci-dessous
    # échoueraient.
    sources = {
        "date_power.csv": np.full(len(dates), 10.0),
        "date_power_fill_nan.csv": np.full(len(dates), 20.0),
        "date_power_fill_nan_neg_price.csv": neg_price_values,
    }
    for name, values in sources.items():
        pd.DataFrame({"Date": dates, "Puissance": values}).to_csv(folder / name, sep=";", index=False)

    monkeypatch.setattr(puissance_store.config, "PUISSANCE_SOURCE_ROOT", source_root)
    monkeypatch.setattr(puissance_store.config, "CONSIGNES_ROOT", tmp_path / "no_consignes")

    dest = tmp_path / "puissance.csv"
    puissance_store.export_puissance_csv("Castillon_Apas_G1", dest)

    # Étage 1 (puissance.csv, minute par minute) : power_output doit reprendre
    # Puissance_neg_price brut, pic non nettoyé compris - preuve que le pic
    # injecté circule bien jusqu'à power_output avant l'étage de nettoyage.
    minute_df = pd.read_csv(dest, sep=";")
    assert minute_df["power_output"].max() == 500.0
    assert minute_df["power_output"].iloc[0] == 100.0

    horaire_path = tmp_path / "puissance_horaire.csv"
    assert horaire_path.exists()
    df = pd.read_csv(horaire_path, sep=";")
    assert list(df.columns) == ["Date", "power_output"]
    assert len(df) == 48  # 2 jours pleins = 48 heures

    # Étage 2 (puissance_horaire.csv) : le pic de chaos doit avoir été détecté
    # et interpolé par cleaning.py - preuve que la vraie fonction de nettoyage
    # est exercée par ce point d'entrée, pas un pass-through de la valeur
    # brute (qui laisserait apparaître 500.0).
    assert df["power_output"].max() == 100.0
    assert df["power_output"].min() == 100.0
