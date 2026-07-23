from __future__ import annotations

import pytest

from previ_r2d2.preprocessing.onboarding import validation


def make_complete_rec(dossier="apas_G1_G4", flex_strategy="DEFAULT"):
    return {
        "dossier": dossier,
        "flex_strategy": flex_strategy,
        "facteur_debit": 1.0,
        "station_vigicrue_reference": "O1234567",
        "stations_vigicrue_amont": ["O7654321"],
        "groupes": [{
            "nom_groupe": "G1",
            "priorite": 1,
            "debit_armement_turbine": 0.5,
            "debit_max": 10.0,
            "rendement": 0.85,
            "chute_disponible_polynome": [[0, 1, 2]],
        }],
    }


def test_missing_fields_empty_for_complete_default_record(tmp_path, monkeypatch):
    from previ_r2d2.preprocessing.puissance import puissance_store

    monkeypatch.setattr(puissance_store.config, "PUISSANCE_SOURCE_ROOT", tmp_path)
    (tmp_path / "Apas_source").mkdir()
    mapping_path = tmp_path / "puissance_mapping.yaml"
    mapping_path.write_text("apas_G1_G4: Apas_source\n", encoding="utf-8")

    rec = make_complete_rec()

    assert validation.missing_fields(rec, mapping_path=mapping_path) == []


def test_missing_fields_flags_empty_flex_strategy(tmp_path):
    rec = make_complete_rec(flex_strategy="")

    missing = validation.missing_fields(rec, mapping_path=tmp_path / "no_mapping.yaml")

    assert "flex_strategy" in missing


def test_missing_fields_flags_missing_station_for_default(tmp_path):
    rec = make_complete_rec()
    del rec["station_vigicrue_reference"]

    missing = validation.missing_fields(rec, mapping_path=tmp_path / "no_mapping.yaml")

    assert "station_vigicrue_reference" in missing


def test_missing_fields_does_not_require_station_for_haute_chute(tmp_path):
    rec = make_complete_rec(flex_strategy="HAUTE_CHUTE")
    del rec["station_vigicrue_reference"]
    automate_sync_path = tmp_path / "automate_sync.yaml"
    automate_sync_path.write_text("apas_G1_G4: {}\n", encoding="utf-8")

    missing = validation.missing_fields(
        rec, mapping_path=tmp_path / "no_mapping.yaml", automate_sync_path=automate_sync_path
    )

    assert "station_vigicrue_reference" not in missing


def test_missing_fields_does_not_require_amont_for_haute_chute(tmp_path):
    rec = make_complete_rec(flex_strategy="HAUTE_CHUTE")
    del rec["stations_vigicrue_amont"]
    automate_sync_path = tmp_path / "automate_sync.yaml"
    automate_sync_path.write_text("apas_G1_G4: {}\n", encoding="utf-8")

    missing = validation.missing_fields(
        rec, mapping_path=tmp_path / "no_mapping.yaml", automate_sync_path=automate_sync_path
    )

    assert "stations_vigicrue_amont" not in missing


def test_missing_fields_flags_missing_amont_for_default(tmp_path):
    rec = make_complete_rec()
    del rec["stations_vigicrue_amont"]

    missing = validation.missing_fields(rec, mapping_path=tmp_path / "no_mapping.yaml")

    assert "stations_vigicrue_amont" in missing


def test_missing_fields_flags_empty_amont_for_default(tmp_path):
    rec = make_complete_rec()
    rec["stations_vigicrue_amont"] = []

    missing = validation.missing_fields(rec, mapping_path=tmp_path / "no_mapping.yaml")

    assert "stations_vigicrue_amont" in missing


def test_missing_fields_flags_missing_automate_sync_for_haute_chute(tmp_path):
    rec = make_complete_rec(flex_strategy="HAUTE_CHUTE")
    automate_sync_path = tmp_path / "automate_sync.yaml"
    automate_sync_path.write_text("autre_dossier: {}\n", encoding="utf-8")

    missing = validation.missing_fields(
        rec, mapping_path=tmp_path / "no_mapping.yaml", automate_sync_path=automate_sync_path
    )

    assert any("automate_sync" in m for m in missing)


def test_missing_fields_empty_when_automate_sync_present_for_haute_chute(tmp_path, monkeypatch):
    from previ_r2d2.preprocessing.puissance import puissance_store

    monkeypatch.setattr(puissance_store.config, "PUISSANCE_SOURCE_ROOT", tmp_path)
    (tmp_path / "Apas_source").mkdir()
    mapping_path = tmp_path / "puissance_mapping.yaml"
    mapping_path.write_text("apas_G1_G4: Apas_source\n", encoding="utf-8")
    automate_sync_path = tmp_path / "automate_sync.yaml"
    automate_sync_path.write_text("apas_G1_G4: {}\n", encoding="utf-8")

    rec = make_complete_rec(flex_strategy="HAUTE_CHUTE")
    del rec["station_vigicrue_reference"]
    del rec["stations_vigicrue_amont"]

    missing = validation.missing_fields(
        rec, mapping_path=mapping_path, automate_sync_path=automate_sync_path
    )

    assert missing == []


def test_missing_fields_flags_incomplete_groupe(tmp_path):
    rec = make_complete_rec()
    del rec["groupes"][0]["rendement"]

    missing = validation.missing_fields(rec, mapping_path=tmp_path / "no_mapping.yaml")

    assert any("rendement" in m for m in missing)


def test_missing_fields_flags_no_groupes(tmp_path):
    rec = make_complete_rec()
    rec["groupes"] = []

    missing = validation.missing_fields(rec, mapping_path=tmp_path / "no_mapping.yaml")

    assert "groupes" in missing


def test_missing_fields_flags_unresolved_puissance_mapping(tmp_path, monkeypatch):
    from previ_r2d2.preprocessing.puissance import puissance_store

    monkeypatch.setattr(puissance_store.config, "PUISSANCE_SOURCE_ROOT", tmp_path)
    rec = make_complete_rec(dossier="dossier_sans_source")

    missing = validation.missing_fields(rec, mapping_path=tmp_path / "no_mapping.yaml")

    assert any("puissance_mapping" in m for m in missing)
