from __future__ import annotations

import pytest

from previ_r2d2.preprocessing.puissance import puissance_store


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
