from __future__ import annotations

import logging

from previ_r2d2.preprocessing.bv import bv_builder

from ._helpers import write_config_raccordement


def test_discover_centrale_records_reads_all_config_raccordement(tmp_path):
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "apas_G1_G4", "15e59b6d", 43.1312, 0.922689, 288)
    write_config_raccordement(centrales_dir, "touzac_g2_G2", "uuid-touzac", 44.4667, 1.0833, 60)
    (centrales_dir / "config-general.json").write_text("[]", encoding="utf-8")

    records = bv_builder.discover_centrale_records(centrales_dir)

    assert sorted(r["dossier"] for r in records) == ["apas_G1_G4", "touzac_g2_G2"]


def test_discover_centrale_records_skips_malformed_json_without_aborting(tmp_path, caplog):
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "apas_G1_G4", "15e59b6d", 43.1312, 0.922689, 288)
    bad_dir = centrales_dir / "bad_dossier"
    bad_dir.mkdir()
    (bad_dir / "config-raccordement.json").write_text("{not valid json", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        records = bv_builder.discover_centrale_records(centrales_dir)

    assert [r["dossier"] for r in records] == ["apas_G1_G4"]
    assert "bad_dossier" in caplog.text


def test_group_by_site_groups_shared_centrale_uuid():
    records = [
        {"dossier": "bonneval_G1", "centrale_uuid": "ae1afc4b",
         "adresse_lat": 45.6436, "adresse_lng": 6.78665, "adresse_alt": 1035},
        {"dossier": "bonneval_G2", "centrale_uuid": "ae1afc4b",
         "adresse_lat": 45.6436, "adresse_lng": 6.78665, "adresse_alt": 1035},
        {"dossier": "apas_G1_G4", "centrale_uuid": "15e59b6d",
         "adresse_lat": 43.1312, "adresse_lng": 0.922689, "adresse_alt": 288},
    ]

    groups = bv_builder.group_by_site(records)

    assert groups["ae1afc4b"] == records[:2]
    assert groups["15e59b6d"] == records[2:]
