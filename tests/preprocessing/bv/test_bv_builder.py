from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest

from previ_r2d2.common.config import ROOT
from previ_r2d2.preprocessing.bv import bv_builder
from previ_r2d2.preprocessing.bv.delineation import Physio
from previ_r2d2.preprocessing.bv.rules import load_rules

from ._helpers import write_config_raccordement

import numpy as np
import pandas as pd


class _FakeHubEau:
    def __init__(self, coords: dict, raises: bool = False):
        self._coords = coords
        self._raises = raises

    def stations_referentiel(self, codes):
        if self._raises:
            raise RuntimeError("boom")
        return {c: self._coords[c] for c in codes if c in self._coords}


# Règles réelles déjà calées (centrales/REFERENCE/bv_rules.json) et
# physiographie réelle d'apas_G1_G4 (centrales/REFERENCE/centrales_calibration.json :
# alt_mean=1201 m, surface=2052 km²). alt_mean=1201 tombe dans la bande
# 1000-1300 m documentée comme ambiguë -> kbase_needs_review=True attendu ;
# kc_unit prédit ≈ 0.878 (réel 0.915, écart cohérent avec le MAE=0.034
# documenté et vérifié au Task 5).
FAKE_RULES = load_rules(ROOT / "centrales" / "REFERENCE" / "bv_rules.json")
FAKE_PHYSIO = Physio(surface_km2=2052.3, alt_mean=1201, alt_min=300, alt_max=2400,
                      aspect_mean=190.0, gravelius=1.55, polygon=object())

MEASURE_POLYGON = "previ_r2d2.preprocessing.bv.delineation.measure_from_polygon"
DELINEATE = "previ_r2d2.preprocessing.bv.delineation.delineate_and_measure"
SELECT_POINTS = "previ_r2d2.preprocessing.bv.delineation.select_meteo_points"


def test_load_bv_mapping_reads_yaml(tmp_path):
    path = tmp_path / "bv_mapping.yaml"
    path.write_text("apas_G1_G4: Apas\ncounozouls_G1: null\n", encoding="utf-8")

    mapping = bv_builder.load_bv_mapping(path)

    assert mapping == {"apas_G1_G4": "Apas", "counozouls_G1": None}


def test_load_bv_mapping_missing_file_returns_empty_dict(tmp_path):
    assert bv_builder.load_bv_mapping(tmp_path / "absent.yaml") == {}


def test_resolve_shapefile_path_found(tmp_path):
    shapefiles_dir = tmp_path / "shapefiles"
    shapefiles_dir.mkdir()
    (shapefiles_dir / "BV_Apas.shp").write_bytes(b"")

    path = bv_builder.resolve_shapefile_path("apas_G1_G4", {"apas_G1_G4": "Apas"}, shapefiles_dir)

    assert path == shapefiles_dir / "BV_Apas.shp"


def test_resolve_shapefile_path_no_mapping_entry_returns_none(tmp_path):
    assert bv_builder.resolve_shapefile_path("counozouls_G1", {"counozouls_G1": None}, tmp_path) is None
    assert bv_builder.resolve_shapefile_path("unknown_dossier", {}, tmp_path) is None


def test_resolve_shapefile_path_mapped_but_file_missing_falls_back(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        path = bv_builder.resolve_shapefile_path("apas_G1_G4", {"apas_G1_G4": "Apas"}, tmp_path)

    assert path is None
    assert "introuvable" in caplog.text


def test_build_bv_record_uses_shapefile_when_available(tmp_path):
    with patch(MEASURE_POLYGON, return_value=FAKE_PHYSIO) as mock_measure, \
         patch(DELINEATE) as mock_delineate, \
         patch(SELECT_POINTS, return_value=[(43.1, 0.9)]):
        record = bv_builder.build_bv_record(
            "apas_G1_G4", 43.1312, 0.922689, adresse_alt=288,
            mnt_path=tmp_path / "mnt.tif", rules=FAKE_RULES,
            shapefile_path=tmp_path / "BV_Apas.shp", generated_utc="2026-07-07T08:30:00Z",
        )

    mock_measure.assert_called_once_with(tmp_path / "BV_Apas.shp", tmp_path / "mnt.tif")
    mock_delineate.assert_not_called()
    assert record["centrale"] == "apas_G1_G4"
    assert record["source"] == {"generated_utc": "2026-07-07T08:30:00Z"}
    assert record["exutoire"] == {"lat": 43.1312, "lon": 0.922689}
    assert record["bassin_versant"]["surface_km2"] == 2052.3
    assert record["parametres_calage"]["kc_unit"] == pytest.approx(0.878, abs=0.001)
    assert record["parametres_calage"]["K_base"] == 3.0
    assert record["parametres_calage"]["kbase_needs_review"] is True
    assert record["parametres_calage"]["exposition_needs_review"] is True
    assert "altitude_needs_review" not in record["parametres_calage"]
    assert record["stations_meteo_nwp"] == [{"id": 1, "lat": 43.1, "lon": 0.9}]


def test_build_bv_record_falls_back_to_mnt_delineation_without_shapefile(tmp_path):
    with patch(MEASURE_POLYGON) as mock_measure, \
         patch(DELINEATE, return_value=FAKE_PHYSIO) as mock_delineate, \
         patch(SELECT_POINTS, return_value=[(43.1, 0.9)]):
        record = bv_builder.build_bv_record(
            "counozouls_G1", 42.7176, 2.2292, adresse_alt=None,
            mnt_path=tmp_path / "mnt.tif", rules=FAKE_RULES,
            shapefile_path=None, generated_utc="2026-07-07T08:30:00Z",
        )

    mock_measure.assert_not_called()
    mock_delineate.assert_called_once_with(2.2292, 42.7176, tmp_path / "mnt.tif")
    assert record["centrale"] == "counozouls_G1"


def test_build_bv_record_logs_altitude_mismatch(tmp_path, caplog):
    with patch(MEASURE_POLYGON, return_value=FAKE_PHYSIO), \
         patch(SELECT_POINTS, return_value=[(43.1, 0.9)]), \
         caplog.at_level(logging.WARNING):
        bv_builder.build_bv_record(
            "apas_G1_G4", 43.1312, 0.922689, adresse_alt=900.0,  # |900-300|=600 > 150
            mnt_path=tmp_path / "mnt.tif", rules=FAKE_RULES,
            shapefile_path=tmp_path / "BV_Apas.shp", generated_utc="2026-07-07T08:30:00Z",
        )

    assert "écart" in caplog.text


def test_build_bv_record_no_warning_when_altitude_close_or_unknown(tmp_path, caplog):
    with patch(MEASURE_POLYGON, return_value=FAKE_PHYSIO), \
         patch(SELECT_POINTS, return_value=[(43.1, 0.9)]), \
         caplog.at_level(logging.WARNING):
        bv_builder.build_bv_record(
            "apas_G1_G4", 43.1312, 0.922689, adresse_alt=None,
            mnt_path=tmp_path / "mnt.tif", rules=FAKE_RULES,
            shapefile_path=tmp_path / "BV_Apas.shp", generated_utc="2026-07-07T08:30:00Z",
        )

    assert "écart" not in caplog.text


def test_run_batch_dedupes_shared_physical_site(tmp_path):
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "bonneval_G1", "ae1afc4b", 45.6436, 6.78665, 1035)
    write_config_raccordement(centrales_dir, "bonneval_G2", "ae1afc4b", 45.6436, 6.78665, 1035)
    mapping = {"bonneval_G1": "Bonneval", "bonneval_G2": "Bonneval"}
    shapefiles_dir = tmp_path / "shapefiles"
    shapefiles_dir.mkdir()
    (shapefiles_dir / "BV_Bonneval.shp").write_bytes(b"")

    with patch(MEASURE_POLYGON, return_value=FAKE_PHYSIO) as mock_measure, \
         patch(SELECT_POINTS, return_value=[(45.6, 6.8)]):
        result = bv_builder.run_batch(centrales_dir, tmp_path / "mnt.tif", FAKE_RULES, mapping,
                                       shapefiles_dir, generated_utc="2026-07-07T08:30:00Z")

    assert mock_measure.call_count == 1
    assert (centrales_dir / "bonneval_G1" / "bv.json").exists()
    assert (centrales_dir / "bonneval_G2" / "bv.json").exists()
    assert sorted(result["done"]) == ["bonneval_G1", "bonneval_G2"]


def test_run_batch_skips_group_with_existing_bv_json(tmp_path):
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "apas_G1_G4", "15e59b6d", 43.1312, 0.922689, 288)
    (centrales_dir / "apas_G1_G4" / "bv.json").write_text("{}", encoding="utf-8")

    with patch(MEASURE_POLYGON) as mock_measure, patch(DELINEATE) as mock_delineate:
        result = bv_builder.run_batch(centrales_dir, tmp_path / "mnt.tif", FAKE_RULES, {}, tmp_path)

    mock_measure.assert_not_called()
    mock_delineate.assert_not_called()
    assert result["skipped"] == ["apas_G1_G4"]


def test_run_batch_force_recomputes_existing(tmp_path):
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "apas_G1_G4", "15e59b6d", 43.1312, 0.922689, 288)
    (centrales_dir / "apas_G1_G4" / "bv.json").write_text("{}", encoding="utf-8")

    with patch(DELINEATE, return_value=FAKE_PHYSIO) as mock_delineate, \
         patch(SELECT_POINTS, return_value=[(43.1, 0.9)]):
        result = bv_builder.run_batch(centrales_dir, tmp_path / "mnt.tif", FAKE_RULES, {}, tmp_path,
                                       force=True)

    mock_delineate.assert_called_once()
    assert result["done"] == ["apas_G1_G4"]


def test_run_batch_continues_after_one_centrale_fails(tmp_path):
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "bad_dossier", "uuid-bad", 0.0, 0.0, None)
    write_config_raccordement(centrales_dir, "apas_G1_G4", "15e59b6d", 43.1312, 0.922689, 288)

    def fake_delineate(lon, lat, mnt_path, *args, **kwargs):
        if lon == 0.0:
            raise RuntimeError("Bassin vide")
        return FAKE_PHYSIO

    with patch(DELINEATE, side_effect=fake_delineate), \
         patch(SELECT_POINTS, return_value=[(43.1, 0.9)]):
        result = bv_builder.run_batch(centrales_dir, tmp_path / "mnt.tif", FAKE_RULES, {}, tmp_path)

    assert result["done"] == ["apas_G1_G4"]
    assert result["errors"][0][0] == "uuid-bad"


def test_run_batch_writes_only_missing_dossier_in_partial_group(tmp_path):
    """Un groupe partiellement présent ne doit pas écraser le bv.json déjà là
    (potentiellement corrigé à la main) — seul le dossier manquant est écrit."""
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "bonneval_G1", "ae1afc4b", 45.6436, 6.78665, 1035)
    write_config_raccordement(centrales_dir, "bonneval_G2", "ae1afc4b", 45.6436, 6.78665, 1035)
    (centrales_dir / "bonneval_G1" / "bv.json").write_text(
        json.dumps({"sentinel": "manual-fix"}), encoding="utf-8")
    mapping = {"bonneval_G1": "Bonneval", "bonneval_G2": "Bonneval"}
    shapefiles_dir = tmp_path / "shapefiles"
    shapefiles_dir.mkdir()
    (shapefiles_dir / "BV_Bonneval.shp").write_bytes(b"")

    with patch(MEASURE_POLYGON, return_value=FAKE_PHYSIO), \
         patch(SELECT_POINTS, return_value=[(45.6, 6.8)]):
        result = bv_builder.run_batch(centrales_dir, tmp_path / "mnt.tif", FAKE_RULES, mapping,
                                       shapefiles_dir, generated_utc="2026-07-07T08:30:00Z")

    assert json.loads((centrales_dir / "bonneval_G1" / "bv.json").read_text()) == {"sentinel": "manual-fix"}
    assert (centrales_dir / "bonneval_G2" / "bv.json").exists()
    assert result["done"] == ["bonneval_G2"]
    assert result["skipped"] == ["bonneval_G1"]


def test_run_single_writes_bv_json_via_shapefile(tmp_path):
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "apas_G1_G4", "15e59b6d", 43.1312, 0.922689, 288)
    shapefiles_dir = tmp_path / "shapefiles"
    shapefiles_dir.mkdir()
    (shapefiles_dir / "BV_Apas.shp").write_bytes(b"")

    with patch(MEASURE_POLYGON, return_value=FAKE_PHYSIO) as mock_measure, \
         patch(SELECT_POINTS, return_value=[(43.1, 0.9)]):
        record = bv_builder.run_single(
            centrales_dir, "apas_G1_G4", tmp_path / "mnt.tif", FAKE_RULES,
            {"apas_G1_G4": "Apas"}, shapefiles_dir, generated_utc="2026-07-07T08:30:00Z",
        )

    mock_measure.assert_called_once()
    assert record["centrale"] == "apas_G1_G4"
    written = json.loads((centrales_dir / "apas_G1_G4" / "bv.json").read_text())
    assert written == record


def test_run_single_falls_back_to_mnt_without_mapping(tmp_path):
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "counozouls_G1", "uuid-c", 42.7176, 2.2292, None)

    with patch(MEASURE_POLYGON) as mock_measure, \
         patch(DELINEATE, return_value=FAKE_PHYSIO) as mock_delineate, \
         patch(SELECT_POINTS, return_value=[(42.7, 2.2)]):
        record = bv_builder.run_single(
            centrales_dir, "counozouls_G1", tmp_path / "mnt.tif", FAKE_RULES,
            {}, tmp_path, generated_utc="2026-07-07T08:30:00Z",
        )

    mock_measure.assert_not_called()
    mock_delineate.assert_called_once()
    assert record["centrale"] == "counozouls_G1"


def test_build_bv_record_without_hubeau_client_omits_stations_key(tmp_path):
    with patch(DELINEATE, return_value=FAKE_PHYSIO), patch(SELECT_POINTS, return_value=[]):
        record = bv_builder.build_bv_record(
            "apas_G1_G4", 43.13, 0.92, 288, tmp_path / "mnt.tif", FAKE_RULES,
            shapefile_path=None, generated_utc="2026-01-01T00:00:00Z",
        )

    assert "stations_hydrometriques" not in record


def test_build_bv_record_adds_stations_hydrometriques_without_debit_dir(tmp_path):
    hubeau = _FakeHubEau({
        "O020002001": {"lat": 43.10, "lon": 0.90, "altitude": 250.0},
        "O001531001": {"lat": 43.05, "lon": 0.85, "altitude": None},
    })
    with patch(DELINEATE, return_value=FAKE_PHYSIO), patch(SELECT_POINTS, return_value=[]):
        record = bv_builder.build_bv_record(
            "apas_G1_G4", 43.13, 0.92, 288, tmp_path / "mnt.tif", FAKE_RULES,
            shapefile_path=None, generated_utc="2026-01-01T00:00:00Z",
            station_reference="O020002001", stations_amont=["O001531001"],
            hubeau_client=hubeau,
        )

    assert record["stations_hydrometriques"] == [
        {"code": "O020002001", "role": "reference", "lat": 43.10, "lon": 0.90, "altitude": 250.0},
        {"code": "O001531001", "role": "amont", "lat": 43.05, "lon": 0.85, "altitude": None},
    ]


def test_build_bv_record_omits_key_when_hubeau_fails(tmp_path):
    hubeau = _FakeHubEau({}, raises=True)
    with patch(DELINEATE, return_value=FAKE_PHYSIO), patch(SELECT_POINTS, return_value=[]):
        record = bv_builder.build_bv_record(
            "apas_G1_G4", 43.13, 0.92, 288, tmp_path / "mnt.tif", FAKE_RULES,
            shapefile_path=None, generated_utc="2026-01-01T00:00:00Z",
            station_reference="O020002001", stations_amont=[],
            hubeau_client=hubeau,
        )

    assert "stations_hydrometriques" not in record


def test_build_bv_record_computes_transit_vers_reference_h(tmp_path):
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-07-01", periods=250, freq="h")
    base = rng.normal(size=260)
    amont_vals = base[0:250]
    reference_vals = np.concatenate([rng.normal(size=4), base[0:246]])

    debit_dir = tmp_path / "apas_G1_G4"
    debit_dir.mkdir()
    pd.DataFrame({"Date (TU)": idx, "Valeur (en m³/s)": reference_vals}).to_csv(
        debit_dir / "O020002001.csv", sep=";", index=False)
    pd.DataFrame({"Date (TU)": idx, "Valeur (en m³/s)": amont_vals}).to_csv(
        debit_dir / "amont_O001531001.csv", sep=";", index=False)

    hubeau = _FakeHubEau({
        "O020002001": {"lat": 43.10, "lon": 0.90, "altitude": 250.0},
        "O001531001": {"lat": 43.05, "lon": 0.85, "altitude": None},
    })
    with patch(DELINEATE, return_value=FAKE_PHYSIO), patch(SELECT_POINTS, return_value=[]):
        record = bv_builder.build_bv_record(
            "apas_G1_G4", 43.13, 0.92, 288, tmp_path / "mnt.tif", FAKE_RULES,
            shapefile_path=None, generated_utc="2026-01-01T00:00:00Z",
            station_reference="O020002001", stations_amont=["O001531001"],
            hubeau_client=hubeau, debit_dir=debit_dir,
        )

    amont_entry = next(s for s in record["stations_hydrometriques"] if s["role"] == "amont")
    assert amont_entry["transit_vers_reference_h"] == {"JJA": 4}


def _write_station_fields(centrales_dir, dossier, reference, amont):
    cfg_path = centrales_dir / dossier / "config-raccordement.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["station_vigicrue_reference"] = reference
    cfg["stations_vigicrue_amont"] = amont
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")


def test_run_single_adds_transit_vers_centrale_h(tmp_path):
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "apas_G1_G4", "uuid-1", 43.13, 0.92, 288)
    _write_station_fields(centrales_dir, "apas_G1_G4", "O020002001", [])

    data_dir = tmp_path / "data"
    dossier_dir = data_dir / "apas_G1_G4"
    dossier_dir.mkdir(parents=True)
    rng = np.random.default_rng(2)
    idx = pd.date_range("2025-07-01", periods=250, freq="h")
    base = rng.uniform(20, 100, size=260)
    reference_vals = base[0:250]
    power_vals = np.concatenate([rng.uniform(20, 100, size=2), base[0:248]])
    pd.DataFrame({"Date (TU)": idx, "Valeur (en m³/s)": reference_vals}).to_csv(
        dossier_dir / "O020002001.csv", sep=";", index=False)
    pd.DataFrame({"Date": idx, "power_output": power_vals}).to_csv(
        dossier_dir / "puissance_horaire.csv", sep=";", index=False)

    hubeau = _FakeHubEau({"O020002001": {"lat": 43.10, "lon": 0.90, "altitude": 250.0}})

    with patch(DELINEATE, return_value=FAKE_PHYSIO), patch(SELECT_POINTS, return_value=[]):
        record = bv_builder.run_single(
            centrales_dir, "apas_G1_G4", tmp_path / "mnt.tif", FAKE_RULES, {}, tmp_path / "shapefiles",
            hubeau_client=hubeau, data_dir=data_dir,
        )

    assert record["transit_vers_centrale_h"] == {"JJA": 2}


def test_run_batch_computes_transit_vers_centrale_h_per_dossier(tmp_path):
    """bonneval_G1/G2 partagent le même site physique (centrale_uuid) et donc
    la même station de référence, mais ont chacun leur propre power_output
    (groupes turbines différents) -> transit_vers_centrale_h doit être
    recalculé pour chaque dossier individuellement, pas copié depuis le
    premier dossier traité dans le groupe (régression : `record` au lieu de
    `payload` dans la boucle `for rec in group` de run_batch)."""
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "bonneval_G1", "ae1afc4b", 45.6436, 6.78665, 1035)
    write_config_raccordement(centrales_dir, "bonneval_G2", "ae1afc4b", 45.6436, 6.78665, 1035)
    _write_station_fields(centrales_dir, "bonneval_G1", "O020002001", [])
    _write_station_fields(centrales_dir, "bonneval_G2", "O020002001", [])
    mapping = {"bonneval_G1": "Bonneval", "bonneval_G2": "Bonneval"}
    shapefiles_dir = tmp_path / "shapefiles"
    shapefiles_dir.mkdir()
    (shapefiles_dir / "BV_Bonneval.shp").write_bytes(b"")

    idx = pd.date_range("2025-07-01", periods=250, freq="h")
    rng = np.random.default_rng(2)
    base = rng.uniform(20, 100, size=260)
    reference_vals = base[0:250]  # débit de la station de référence, partagé par les deux dossiers

    data_dir = tmp_path / "data"
    for dossier, shift in (("bonneval_G1", 2), ("bonneval_G2", 5)):
        dossier_dir = data_dir / dossier
        dossier_dir.mkdir(parents=True)
        # Décalage temporel distinct par dossier -> lag de corrélation différent
        # (power_output décalé de `shift` heures par rapport au même débit de référence).
        power_vals = np.concatenate([rng.uniform(20, 100, size=shift), base[0:250 - shift]])
        pd.DataFrame({"Date (TU)": idx, "Valeur (en m³/s)": reference_vals}).to_csv(
            dossier_dir / "O020002001.csv", sep=";", index=False)
        pd.DataFrame({"Date": idx, "power_output": power_vals}).to_csv(
            dossier_dir / "puissance_horaire.csv", sep=";", index=False)

    hubeau = _FakeHubEau({"O020002001": {"lat": 45.6, "lon": 6.8, "altitude": 1000.0}})

    with patch(MEASURE_POLYGON, return_value=FAKE_PHYSIO), \
         patch(SELECT_POINTS, return_value=[(45.6, 6.8)]):
        result = bv_builder.run_batch(centrales_dir, tmp_path / "mnt.tif", FAKE_RULES, mapping,
                                       shapefiles_dir, generated_utc="2026-07-07T08:30:00Z",
                                       hubeau_client=hubeau, data_dir=data_dir)

    assert sorted(result["done"]) == ["bonneval_G1", "bonneval_G2"]
    bv_g1 = json.loads((centrales_dir / "bonneval_G1" / "bv.json").read_text())
    bv_g2 = json.loads((centrales_dir / "bonneval_G2" / "bv.json").read_text())
    assert bv_g1["transit_vers_centrale_h"] == {"JJA": 2}
    assert bv_g2["transit_vers_centrale_h"] == {"JJA": 5}
    assert bv_g1["transit_vers_centrale_h"] != bv_g2["transit_vers_centrale_h"]


def test_run_single_keeps_existing_transit_vers_centrale_h_when_puissance_missing(tmp_path):
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "apas_G1_G4", "uuid-1", 43.13, 0.92, 288)
    _write_station_fields(centrales_dir, "apas_G1_G4", "O020002001", [])
    (centrales_dir / "apas_G1_G4" / "bv.json").write_text(
        json.dumps({"transit_vers_centrale_h": {"JJA": 3}}), encoding="utf-8")

    data_dir = tmp_path / "data"
    (data_dir / "apas_G1_G4").mkdir(parents=True)

    hubeau = _FakeHubEau({"O020002001": {"lat": 43.10, "lon": 0.90, "altitude": 250.0}})

    with patch(DELINEATE, return_value=FAKE_PHYSIO), patch(SELECT_POINTS, return_value=[]):
        record = bv_builder.run_single(
            centrales_dir, "apas_G1_G4", tmp_path / "mnt.tif", FAKE_RULES, {}, tmp_path / "shapefiles",
            hubeau_client=hubeau, data_dir=data_dir,
        )

    assert record["transit_vers_centrale_h"] == {"JJA": 3}


def test_run_single_prefers_direct_correlation_over_geometric_when_both_available(tmp_path):
    """Quand la corrélation directe référence -> puissance est fiable (>= 0.9)
    ET qu'un repli géométrique est aussi calculable (station amont avec
    transit_vers_reference_h), la valeur directe doit l'emporter -- pas le
    repli géométrique. Construction : lag direct = 2 (reprise exacte de
    `test_run_single_adds_transit_vers_centrale_h`, seed=2, corr≈0.9999) ;
    station amont/distances choisies pour que le géométrique donnerait 3.2 si
    (par régression) il était utilisé à la place -- valeur bien distincte de 2,
    ce qui prouve que le test discrimine vraiment les deux chemins."""
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "apas_G1_G4", "uuid-1", 43.13, 0.92, 288)
    _write_station_fields(centrales_dir, "apas_G1_G4", "O020002001", ["O001531001"])

    data_dir = tmp_path / "data"
    dossier_dir = data_dir / "apas_G1_G4"
    dossier_dir.mkdir(parents=True)
    rng = np.random.default_rng(2)
    idx = pd.date_range("2025-07-01", periods=250, freq="h")
    base = rng.uniform(20, 100, size=260)
    reference_vals = base[0:250]
    power_vals = np.concatenate([rng.uniform(20, 100, size=2), base[0:248]])
    # Station amont menant la référence de 6 h (copie décalée exacte -> lag
    # amont->référence sans ambiguïté), utilisée uniquement pour le repli
    # géométrique (transit_vers_reference_h).
    amont_lag = 6
    rng_amont = np.random.default_rng(42)
    amont_vals = np.concatenate([reference_vals[amont_lag:], rng_amont.uniform(20, 100, size=amont_lag)])

    pd.DataFrame({"Date (TU)": idx, "Valeur (en m³/s)": reference_vals}).to_csv(
        dossier_dir / "O020002001.csv", sep=";", index=False)
    pd.DataFrame({"Date (TU)": idx, "Valeur (en m³/s)": amont_vals}).to_csv(
        dossier_dir / "amont_O001531001.csv", sep=";", index=False)
    pd.DataFrame({"Date": idx, "power_output": power_vals}).to_csv(
        dossier_dir / "puissance_horaire.csv", sep=";", index=False)

    # Distances : d(référence, centrale) / d(amont, référence) ≈ 0.5388 ;
    # avec le lag amont->référence de 6 h, le géométrique donnerait
    # 6 * 0.5388 ≈ 3.2 -- distinct du lag direct (2).
    hubeau = _FakeHubEau({
        "O020002001": {"lat": 43.10, "lon": 0.90, "altitude": 250.0},
        "O001531001": {"lat": 43.05, "lon": 0.85, "altitude": None},
    })

    with patch(DELINEATE, return_value=FAKE_PHYSIO), patch(SELECT_POINTS, return_value=[]):
        record = bv_builder.run_single(
            centrales_dir, "apas_G1_G4", tmp_path / "mnt.tif", FAKE_RULES, {}, tmp_path / "shapefiles",
            hubeau_client=hubeau, data_dir=data_dir,
        )

    amont_entry = next(s for s in record["stations_hydrometriques"] if s["role"] == "amont")
    assert amont_entry["transit_vers_reference_h"] == {"JJA": amont_lag}  # géométrique bien calculable
    assert record["transit_vers_centrale_h"] == {"JJA": 2}  # valeur directe, pas 3.2 (géométrique)


def test_run_single_falls_back_to_geometric_transit_when_correlation_unreliable(tmp_path):
    centrales_dir = tmp_path / "centrales"
    write_config_raccordement(centrales_dir, "apas_G1_G4", "uuid-1", 43.1312, 0.922689, 288)
    _write_station_fields(centrales_dir, "apas_G1_G4", "O020002001", ["O001531001"])

    data_dir = tmp_path / "data"
    dossier_dir = data_dir / "apas_G1_G4"
    dossier_dir.mkdir(parents=True)

    idx = pd.date_range("2025-07-01", periods=250, freq="h")
    rng = np.random.default_rng(3)
    reference_vals = rng.uniform(20, 100, size=250)
    power_vals = rng.uniform(20, 100, size=250)  # non corrélé -> sous le seuil 0.9
    amont_vals = rng.uniform(20, 100, size=250)

    pd.DataFrame({"Date (TU)": idx, "Valeur (en m³/s)": reference_vals}).to_csv(
        dossier_dir / "O020002001.csv", sep=";", index=False)
    pd.DataFrame({"Date (TU)": idx, "Valeur (en m³/s)": amont_vals}).to_csv(
        dossier_dir / "amont_O001531001.csv", sep=";", index=False)
    pd.DataFrame({"Date": idx, "power_output": power_vals}).to_csv(
        dossier_dir / "puissance_horaire.csv", sep=";", index=False)

    hubeau = _FakeHubEau({
        "O020002001": {"lat": 43.097995649, "lon": 0.706150546, "altitude": 357.0},
        "O001531001": {"lat": 42.867149436, "lon": 0.747808596, "altitude": 552.0},
    })

    with patch(DELINEATE, return_value=FAKE_PHYSIO), patch(SELECT_POINTS, return_value=[]):
        record = bv_builder.run_single(
            centrales_dir, "apas_G1_G4", tmp_path / "mnt.tif", FAKE_RULES, {}, tmp_path / "shapefiles",
            hubeau_client=hubeau, data_dir=data_dir,
        )

    assert "transit_vers_centrale_h" in record
    assert record["transit_vers_centrale_h"]  # non vide : repli géométrique appliqué
