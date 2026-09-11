from __future__ import annotations

import pytest

from projet_hydro.model.pipeline.bv_config import (
    bv_params_from_bv_json,
    transit_amont_from_bv_json,
    transit_centrale_from_bv_json,
)


def make_bv_json(stations_hydrometriques=None):
    return {
        "bassin_versant": {"altitude_moyenne_m": 850.0, "surface_km2": 120.5},
        "parametres_calage": {"K_base": 0.9, "exposition": 1.1, "kc_unit": 1.0},
        "stations_hydrometriques": stations_hydrometriques or [],
    }


def test_bv_params_from_bv_json_maps_nested_to_flat():
    bv_json = make_bv_json()

    result = bv_params_from_bv_json(bv_json)

    assert result == {
        "altitude_bv": 850.0,
        "surface_km2": 120.5,
        "k_base": 0.9,
        "exposition": 1.1,
        "kc_unit": 1.0,
    }


def test_bv_params_from_bv_json_raises_key_error_when_field_missing():
    with pytest.raises(KeyError):
        bv_params_from_bv_json({"bassin_versant": {}})


def test_transit_amont_single_amont_station_uses_debit_amont_key():
    bv_json = make_bv_json([
        {"code": "REF01", "role": "reference"},
        {"code": "AMT01", "role": "amont", "transit_vers_reference_h": {"DJF": 5, "MAM": 3, "JJA": 2, "SON": 4}},
    ])

    result = transit_amont_from_bv_json(bv_json)

    assert result == {"debit_amont": {"hiver": 5, "printemps": 3, "ete": 2, "automne": 4}}


def test_transit_amont_multiple_amont_stations_use_coded_keys():
    bv_json = make_bv_json([
        {"code": "REF01", "role": "reference"},
        {"code": "AMT01", "role": "amont", "transit_vers_reference_h": {"DJF": 5, "MAM": 3, "JJA": 2, "SON": 4}},
        {"code": "AMT02", "role": "amont", "transit_vers_reference_h": {"DJF": 10, "MAM": 8, "JJA": 6, "SON": 9}},
    ])

    result = transit_amont_from_bv_json(bv_json)

    assert result == {
        "debit_amont_AMT01": {"hiver": 5, "printemps": 3, "ete": 2, "automne": 4},
        "debit_amont_AMT02": {"hiver": 10, "printemps": 8, "ete": 6, "automne": 9},
    }


def test_transit_amont_no_amont_station_returns_empty_dict():
    bv_json = make_bv_json([{"code": "REF01", "role": "reference"}])

    result = transit_amont_from_bv_json(bv_json)

    assert result == {}


def test_transit_amont_station_without_transit_data_gets_empty_seasons():
    bv_json = make_bv_json([{"code": "AMT01", "role": "amont"}])

    result = transit_amont_from_bv_json(bv_json)

    assert result == {"debit_amont": {}}


def test_transit_centrale_from_bv_json_translates_season_keys():
    bv_json = {"transit_vers_centrale_h": {"DJF": 5.1, "MAM": 5.9, "JJA": 4.9, "SON": 4.5}}

    result = transit_centrale_from_bv_json(bv_json)

    assert result == {"hiver": 5.1, "printemps": 5.9, "ete": 4.9, "automne": 4.5}


def test_transit_centrale_from_bv_json_empty_when_key_absent():
    assert transit_centrale_from_bv_json({}) == {}
