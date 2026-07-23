from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from previ_r2d2.preprocessing.debit.hubeau import HubEauClient, HubEauError


def _fake_response(status_code: int, json_data: dict):
    resp = MagicMock()
    resp.status_code = status_code
    resp.reason = "Erreur"
    resp.json.return_value = json_data
    return resp


def test_stations_referentiel_returns_coords_by_code():
    client = HubEauClient()
    response = _fake_response(200, {
        "data": [
            {"code_station": "O020002001", "latitude_station": 43.10,
             "longitude_station": 0.90, "altitude_ref_alti_station": 250.0},
            {"code_station": "O001531001", "latitude_station": 43.05,
             "longitude_station": 0.85, "altitude_ref_alti_station": None},
        ]
    })

    with patch.object(client._session, "get", return_value=response) as mock_get:
        result = client.stations_referentiel(["O020002001", "O001531001"])

    assert result == {
        "O020002001": {"lat": 43.10, "lon": 0.90, "altitude": 250.0},
        "O001531001": {"lat": 43.05, "lon": 0.85, "altitude": None},
    }
    assert mock_get.call_args.kwargs["params"]["code_station"] == "O020002001,O001531001"


def test_stations_referentiel_omits_codes_absent_from_response():
    client = HubEauClient()
    response = _fake_response(200, {"data": []})

    with patch.object(client._session, "get", return_value=response):
        result = client.stations_referentiel(["O999999999"])

    assert result == {}


def test_stations_referentiel_raises_on_http_error():
    client = HubEauClient()
    response = _fake_response(500, {})

    with patch.object(client._session, "get", return_value=response):
        with pytest.raises(HubEauError):
            client.stations_referentiel(["O020002001"])
