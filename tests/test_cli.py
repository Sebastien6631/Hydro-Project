from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from previ_r2d2 import cli
from previ_r2d2.common import config


def make_centrales_dir(tmp_path, dossiers):
    for name in dossiers:
        (tmp_path / name).mkdir()
        (tmp_path / name / "bv.json").write_text(json.dumps({"exutoire": {"lat": 1.0, "lon": 2.0}}))
    return tmp_path


def test_parse_args_requires_train_or_predict():
    with pytest.raises(SystemExit):
        cli.parse_args([])


def test_parse_args_requires_dossier_or_all_dossiers():
    with pytest.raises(SystemExit):
        cli.parse_args(["--train"])


def test_parse_args_requires_horizon_with_dossier():
    with pytest.raises(SystemExit):
        cli.parse_args(["--train", "--dossier", "nancy"])


def test_parse_args_nominal_single_dossier():
    args = cli.parse_args(["--train", "--dossier", "nancy", "--horizon", "8"])

    assert args.dossier == "nancy"
    assert args.horizon == 8
    assert args.meta == "ridge"
    assert args.epochs == 40
    assert args.n_trials_lgbm == 30
    assert args.n_trials_final == 50
    assert args.force_lgbm is False
    assert args.force_lstm is False


def test_parse_args_all_dossiers_does_not_require_horizon():
    args = cli.parse_args(["--train", "--all-dossiers"])

    assert args.all_dossiers is True
    assert args.dossier is None


def test_discover_dossiers_finds_only_dirs_with_bv_json(tmp_path, monkeypatch):
    make_centrales_dir(tmp_path, ["dossierA", "dossierB"])
    (tmp_path / "dossierC_no_bv").mkdir()
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)

    assert cli.discover_dossiers() == ["dossierA", "dossierB"]


def test_load_bv_json_returns_parsed_dict(tmp_path, monkeypatch):
    make_centrales_dir(tmp_path, ["dossierA"])
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)

    bv_json = cli.load_bv_json("dossierA")

    assert bv_json["exutoire"] == {"lat": 1.0, "lon": 2.0}


def test_main_predict_returns_1_without_calling_train(monkeypatch):
    with patch.object(cli, "train_command") as mock_train:
        rc = cli.main(["--predict", "--dossier", "x", "--horizon", "8"])

    assert rc == 1
    mock_train.assert_not_called()


def test_train_command_single_dossier_calls_run_training_once(tmp_path, monkeypatch):
    make_centrales_dir(tmp_path, ["nancy"])
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    args = cli.parse_args(["--train", "--dossier", "nancy", "--horizon", "8"])

    with patch.object(cli, "run_training") as mock_run:
        rc = cli.train_command(args)

    assert rc == 0
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0] == "nancy"
    assert call_args[0][1] == 8
    assert call_args[0][2] == {"lat": 1.0, "lon": 2.0}


def test_train_command_all_dossiers_tries_every_pair_and_continues_after_error(tmp_path, monkeypatch):
    make_centrales_dir(tmp_path, ["d1", "d2", "d3"])
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    monkeypatch.setattr(cli, "HORIZONS", [8])
    args = cli.parse_args(["--train", "--all-dossiers"])

    with patch.object(cli, "run_training") as mock_run:
        mock_run.side_effect = [None, RuntimeError("boom"), None]
        rc = cli.train_command(args)

    assert rc == 1
    assert mock_run.call_count == 3


def test_main_predict_calls_predict_command_not_train(monkeypatch):
    with patch.object(cli, "predict_command") as mock_predict, patch.object(cli, "train_command") as mock_train:
        cli.main(["--predict", "--dossier", "nancy", "--horizon", "8"])

    mock_predict.assert_called_once()
    mock_train.assert_not_called()


def test_predict_command_single_dossier_calls_run_prediction_once(tmp_path, monkeypatch):
    make_centrales_dir(tmp_path, ["nancy"])
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    args = cli.parse_args(["--predict", "--dossier", "nancy", "--horizon", "8"])

    with patch.object(cli, "run_prediction") as mock_run:
        rc = cli.predict_command(args)

    assert rc == 0
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0] == "nancy"
    assert call_args[0][1] == 8


def test_predict_command_all_dossiers_tries_every_pair_and_continues_after_error(tmp_path, monkeypatch):
    make_centrales_dir(tmp_path, ["d1", "d2", "d3"])
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    monkeypatch.setattr(cli, "HORIZONS", [8])
    args = cli.parse_args(["--predict", "--all-dossiers"])

    with patch.object(cli, "run_prediction") as mock_run:
        mock_run.side_effect = [None, RuntimeError("boom"), None]
        rc = cli.predict_command(args)

    assert rc == 1
    assert mock_run.call_count == 3
