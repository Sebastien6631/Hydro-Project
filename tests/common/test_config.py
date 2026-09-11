from __future__ import annotations

from projet_hydro.common import config


def test_centrales_dir_is_root_slash_centrales():
    assert config.CENTRALES_DIR == config.ROOT / "centrales"


def test_reference_dir_is_centrales_dir_slash_reference():
    assert config.REFERENCE_DIR == config.CENTRALES_DIR / "REFERENCE"


def test_models_dir_is_under_root():
    assert config.MODELS_DIR == config.ROOT / "models"


def test_archive_root_is_under_project_root():
    assert config.ARCHIVE_ROOT == config.ROOT / "ARCHIVE"


def test_nas_data_root_is_centrales_dir():
    assert config.NAS_DATA_ROOT == config.CENTRALES_DIR
