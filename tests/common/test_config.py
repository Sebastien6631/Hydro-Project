from __future__ import annotations

from previ_r2d2.common import config


def test_centrales_dir_is_root_slash_centrales():
    assert config.CENTRALES_DIR == config.ROOT / "centrales"


def test_reference_dir_is_centrales_dir_slash_reference():
    assert config.REFERENCE_DIR == config.CENTRALES_DIR / "REFERENCE"


def test_models_dir_is_under_root():
    assert config.MODELS_DIR == config.ROOT / "models"


def test_nas_archive_root_is_under_nas_data_root():
    assert config.NAS_ARCHIVE_ROOT == config.NAS_DATA_ROOT / "ARCHIVE"
