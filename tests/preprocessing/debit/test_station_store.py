from __future__ import annotations

from projet_hydro.common import config
from projet_hydro.preprocessing.debit import station_store


def test_resolve_nas_path_writes_index_inside_monkeypatched_nas_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NAS_DATA_ROOT", tmp_path)

    station_store.resolve_nas_path("TEST_CODE_NE_PAS_UTILISER", "dossier_test", "reference")

    assert (tmp_path / "_index.json").exists()
