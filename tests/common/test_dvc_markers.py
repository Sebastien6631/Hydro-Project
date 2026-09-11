from __future__ import annotations

import json

from projet_hydro.common import dvc_markers


def test_write_creates_marker_file_with_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "dvc_markers")

    dvc_markers.write("debit")

    path = tmp_path / "dvc_markers" / "debit.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "last_run" in data


def test_write_overwrites_previous_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "dvc_markers")

    dvc_markers.write("debit")
    first = (tmp_path / "dvc_markers" / "debit.json").read_text(encoding="utf-8")
    dvc_markers.write("debit")
    second = (tmp_path / "dvc_markers" / "debit.json").read_text(encoding="utf-8")

    assert json.loads(first)["last_run"] <= json.loads(second)["last_run"]
