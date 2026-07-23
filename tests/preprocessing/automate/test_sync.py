from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest

from previ_r2d2.preprocessing.automate.sync import apply_correction, sync_variable


def test_apply_correction_multiplies_and_offsets(tmp_path):
    f = tmp_path / "12.txt"
    f.write_text("12:00:00  16.73\n12:01:00  16.80\n")

    corrected = apply_correction(f, multiply=10, offset=0)

    assert corrected is True
    assert f.read_text() == "12:00:00  167.3\n12:01:00  168\n"


def test_apply_correction_skips_old_semicolon_format(tmp_path):
    f = tmp_path / "12.txt"
    original = "12/07/2026;12:00:00;16,73; ;\n"
    f.write_text(original)

    corrected = apply_correction(f, multiply=10, offset=0)

    assert corrected is False
    assert f.read_text() == original


def test_apply_correction_applies_offset(tmp_path):
    f = tmp_path / "12.txt"
    f.write_text("12:00:00  100\n")

    apply_correction(f, multiply=1, offset=500)

    assert f.read_text() == "12:00:00  600\n"


def test_sync_variable_renames_extensionless_files_and_corrects_only_new_ones(tmp_path):
    # sync_variable ne scanne que le jour courant (renommage) -- une date figée
    # casserait ce test dès que "aujourd'hui" ne matche plus la date codée en dur.
    today = datetime.date.today()
    dest_dir = tmp_path / "Pression"
    day_dir = dest_dir / f"{today:%Y}" / f"{today:%m}" / f"{today:%d}"
    day_dir.mkdir(parents=True)
    # Fichier déjà présent d'un run précédent, déjà renommé + déjà corrigé.
    (day_dir / "10.txt").write_text("10:00:00  50\n")

    def fake_rsync(cmd, **kwargs):
        # Simule rsync : dépose un nouveau fichier SANS extension (comme l'automate le fait).
        (day_dir / "11").write_text("11:00:00  5\n")
        result = type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return result

    with patch("previ_r2d2.preprocessing.automate.sync.subprocess.run", side_effect=fake_rsync):
        downloaded, corrected_files = sync_variable(
            var="Pression",
            source_host="user@host",
            source_base="/tmp/user/mes",
            dest_dir=dest_dir,
            corrections={"multiply": 10, "offset": 0},
        )

    assert (day_dir / "11.txt").exists()  # renommé
    assert not (day_dir / "11").exists()
    assert (day_dir / "11.txt").read_text() == "11:00:00  50\n"  # corrigé (nouveau fichier)
    assert (day_dir / "10.txt").read_text() == "10:00:00  50\n"  # PAS re-corrigé (fichier ancien)
    assert corrected_files == [day_dir / "11.txt"]


def test_sync_variable_uses_source_var_for_rsync_path_when_different_from_dest(tmp_path):
    # Cas Melles : dossier distant (nomenclature hydrospot, ex. "PosInj1") différent
    # du dossier destination local (nomenclature TokAPI, "ML2_POS_STAB_INJEC1_G1").
    dest_dir = tmp_path / "ML2_POS_STAB_INJEC1_G1"
    captured_cmd = {}

    def fake_rsync(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with patch("previ_r2d2.preprocessing.automate.sync.subprocess.run", side_effect=fake_rsync):
        sync_variable(
            var="ML2_POS_STAB_INJEC1_G1",
            source_var="PosInj1",
            source_host="user@host",
            source_base="/tmp/user/mes",
            dest_dir=dest_dir,
            corrections=None,
        )

    assert captured_cmd["cmd"][-2] == "user@host:/tmp/user/mes/PosInj1/"


def test_sync_variable_raises_on_rsync_failure(tmp_path):
    def fake_rsync(cmd, **kwargs):
        return type("Result", (), {"returncode": 23, "stdout": "", "stderr": "some error"})()

    with patch("previ_r2d2.preprocessing.automate.sync.subprocess.run", side_effect=fake_rsync):
        with pytest.raises(RuntimeError, match="some error"):
            sync_variable(
                var="Pression",
                source_host="user@host",
                source_base="/tmp/user/mes",
                dest_dir=tmp_path / "Pression",
                corrections=None,
            )
