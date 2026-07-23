from __future__ import annotations

from previ_r2d2.preprocessing.meteo.retention import clean_year, files_to_clean


def _touch(path, size=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_files_to_clean_excludes_today(tmp_path):
    annee_dir = tmp_path / "2026"
    today_file = annee_dir / "07" / "09" / "BARTHE_ENR_EC_OP_recent_2026070900_048.csv"
    past_file = annee_dir / "07" / "08" / "BARTHE_ENR_EC_OP_recent_2026070800_048.csv"
    _touch(today_file)
    _touch(past_file)

    result = files_to_clean(annee_dir, today="20260709")

    assert result == [past_file]


def test_files_to_clean_keeps_short_lead_time_files(tmp_path):
    annee_dir = tmp_path / "2026"
    short_lead = annee_dir / "07" / "08" / "BARTHE_ENR_EC_OP_recent_2026070800_012.csv"
    long_lead = annee_dir / "07" / "08" / "BARTHE_ENR_EC_OP_recent_2026070800_048.csv"
    _touch(short_lead)
    _touch(long_lead)

    result = files_to_clean(annee_dir, today="20260709")

    assert result == [long_lead]


def test_files_to_clean_ignores_non_matching_filenames(tmp_path):
    annee_dir = tmp_path / "2026"
    other_file = annee_dir / "07" / "08" / "autre_fichier.csv"
    _touch(other_file)

    result = files_to_clean(annee_dir, today="20260709")

    assert result == []


def test_clean_year_dry_run_does_not_delete(tmp_path):
    annee_dir = tmp_path / "2026"
    target = annee_dir / "07" / "08" / "BARTHE_ENR_EC_OP_recent_2026070800_048.csv"
    _touch(target, size=100)

    targets, total_size = clean_year(annee_dir, today="20260709", dry_run=True)

    assert targets == [target]
    assert total_size == 100
    assert target.exists()


def test_clean_year_deletes_matching_files(tmp_path):
    annee_dir = tmp_path / "2026"
    target = annee_dir / "07" / "08" / "BARTHE_ENR_EC_OP_recent_2026070800_048.csv"
    kept = annee_dir / "07" / "08" / "BARTHE_ENR_EC_OP_recent_2026070800_012.csv"
    _touch(target, size=50)
    _touch(kept)

    targets, total_size = clean_year(annee_dir, today="20260709", dry_run=False)

    assert targets == [target]
    assert total_size == 50
    assert not target.exists()
    assert kept.exists()
