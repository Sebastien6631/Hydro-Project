from __future__ import annotations

import ftplib

import pytest

from previ_r2d2.preprocessing.meteo.nwp_ftp import download_ftp_dir


class _FakeFTP:
    """Simule un FTP minimal sur un arbre en mémoire (dirs = dict imbriqué,
    fichiers = valeurs bytes), pour tester download_ftp_dir sans réseau."""

    def __init__(self, tree, unreadable=frozenset()):
        self._tree = tree
        self._path: list[str] = []
        self._unreadable = unreadable

    def _resolve(self, path: str):
        node = self._tree
        for part in [p for p in path.split("/") if p]:
            node = node[part]
        return node

    def cwd(self, path: str) -> None:
        if path == "..":
            self._path.pop()
            return
        target = path if path.startswith("/") else "/".join([*self._path, path])
        if target in self._unreadable:
            raise ftplib.error_perm("550 Dossier inaccessible")
        node = self._resolve(target)
        if not isinstance(node, dict):
            raise ftplib.error_perm("550 Not a directory")
        self._path = [p for p in target.split("/") if p]

    def nlst(self) -> list[str]:
        return list(self._resolve("/".join(self._path)).keys())

    def retrbinary(self, cmd: str, callback) -> None:
        name = cmd.split(" ", 1)[1]
        node = self._resolve("/".join(self._path))
        callback(node[name])


def test_download_ftp_dir_downloads_files_recursively(tmp_path):
    tree = {
        "2026": {
            "07": {
                "09": {
                    "file1.csv": b"contenu1",
                    "sub": {"file2.csv": b"contenu2"},
                }
            }
        }
    }
    ftp = _FakeFTP(tree)
    local_dir = tmp_path / "out"

    downloaded, errors = download_ftp_dir(ftp, "/2026/07/09", str(local_dir))

    assert downloaded == 2
    assert errors == []
    assert (local_dir / "file1.csv").read_bytes() == b"contenu1"
    assert (local_dir / "sub" / "file2.csv").read_bytes() == b"contenu2"


def test_download_ftp_dir_reports_error_when_remote_dir_unreadable(tmp_path):
    tree = {"2026": {"07": {"09": {}}}}
    ftp = _FakeFTP(tree, unreadable={"/2026/07/09/absent"})
    local_dir = tmp_path / "out"

    downloaded, errors = download_ftp_dir(ftp, "/2026/07/09/absent", str(local_dir))

    assert downloaded == 0
    assert len(errors) == 1
    assert "/2026/07/09/absent" in errors[0]


def test_download_ftp_dir_continues_after_one_file_fails(tmp_path):
    tree = {
        "2026": {
            "07": {
                "09": {
                    "ok.csv": b"contenu",
                    "bad.csv": b"contenu2",
                }
            }
        }
    }
    ftp = _FakeFTP(tree)

    def failing_retrbinary(cmd, callback):
        if "bad.csv" in cmd:
            raise OSError("disque plein")
        name = cmd.split(" ", 1)[1]
        callback(tree["2026"]["07"]["09"][name])

    ftp.retrbinary = failing_retrbinary
    local_dir = tmp_path / "out"

    downloaded, errors = download_ftp_dir(ftp, "/2026/07/09", str(local_dir))

    assert downloaded == 1
    assert (local_dir / "ok.csv").read_bytes() == b"contenu"
    # `open(local_path, "wb")` crée le fichier avant l'échec de retrbinary --
    # un fichier vide reste (même comportement que le script bash original,
    # qui ne nettoie pas non plus les fichiers partiels en cas d'échec).
    assert (local_dir / "bad.csv").read_bytes() == b""
    assert len(errors) == 1
    assert "bad.csv" in errors[0]
