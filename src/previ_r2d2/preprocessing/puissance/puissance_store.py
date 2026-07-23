"""Récupération de la puissance depuis hydrospot_stream (source brute Previ_v2).

Port de la logique de fusion de Previ_v2/cron/scripts/fusion_csv.py : les 3
fichiers bruts d'une station (`date_power.csv`, `date_power_fill_nan.csv`,
`date_power_fill_nan_neg_price.csv`) sont fusionnés sur `Date` en un seul
CSV. `MA_baisse`/`power_output` reprennent l'étage 1 de Previ_v2
(write_clean_data_v3.py::ConsignesProcessor) : interpolation linéaire sur
les fenêtres de consigne EDF (cf. `consignes.py`), sinon priorité
`Puissance_neg_price` > `Puissance` > 0.0.

`export_puissance_csv` écrit aussi `puissance_horaire.csv` (une ligne/heure) à
côté de `puissance.csv` : `power_output` nettoyé (détection de chaos,
interpolation, ré-échantillonnage horaire par médiane -- cf. `cleaning.py`,
étage 2 de Previ_v2, `data_manager.py::DataManager.load_data`). `puissance.csv`
reste minute par minute, étage 1 uniquement.

Le dossier previ-R2-D2 (ex. `apas_G1_G4`) est mis en correspondance avec un
dossier hydrospot_stream (ex. `Castillon_Apas_G1`) par mot-clé centrale +
numéro de groupe.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

from previ_r2d2.common import config
from previ_r2d2.preprocessing.puissance import cleaning, consignes

SOURCE_FILES = {
    "date_power.csv": "Puissance",
    "date_power_fill_nan.csv": "Puissance_fill_nan",
    "date_power_fill_nan_neg_price.csv": "Puissance_neg_price",
}


class PuissanceMatchError(RuntimeError):
    """Levée quand le dossier hydrospot_stream ne peut pas être déterminé sans ambiguïté."""


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower()


def load_puissance_mapping(path: Path) -> dict:
    """Charge `config/puissance_mapping.yaml` (dossier -> nom de dossier hydrospot_stream).

    Recours explicite pour les cas où l'heuristique mot-clé+groupe ne peut pas
    fonctionner : plusieurs groupes previ-R2-D2 partagent un seul dossier
    hydrospot_stream (ex. la_bastide_G1_G2_G3 -> LabastideSalat_Village_G1),
    ou la convention de groupe diffère (ex. nancy_A -> Nancy_rueDaum_G1).
    """
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def find_source_folder(rec: dict, mapping: dict | None = None) -> str:
    """Trouve le dossier hydrospot_stream correspondant à ce raccordement.

    Priorité à `mapping` (dossier -> nom de dossier hydrospot_stream, cf.
    `load_puissance_mapping`) si le dossier y figure. Sinon, heuristique :
    le nom du dossier source doit contenir le mot-clé centrale (insensible
    casse/accents) ET se terminer par un des numéros de groupe.
    Lève `PuissanceMatchError` si 0 ou plusieurs candidats (ou si le dossier
    mappé n'existe pas).
    """
    dossier = rec.get("dossier")
    mapping = mapping or {}
    if dossier in mapping:
        folder = mapping[dossier]
        if not (config.PUISSANCE_SOURCE_ROOT / folder).is_dir():
            raise PuissanceMatchError(
                f"{dossier} : dossier mappé {folder!r} introuvable dans "
                f"{config.PUISSANCE_SOURCE_ROOT}"
            )
        return folder

    keyword = _normalize(rec.get("centrale") or rec.get("amenagement") or "")
    groupes = [
        g.get("nom_groupe") for g in rec.get("groupes", []) if g.get("nom_groupe")
    ]
    if not keyword or not groupes:
        raise PuissanceMatchError(
            f"{rec.get('dossier')} : mot-clé centrale ou groupes manquants"
        )

    candidates = []
    for folder in config.PUISSANCE_SOURCE_ROOT.iterdir():
        if not folder.is_dir():
            continue
        name_norm = _normalize(folder.name)
        if keyword not in name_norm:
            continue
        if any(re.search(rf"_{re.escape(g.lower())}$", name_norm) for g in groupes):
            candidates.append(folder.name)

    if len(candidates) != 1:
        raise PuissanceMatchError(
            f"{rec.get('dossier')} : {len(candidates)} correspondance(s) "
            f"hydrospot_stream pour mot-clé={keyword!r} groupes={groupes} "
            f"({candidates})"
        )
    return candidates[0]


def _read_source(fichier) -> pd.DataFrame:
    df = pd.read_csv(fichier, sep=";", parse_dates=["Date"])
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        # Une ligne corrompue dans le fichier source fait échouer le parsing
        # vectorisé ; le dtype de repli n'est pas toujours "object" (ex. `str`
        # sous pandas 3.x) — on re-coerce dans tous les cas plutôt que de ne
        # tester qu'une seule valeur de dtype possible.
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"].notna()].reset_index(drop=True)
    return df


def export_puissance_csv(source_folder: str, dest_path) -> int:
    """Fusionne les 3 fichiers bruts de `source_folder` vers `dest_path`.

    Colonnes de sortie : Date;Puissance;Puissance_fill_nan;Puissance_neg_price;
    MA_baisse;power_output (les deux dernières identiques, cf. docstring du
    module). Écrit aussi `<dest_path.parent>/puissance_horaire.csv` (power_output
    nettoyé, horaire, cf. `cleaning.py`). Renvoie le nombre de lignes écrites.
    """
    src_dir = config.PUISSANCE_SOURCE_ROOT / source_folder

    merged: pd.DataFrame | None = None
    for filename, column in SOURCE_FILES.items():
        path = src_dir / filename
        if not path.exists():
            raise PuissanceMatchError(f"{path} introuvable")
        df = _read_source(path).rename(columns={"Puissance": column})
        merged = df if merged is None else pd.merge(merged, df, on="Date", how="outer")

    merged = merged.sort_values("Date")
    value_cols = list(SOURCE_FILES.values())
    merged[value_cols] = merged[value_cols].fillna(0)
    for col in value_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").round(1)

    prefix = consignes.consigne_prefix_for(source_folder)
    true_events, false_events = consignes.read_consigne_events(
        prefix, merged["Date"].min(), merged["Date"].max(),
    )
    interpolated = consignes.interpolate_consignes(merged, true_events, false_events)
    merged["MA_baisse"] = merged.apply(
        lambda row: (
            interpolated[row["Date"]] if row["Date"] in interpolated.index
            else row["Puissance_neg_price"] if row["Puissance_neg_price"] > 0
            else row["Puissance"] if row["Puissance"] > 0
            else 0.0
        ),
        axis=1,
    ).round(1)
    merged["power_output"] = merged["MA_baisse"]

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(dest_path, index=False, sep=";", date_format="%Y-%m-%d %H:%M")

    horaire = cleaning.clean_and_resample_hourly(merged[["Date", "power_output"]].set_index("Date"))
    horaire_path = dest_path.parent / "puissance_horaire.csv"
    horaire.to_csv(horaire_path, sep=";", date_format="%Y-%m-%d %H:%M")

    return len(merged)
