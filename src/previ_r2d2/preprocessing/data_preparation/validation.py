"""Contrat de données de `data_preparation.csv` — validation hand-rolled.

ponytail : pas de pandera / great-expectations pour 2 centrales et une
poignée de règles. stdlib + pandas (déjà là).

Deux niveaux :
  - erreurs   : la donnée est inexploitable pour l'entraînement -> bloque
  - warnings  : la donnée passe mais quelque chose mérite un œil
`strict=True` promeut les warnings en erreurs (utilisé par la CI / le stage
DVC quand on veut être intransigeant).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

TARGET_COL = "debit_m3s"
MAX_TARGET_NAN_SHARE = 0.10   # au-delà : warning sur la cible
MIN_HISTORY_DAYS = 180        # sous ce seuil : split train/test peu fiable
MAX_GAP_HOURS = 24            # trou dans l'index horaire au-delà -> warning


@dataclass
class ValidationReport:
    dossier: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_data_preparation(
    df: pd.DataFrame, dossier: str = "?", *, strict: bool = False
) -> ValidationReport:
    """Vérifie `df` (tel que renvoyé par `read_data_preparation_csv`)."""
    rep = ValidationReport(dossier)

    if df.empty:
        rep.errors.append("fichier vide ou absent")
        return _finish(rep, strict)

    if TARGET_COL not in df.columns:
        rep.errors.append(f"colonne cible '{TARGET_COL}' absente")
        return _finish(rep, strict)

    if not isinstance(df.index, pd.DatetimeIndex):
        rep.errors.append("l'index n'est pas temporel")
        return _finish(rep, strict)

    if not df.index.is_monotonic_increasing:
        rep.errors.append("index non trié chronologiquement")
    if df.index.has_duplicates:
        rep.errors.append(f"{int(df.index.duplicated().sum())} horodatage(s) en double")

    target = df[TARGET_COL]
    if not pd.api.types.is_numeric_dtype(target):
        rep.errors.append(f"'{TARGET_COL}' non numérique ({target.dtype})")
        return _finish(rep, strict)

    valid = target.dropna()
    if valid.empty:
        rep.errors.append(f"'{TARGET_COL}' entièrement vide")
        return _finish(rep, strict)
    if (valid < 0).any():
        rep.errors.append(f"{int((valid < 0).sum())} valeur(s) de débit négative(s)")

    nan_share = float(target.isna().mean())
    if nan_share > MAX_TARGET_NAN_SHARE:
        rep.warnings.append(
            f"{nan_share:.0%} de la cible manquante (seuil {MAX_TARGET_NAN_SHARE:.0%})"
        )

    span_days = (df.index.max() - df.index.min()).total_seconds() / 86400
    if span_days < MIN_HISTORY_DAYS:
        rep.warnings.append(
            f"historique de {span_days:.0f} j (< {MIN_HISTORY_DAYS} j recommandés)"
        )

    gaps = df.index.to_series().diff().dropna()
    big = gaps[gaps > pd.Timedelta(hours=MAX_GAP_HOURS)]
    if not big.empty:
        rep.warnings.append(f"{len(big)} trou(s) > {MAX_GAP_HOURS} h dans l'index (max {big.max()})")

    dead = [c for c in df.columns if c != TARGET_COL and df[c].notna().sum() == 0]
    if dead:
        shown = ", ".join(dead[:5]) + (" ..." if len(dead) > 5 else "")
        rep.warnings.append(f"{len(dead)} colonne(s) entièrement vide(s) : {shown}")

    return _finish(rep, strict)


def _finish(rep: ValidationReport, strict: bool) -> ValidationReport:
    if strict and rep.warnings:
        rep.errors.extend(rep.warnings)
        rep.warnings = []
    return rep
