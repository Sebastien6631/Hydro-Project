"""Orchestration de l'onboarding BV : centrales/ -> centrales/<dossier>/bv.json.

Découvre les centrales depuis `centrales/*/config-raccordement.json`, résout
pour chacune son bassin versant soit via un shapefile connu (mesure exacte,
`config/bv_mapping.yaml`), soit via une délimitation MNT depuis son point
exutoire (repli). Regroupe les dossiers qui partagent le même site physique
(même `centrale_uuid` — raccordements RTE scindés par groupe, ex.
bonneval_G1/bonneval_G2) pour ne mesurer qu'une fois par site, puis écrit un
`bv.json` autoporteur dans chaque dossier concerné.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from previ_r2d2.common import config
from previ_r2d2.preprocessing.bv import delineation, transit
from previ_r2d2.preprocessing.bv.rules import Rules, estimate_exposition, estimate_kbase, estimate_kc_unit
from previ_r2d2.preprocessing.debit import station_store
from previ_r2d2.preprocessing.debit.hubeau import HubEauClient
from previ_r2d2.preprocessing.meteo import nwp_reader

logger = logging.getLogger(__name__)

ALTITUDE_REVIEW_THRESHOLD_M = 150.0

HISTORICAL_GRID_REFERENCE = Path("2021") / "01" / "01" / "BARTHE_ENR_EC_OP_recent_2021010100_000.csv"


def historical_grid_points(nas_meteo: Path) -> set[tuple[float, float]] | None:
    """Points de la grille NWP de référence 2021 (couverture historique
    minimale garantie), ou None si absente/illisible -- comportement alors
    identique à avant ce fix (sélection géométrique pure dans select_meteo_points)."""
    path = nas_meteo / HISTORICAL_GRID_REFERENCE
    if not path.exists():
        return None
    try:
        return nwp_reader.grid_points(path)
    except Exception as exc:
        logger.warning("Grille de référence 2021 illisible (%s), repli géométrique.", exc)
        return None


def discover_centrale_records(centrales_dir: Path) -> list[dict]:
    """Charge le config-raccordement.json de chaque dossier de `centrales/`.

    Un fichier illisible (JSON corrompu) est loggé et ignoré plutôt que de
    faire échouer la découverte pour toutes les autres centrales.
    """
    records = []
    for path in sorted(centrales_dir.glob("*/config-raccordement.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("%s : config-raccordement.json illisible (%s), ignoré.",
                         path.parent.name, exc)
    return records


def group_by_site(records: list[dict]) -> dict[str, list[dict]]:
    """Regroupe les enregistrements partageant le même site physique."""
    groups: dict[str, list[dict]] = {}
    for rec in records:
        key = rec.get("centrale_uuid") or rec["dossier"]
        groups.setdefault(key, []).append(rec)
    return groups


def bv_json_path(centrales_dir: Path, dossier: str) -> Path:
    return centrales_dir / dossier / "bv.json"


def group_is_complete(centrales_dir: Path, group: list[dict]) -> bool:
    return all(bv_json_path(centrales_dir, rec["dossier"]).exists() for rec in group)


def load_bv_mapping(path: Path) -> dict:
    """Charge `config/bv_mapping.yaml` (dossier -> nom de shapefile, ou None)."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_shapefile_path(dossier: str, mapping: dict, shapefiles_dir: Path) -> Path | None:
    """Chemin du shapefile BV du dossier, ou None (repli MNT)."""
    name = mapping.get(dossier)
    if not name:
        return None
    path = shapefiles_dir / f"BV_{name}.shp"
    if not path.exists():
        logger.warning("%s : shapefile BV_%s.shp introuvable dans %s, repli sur le MNT.",
                        dossier, name, shapefiles_dir)
        return None
    return path


def _log_altitude_mismatch(dossier: str, adresse_alt: float | None, alt_min: float,
                            threshold: float = ALTITUDE_REVIEW_THRESHOLD_M) -> None:
    """Avertit (log uniquement, pas un champ du JSON) si l'exutoire est mal calé."""
    if adresse_alt is None:
        return
    diff = abs(float(adresse_alt) - alt_min)
    if diff > threshold:
        logger.warning("%s : altitude connue (%.0f m) vs BV mesuré (%.0f m) — écart de "
                        "%.0f m (> %.0f m), exutoire ou shapefile à vérifier.",
                        dossier, adresse_alt, alt_min, diff, threshold)


def _try_build_stations_hydrometriques(dossier: str, station_reference: str | None,
                                        stations_amont: list[str], hubeau_client: HubEauClient,
                                        debit_dir: Path | None) -> list[dict] | None:
    """Coordonnées Hub'Eau + transit amont->référence. None si Hub'Eau échoue
    -- l'appelant décide alors de conserver l'ancienne valeur du bv.json."""
    if not station_reference and not stations_amont:
        return []
    codes = ([station_reference] if station_reference else []) + list(stations_amont)
    try:
        coords = hubeau_client.stations_referentiel(codes)
    except Exception as exc:
        logger.warning("%s : Hub'Eau referentiel/stations indisponible (%s)", dossier, exc)
        return None

    df_reference = None
    if station_reference and station_reference in coords and debit_dir is not None:
        ref_path = debit_dir / station_store.filename_for(station_reference, "reference")
        if ref_path.exists():
            df_reference = transit.load_debit_series(ref_path)

    entries: list[dict] = []
    if station_reference and station_reference in coords:
        entries.append({"code": station_reference, "role": "reference", **coords[station_reference]})
    for code in stations_amont:
        if code not in coords:
            continue
        entry = {"code": code, "role": "amont", **coords[code]}
        if df_reference is not None:
            amont_path = debit_dir / station_store.filename_for(code, "amont")
            if amont_path.exists():
                lags = transit.transit_amont_reference(transit.load_debit_series(amont_path), df_reference)
                if lags:
                    entry["transit_vers_reference_h"] = lags
        entries.append(entry)
    return entries


def build_bv_record(dossier: str, lat: float, lon: float, adresse_alt: float | None,
                     mnt_path: Path, rules: Rules, *, shapefile_path: Path | None,
                     generated_utc: str,
                     station_reference: str | None = None,
                     stations_amont: list[str] | None = None,
                     hubeau_client: HubEauClient | None = None,
                     debit_dir: Path | None = None) -> dict:
    """Calcule le bv.json d'une centrale (une mesure, shapefile ou MNT).

    `station_reference`/`stations_amont`/`hubeau_client`/`debit_dir` sont
    optionnels : sans `hubeau_client`, `stations_hydrometriques` n'est pas
    calculé (rétro-compatible). En cas d'échec Hub'Eau, la clé est omise
    plutôt que remplie avec une liste vide -- à l'appelant de conserver
    l'ancienne valeur du bv.json existant.

    `stations_meteo_nwp` préfère les points de grille NWP déjà couverts en
    2021 (cf. `historical_grid_points`) -- repli sur la sélection purement
    géométrique si aucun candidat du polygone n'a de couverture historique.
    """
    if shapefile_path is not None:
        physio = delineation.measure_from_polygon(shapefile_path, mnt_path)
    else:
        physio = delineation.delineate_and_measure(lon, lat, mnt_path)

    kc_unit = estimate_kc_unit(physio.alt_mean, rules)
    kbase, kbase_review = estimate_kbase(physio.alt_mean, rules)
    exposition = estimate_exposition(physio.aspect_mean, rules)
    points = delineation.select_meteo_points(
        physio.polygon, mnt_path, historical_points=historical_grid_points(config.NAS_METEO)
    )
    _log_altitude_mismatch(dossier, adresse_alt, physio.alt_min)

    if kbase_review:
        logger.warning("%s : K_base=%.1f dans la zone ambiguë 1000-1300 m, à confirmer.",
                        dossier, kbase)

    record = {
        "centrale": dossier,
        "source": {"generated_utc": generated_utc},
        "exutoire": {"lat": lat, "lon": lon},
        "bassin_versant": {
            "surface_km2": physio.surface_km2,
            "altitude_moyenne_m": physio.alt_mean,
            "altitude_min_m": physio.alt_min,
            "altitude_max_m": physio.alt_max,
            "gravelius": physio.gravelius,
            "aspect_moyen_deg": physio.aspect_mean,
        },
        "parametres_calage": {
            "K_base": kbase,
            "exposition": exposition,
            "kc_unit": round(kc_unit, 3),
            "kbase_needs_review": kbase_review,
            "exposition_needs_review": True,
        },
        "stations_meteo_nwp": [
            {"id": i + 1, "lat": la, "lon": lo} for i, (la, lo) in enumerate(points)
        ],
    }

    if hubeau_client is not None:
        stations = _try_build_stations_hydrometriques(
            dossier, station_reference, stations_amont or [], hubeau_client, debit_dir,
        )
        if stations is not None:
            record["stations_hydrometriques"] = stations

    return record


def write_bv_json(centrales_dir: Path, dossier: str, record: dict) -> Path:
    path = bv_json_path(centrales_dir, dossier)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _keep_existing_field(record: dict, centrales_dir: Path, dossier: str, field: str) -> None:
    old_path = bv_json_path(centrales_dir, dossier)
    if not old_path.exists():
        return
    try:
        old = json.loads(old_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if field in old:
        record[field] = old[field]


def _merge_transit_centrale(payload: dict, centrales_dir: Path, dossier: str, data_dir: Path) -> None:
    """Calcule `transit_vers_centrale_h` pour ce dossier : priorité à la
    corrélation directe débit référence -> power_output (fiable si >= 0.9) ;
    à défaut, repli géométrique par ratio de distances sur le
    `transit_vers_reference_h` déjà mesuré des stations amont (cf.
    `transit.estimate_transit_centrale_geometric`). Conserve l'ancienne valeur
    du bv.json existant si ni l'un ni l'autre n'est disponible."""
    ref_entry = next((s for s in payload.get("stations_hydrometriques", [])
                       if s.get("role") == "reference"), None)
    lags: dict = {}
    if ref_entry is not None:
        try:
            ref_path = data_dir / dossier / station_store.filename_for(ref_entry["code"], "reference")
            puiss_path = data_dir / dossier / "puissance_horaire.csv"
            if ref_path.exists() and puiss_path.exists():
                lags = transit.transit_reference_centrale(
                    transit.load_debit_series(ref_path), transit.load_puissance_series(puiss_path),
                )
        except Exception as exc:
            logger.warning("%s : échec calcul transit_vers_centrale_h (%s)", dossier, exc)
            lags = {}

    amont_entries = [s for s in payload.get("stations_hydrometriques", []) if s.get("role") == "amont"]
    exutoire = payload.get("exutoire")
    if ref_entry is not None and exutoire is not None and amont_entries:
        try:
            geometrique = transit.estimate_transit_centrale_geometric(
                amont_entries, ref_entry, exutoire["lat"], exutoire["lon"],
            )
        except Exception as exc:
            logger.warning("%s : échec calcul géométrique transit_vers_centrale_h (%s)", dossier, exc)
            geometrique = {}
        for saison, valeur in geometrique.items():
            lags.setdefault(saison, valeur)

    if lags:
        payload["transit_vers_centrale_h"] = lags
    else:
        _keep_existing_field(payload, centrales_dir, dossier, "transit_vers_centrale_h")


def run_batch(centrales_dir: Path, mnt_path: Path, rules: Rules, mapping: dict,
              shapefiles_dir: Path, *, force: bool = False,
              generated_utc: str | None = None,
              hubeau_client: HubEauClient | None = None,
              data_dir: Path | None = None) -> dict:
    """Traite toutes les centrales de `centrales_dir`, une mesure par site physique."""
    generated_utc = generated_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    groups = group_by_site(discover_centrale_records(centrales_dir))

    done: list[str] = []
    skipped: list[str] = []
    errors: list[tuple[str, str]] = []

    for site_key, group in groups.items():
        if not force and group_is_complete(centrales_dir, group):
            skipped.extend(rec["dossier"] for rec in group)
            continue

        rep = group[0]
        shapefile_path = resolve_shapefile_path(rep["dossier"], mapping, shapefiles_dir)
        try:
            record = build_bv_record(
                rep["dossier"], rep["adresse_lat"], rep["adresse_lng"],
                rep.get("adresse_alt"), mnt_path, rules,
                shapefile_path=shapefile_path, generated_utc=generated_utc,
                station_reference=rep.get("station_vigicrue_reference") or None,
                stations_amont=rep.get("stations_vigicrue_amont") or [],
                hubeau_client=hubeau_client,
                debit_dir=(data_dir / rep["dossier"]) if data_dir else None,
            )
        except Exception as exc:
            logger.error("BV %s : échec de la mesure (%s)", site_key, exc)
            errors.append((site_key, str(exc)))
            continue

        if "stations_hydrometriques" not in record:
            _keep_existing_field(record, centrales_dir, rep["dossier"], "stations_hydrometriques")

        dossiers = []
        for rec in group:
            if not force and bv_json_path(centrales_dir, rec["dossier"]).exists():
                skipped.append(rec["dossier"])
                continue
            payload = dict(record)
            payload["centrale"] = rec["dossier"]
            if data_dir is not None:
                _merge_transit_centrale(payload, centrales_dir, rec["dossier"], data_dir)
            write_bv_json(centrales_dir, rec["dossier"], payload)
            dossiers.append(rec["dossier"])
        done.extend(dossiers)
        logger.info("BV %s -> %s (%s)", site_key, dossiers,
                    "shapefile" if shapefile_path else "MNT")

    logger.info("Terminé : %d écrite(s), %d ignorée(s) (déjà présentes), %d erreur(s).",
                len(done), len(skipped), len(errors))
    return {"done": done, "skipped": skipped, "errors": errors}


def run_single(centrales_dir: Path, dossier: str, mnt_path: Path, rules: Rules,
               mapping: dict, shapefiles_dir: Path, *,
               generated_utc: str | None = None,
               hubeau_client: HubEauClient | None = None,
               data_dir: Path | None = None) -> dict:
    """Traite une seule centrale (test ciblé), sans tenir compte du dédoublonnage."""
    path = centrales_dir / dossier / "config-raccordement.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    generated_utc = generated_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    shapefile_path = resolve_shapefile_path(dossier, mapping, shapefiles_dir)

    record = build_bv_record(
        dossier, rec["adresse_lat"], rec["adresse_lng"], rec.get("adresse_alt"),
        mnt_path, rules, shapefile_path=shapefile_path, generated_utc=generated_utc,
        station_reference=rec.get("station_vigicrue_reference") or None,
        stations_amont=rec.get("stations_vigicrue_amont") or [],
        hubeau_client=hubeau_client,
        debit_dir=(data_dir / dossier) if data_dir else None,
    )
    if "stations_hydrometriques" not in record:
        _keep_existing_field(record, centrales_dir, dossier, "stations_hydrometriques")
    if data_dir is not None:
        _merge_transit_centrale(record, centrales_dir, dossier, data_dir)
    write_bv_json(centrales_dir, dossier, record)
    return record
