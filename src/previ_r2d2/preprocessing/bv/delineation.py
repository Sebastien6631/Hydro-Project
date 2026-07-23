"""Mesure de la physiographie d'un bassin versant, par shapefile ou par MNT.

Port direct du script de référence validé `onboarding_bv.py`. `measure_from_polygon`
(méthode principale, précise) exige un shapefile de BV déjà connu ; `delineate_and_measure`
(repli) délimite le BV amont depuis un point exutoire, moins précis mais ne nécessite
qu'une coordonnée. Les deux nécessitent un MNT (GeoTIFF) France entière ; `measure_from_polygon`
suppose ce MNT en EPSG:4326 (voir note du plan/spec), `delineate_and_measure` gère aussi
les MNT en projection métrique (Lambert93 typiquement).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Grille NWP : pas de 0.1° (déduit des fichiers de référence existants).
NWP_GRID_STEP = 0.1
# Nombre maximum de points météo retenus par BV.
MAX_METEO_POINTS = 7


@dataclass
class Physio:
    surface_km2: float
    alt_mean: float
    alt_min: float
    alt_max: float
    aspect_mean: float | None
    gravelius: float
    polygon: object  # shapely.geometry, en EPSG:4326


def measure_from_polygon(shp_path: Path, mnt_path: Path) -> Physio:
    """Physiographie directe depuis un shapefile de BV (saute la délimitation).

    C'est le chemin le plus fiable quand le polygone du BV existe déjà :
    surface géodésique exacte, altitudes et aspect échantillonnés dans le polygone.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.features import geometry_mask
    from pyproj import Geod

    gdf = gpd.read_file(shp_path).to_crs(4326)
    poly = gdf.geometry.union_all()
    geod = Geod(ellps="WGS84")
    area_m2, perim_m = geod.geometry_area_perimeter(poly)
    surface_km2 = abs(area_m2) / 1e6
    gravelius = 0.28 * (perim_m / 1e3) / np.sqrt(surface_km2) if surface_km2 > 0 else float("nan")

    with rasterio.open(mnt_path) as src:
        minx, miny, maxx, maxy = poly.bounds
        win = from_bounds(minx, miny, maxx, maxy, src.transform)
        arr = src.read(1, window=win).astype("float32")
        tr = src.window_transform(win)
        nodata = src.nodata
        mask = geometry_mask([poly], out_shape=arr.shape, transform=tr, invert=True)

    if nodata is not None:
        arr[arr == nodata] = np.nan
    dem_masked = np.where(mask, arr, np.nan)
    vals = dem_masked[np.isfinite(dem_masked)]
    if vals.size == 0:
        raise RuntimeError("Aucune donnée MNT dans le polygone (vérifier CRS / couverture).")
    alt_mean, alt_min, alt_max = float(vals.mean()), float(vals.min()), float(vals.max())
    coslat = max(0.1, float(np.cos(np.radians((miny + maxy) / 2))))
    px_m, py_m = abs(tr.a) * 111320.0 * coslat, abs(tr.e) * 111320.0
    aspect = _mean_aspect(dem_masked, px_m, py_m)

    logger.info("BV (shapefile) : S=%.1f km2 | alt %.0f/%.0f/%.0f m | Kc=%.2f | aspect=%s",
                surface_km2, alt_min, alt_mean, alt_max, gravelius,
                f"{aspect:.0f}deg" if aspect is not None else "n/a")
    return Physio(round(surface_km2, 2), round(alt_mean), round(alt_min),
                  round(alt_max), aspect, round(gravelius, 3), poly)


def delineate_and_measure(lon: float, lat: float, mnt_path: Path,
                           snap_radius_m: float = 1500.0,
                           buffer_m: float = 15000.0,
                           max_buffer_m: float = 200000.0) -> Physio:
    """Délimite le BV amont du point (lon, lat) et calcule sa physiographie.

    Robuste au CRS du MNT : métrique (Lambert 93) ou géographique (EPSG:4326).
    - fenêtrage adaptatif autour de l'exutoire (perf) ;
    - calage sur le max d'accumulation dans un rayon (robuste à l'imprécision) ;
    - expansion tant que le BV grossit, on garde le meilleur tracé ;
    - surface et périmètre calculés en géodésique (exact quel que soit le CRS).
    """
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.features import shapes as rio_shapes
    from pysheds.grid import Grid
    from pyproj import Transformer, Geod
    from shapely.geometry import shape as shp_shape
    from shapely.ops import unary_union, transform as shp_transform

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    with rasterio.open(mnt_path) as src:
        crs = src.crs
    geographic = crs.is_geographic
    coslat = max(0.1, float(np.cos(np.radians(lat))))
    geod = Geod(ellps="WGS84")

    if geographic:
        xr, yr = lon, lat
    else:
        xr, yr = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    to_wgs_tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform

    def window_bounds(cx, cy, bm):
        if geographic:
            dlat = bm / 111320.0
            dlon = bm / (111320.0 * coslat)
            return cx - dlon, cy - dlat, cx + dlon, cy + dlat
        return cx - bm, cy - bm, cx + bm, cy + bm

    logger.info("Délimitation du BV : (lon=%.4f, lat=%.4f) | MNT %s (%s)",
                lon, lat, crs, "géographique" if geographic else "métrique")

    buf = buffer_m
    prev_surface = None
    best = None
    arr = aff = nodata = mask = None
    while True:
        with rasterio.open(mnt_path) as src:
            l, b, r, t = window_bounds(xr, yr, buf)
            win = from_bounds(l, b, r, t, src.transform)
            arr = src.read(1, window=win).astype("float32")
            win_transform = src.window_transform(win)
            nodata = src.nodata
            prof = src.profile.copy()
            prof.update(height=arr.shape[0], width=arr.shape[1], transform=win_transform)

        tmp = Path(tempfile.gettempdir()) / f"_bv_crop_{os.getpid()}.tif"
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(arr, 1)
        grid = Grid.from_raster(str(tmp))
        dem = grid.read_raster(str(tmp))
        conditioned = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
        fdir = grid.flowdir(conditioned, dirmap=dirmap)
        acc = np.array(grid.accumulation(fdir, dirmap=dirmap))
        aff = grid.affine

        c0 = int((xr - aff.c) / aff.a)
        r0 = int((yr - aff.f) / aff.e)
        if geographic:
            rad = max(1, int((snap_radius_m / (111320.0 * coslat)) / abs(aff.a)))
        else:
            rad = max(1, int(snap_radius_m / abs(aff.a)))
        r1, r2 = max(0, r0 - rad), min(acc.shape[0], r0 + rad + 1)
        c1, c2 = max(0, c0 - rad), min(acc.shape[1], c0 + rad + 1)
        sub = acc[r1:r2, c1:c2]
        di, dj = np.unravel_index(np.argmax(sub), sub.shape)
        snap_row, snap_col = r1 + di, c1 + dj

        catch = grid.catchment(x=snap_col, y=snap_row, fdir=fdir, dirmap=dirmap,
                                xytype='index')
        mask = np.asarray(catch).astype(bool)
        tmp.unlink(missing_ok=True)
        if not mask.any():
            raise RuntimeError("Bassin vide : coordonnée hors MNT ou sur du nodata.")

        if geographic:
            w_km = abs(aff.a) * 111.320 * coslat
            h_km = abs(aff.e) * 111.320
            surface = mask.sum() * w_km * h_km
        else:
            surface = mask.sum() * abs(aff.a * aff.e) / 1e6
        touches_edge = (mask[0, :].any() or mask[-1, :].any()
                        or mask[:, 0].any() or mask[:, -1].any())
        logger.info("buffer=%.0f km -> S=%.0f km2 (acc=%d, bord=%s)",
                    buf / 1e3, surface, sub.max(), touches_edge)

        if best is None or surface > best["surface"]:
            best = {"surface": surface, "mask": mask, "arr": arr.copy(),
                    "aff": aff, "nodata": nodata, "edge": touches_edge, "buf": buf}

        grew = prev_surface is None or surface > prev_surface * 1.05
        if (touches_edge or grew) and buf < max_buffer_m:
            prev_surface = surface
            buf = min(buf * 2, max_buffer_m)
            continue
        break

    mask, arr, aff, nodata = best["mask"], best["arr"], best["aff"], best["nodata"]
    if best["edge"] and best["buf"] >= max_buffer_m:
        logger.warning("Surface non stabilisée (bord de fenêtre à %.0f km) : MNT probablement "
                        "incomplet pour ce bassin.", max_buffer_m / 1e3)

    polys = [shp_shape(geom) for geom, val in
             rio_shapes(mask.astype(np.int16), mask=mask, transform=aff) if val == 1]
    poly = unary_union(polys)
    poly_wgs = poly if geographic else shp_transform(to_wgs_tf, poly)
    area_m2, perim_m = geod.geometry_area_perimeter(poly_wgs)
    surface_km2 = abs(area_m2) / 1e6
    perimeter_km = perim_m / 1e3
    gravelius = 0.28 * perimeter_km / np.sqrt(surface_km2) if surface_km2 > 0 else float("nan")

    if nodata is not None:
        arr[arr == nodata] = np.nan
    dem_masked = np.where(mask, arr, np.nan)
    vals = dem_masked[np.isfinite(dem_masked)]
    alt_mean, alt_min, alt_max = float(vals.mean()), float(vals.min()), float(vals.max())
    if geographic:
        px_m, py_m = abs(aff.a) * 111320.0 * coslat, abs(aff.e) * 111320.0
    else:
        px_m, py_m = abs(aff.a), abs(aff.e)
    aspect_mean = _mean_aspect(dem_masked, px_m, py_m)

    from scipy import ndimage as _ndi
    border = mask & ~_ndi.binary_erosion(mask)
    nod_touch = int((border & ~np.isfinite(dem_masked)).sum())
    if nod_touch > border.sum() * 0.05:
        logger.warning("BV bordé par du nodata sur %.0f%% de son contour : surface "
                        "probablement sous-estimée (MNT incomplet pour ce bassin).",
                        100 * nod_touch / max(1, border.sum()))

    logger.info("BV : S=%.1f km2 | alt %.0f/%.0f/%.0f m | Kc=%.2f | aspect=%s",
                surface_km2, alt_min, alt_mean, alt_max, gravelius,
                f"{aspect_mean:.0f}deg" if aspect_mean is not None else "n/a")
    return Physio(round(surface_km2, 2), round(alt_mean), round(alt_min),
                  round(alt_max), aspect_mean, round(gravelius, 3), poly_wgs)


def _mean_aspect(dem: np.ndarray, px: float, py: float) -> float | None:
    """Aspect moyen circulaire (degrés, 0=N) via le gradient du MNT."""
    try:
        gy, gx = np.gradient(dem, py, px)
        aspect = np.degrees(np.arctan2(-gx, gy)) % 360.0
        valid = np.isfinite(aspect)
        if valid.sum() == 0:
            return None
        rad = np.radians(aspect[valid])
        mean = np.degrees(np.arctan2(np.nanmean(np.sin(rad)),
                                      np.nanmean(np.cos(rad)))) % 360.0
        return float(mean)
    except Exception as exc:  # pragma: no cover
        logger.warning("Aspect non calculé : %s", exc)
        return None


def select_meteo_points(polygon, mnt_path: Path | None,
                         step: float = NWP_GRID_STEP,
                         k: int = MAX_METEO_POINTS,
                         historical_points: set[tuple[float, float]] | None = None,
                         ) -> list[tuple[float, float]]:
    """Sélectionne jusqu'à k points de grille NWP représentant le BV.

    1. Génère les nœuds de grille 0.1° dans l'emprise du BV.
    2. Garde ceux dont le centre tombe dans le polygone.
    3. Si `historical_points` est fourni, préfère les candidats qui y figurent
       (grille couverte historiquement) ; repli sur tous les candidats si
       aucun n'a de couverture historique.
    4. Si <= k : tous. Sinon : k-means sur (lat, lon, altitude) puis on retient
       le nœud de grille le plus proche de chaque centroïde.
    """
    from shapely.geometry import Point

    minx, miny, maxx, maxy = polygon.bounds
    lons = np.round(np.arange(np.floor(minx / step) * step,
                               np.ceil(maxx / step) * step + step, step), 1)
    lats = np.round(np.arange(np.floor(miny / step) * step,
                               np.ceil(maxy / step) * step + step, step), 1)
    cand = [(round(la, 1), round(lo, 1)) for la in lats for lo in lons
            if polygon.contains(Point(lo, la))]
    if historical_points:
        # Arrondi identique à `cand` -- évite un faux négatif si les floats
        # bruts du fichier NWP diffèrent d'un ulp de ceux régénérés ici.
        historical_rounded = {(round(la, 1), round(lo, 1)) for la, lo in historical_points}
        filtered = [p for p in cand if p in historical_rounded]
        if filtered:
            cand = filtered
    if not cand:
        c = polygon.centroid
        return [(round(c.y, 1), round(c.x, 1))]
    if len(cand) <= k:
        logger.info("%d point(s) de grille dans le BV (<= %d) : tous retenus.", len(cand), k)
        return cand

    from sklearn.cluster import KMeans
    arr = np.array([[la, lo] for la, lo in cand], dtype=float)
    z = _sample_altitudes(cand, mnt_path)
    feats = np.c_[arr, z]
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(feats)
    chosen = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        centroid = km.cluster_centers_[c]
        best = idx[np.argmin(((feats[idx] - centroid) ** 2).sum(1))]
        chosen.append(cand[best])
    logger.info("%d points de grille -> %d points météo représentatifs.", len(cand), len(chosen))
    return chosen


def _sample_altitudes(points: list[tuple[float, float]], mnt_path: Path | None) -> np.ndarray:
    """Altitude de chaque point (lat, lon). 0 si MNT indisponible."""
    if mnt_path is None:
        return np.zeros(len(points))
    try:
        import rasterio
        from pyproj import Transformer
        with rasterio.open(mnt_path) as src:
            # CRS dérivé du MNT lui-même (pas un DEM_CRS en dur) : le MNT réel
            # de ce projet est en EPSG:4326, alors que le DEM_CRS d'origine
            # (Lambert93) visait le MNT_REDUIT du script de référence.
            to_dem = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = to_dem.transform([lo for _, lo in points], [la for la, _ in points])
            return np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
    except Exception as exc:  # pragma: no cover
        logger.warning("Altitude des points non échantillonnée : %s", exc)
        return np.zeros(len(points))
