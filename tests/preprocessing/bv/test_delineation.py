from __future__ import annotations

from shapely.geometry import box

from previ_r2d2.preprocessing.bv.delineation import select_meteo_points

# Rectangle EPSG:4326 couvrant exactement 6 noeuds de grille 0.1 deg
# (lat 42.6/42.7 x lon 1.8/1.9/2.0), <= MAX_METEO_POINTS (7) : pas de k-means.
SMALL_POLYGON = box(1.75, 42.55, 2.05, 42.75)
SMALL_CANDIDATES = [
    (42.6, 1.8), (42.6, 1.9), (42.6, 2.0),
    (42.7, 1.8), (42.7, 1.9), (42.7, 2.0),
]

# Rectangle plus grand : 25 noeuds de grille candidats (> 7) -> k-means.
BIG_POLYGON = box(1.65, 42.45, 2.15, 42.95)


def test_select_meteo_points_prefers_historical_subset_when_some_candidates_match():
    historical_points = {(42.6, 1.8), (42.6, 1.9)}

    result = select_meteo_points(SMALL_POLYGON, mnt_path=None, historical_points=historical_points)

    assert sorted(result) == sorted(historical_points)


def test_select_meteo_points_falls_back_to_geometric_selection_when_no_historical_match():
    historical_points = {(99.9, 99.9)}  # aucun point de la grille historique dans ce BV

    result = select_meteo_points(SMALL_POLYGON, mnt_path=None, historical_points=historical_points)

    assert sorted(result) == sorted(SMALL_CANDIDATES)


def test_select_meteo_points_default_historical_points_is_none_unchanged_behavior():
    result = select_meteo_points(SMALL_POLYGON, mnt_path=None)

    assert sorted(result) == sorted(SMALL_CANDIDATES)


def test_select_meteo_points_applies_kmeans_on_filtered_historical_subset_when_above_k():
    historical_points = {
        (42.5, 1.7), (42.5, 1.8), (42.5, 1.9), (42.5, 2.0), (42.5, 2.1),
        (42.6, 1.7), (42.6, 1.8), (42.6, 1.9), (42.6, 2.0),
    }  # 9 points, > k=7 -> k-means doit s'appliquer, mais seulement sur ce sous-ensemble

    result = select_meteo_points(BIG_POLYGON, mnt_path=None, historical_points=historical_points)

    assert len(result) == 7
    assert all(point in historical_points for point in result)
