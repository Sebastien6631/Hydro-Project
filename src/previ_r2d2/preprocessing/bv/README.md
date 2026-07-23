# preprocessing/bv — Onboarding BV

Caractérise automatiquement le bassin versant (BV) amont de chaque centrale,
et écrit `centrales/<dossier>/bv.json`. Deux méthodes de mesure :

- **Shapefile** (méthode principale, précise à <1 %/±1 m) : quand
  `config/bv_mapping.yaml` connaît le shapefile BV de la centrale
  (`centrales/REFERENCE/shapefiles/BV_<Nom>.shp`), la physiographie est mesurée
  directement sur le polygone connu.
- **Délimitation MNT** (repli) : pour les centrales sans shapefile connu
  (aujourd'hui : `counozouls_G1`), le BV est délimité depuis le point
  exutoire (`adresse_lat`/`adresse_lng`) par analyse d'accumulation de flux
  sur le MNT — moins précis, sensible au calage du point.

## Utilisation

```bash
# Toutes les centrales de centrales/ (dédoublonnage automatique par site physique)
python cron/scripts/onboarding-bv.py batch --mnt "$PREVI_MNT"

# Recalcule même les centrales qui ont déjà un bv.json
python cron/scripts/onboarding-bv.py batch --mnt "$PREVI_MNT" --force

# Une seule centrale (test ciblé)
python cron/scripts/onboarding-bv.py single --dossier apas_G1_G4 --mnt "$PREVI_MNT"
```

`$PREVI_MNT` doit pointer vers un GeoTIFF France entière en EPSG:4326 (requis
par la méthode shapefile ; la méthode MNT gère aussi le Lambert93). `--rules`,
`--mapping` et `--shapefiles-dir` prennent par défaut respectivement
`centrales/REFERENCE/bv_rules.json`, `config/bv_mapping.yaml` et
`centrales/REFERENCE/shapefiles/`.

**Un `bv.json` déjà présent n'est jamais retraité sans `--force`** (idempotence,
y compris via `dvc repro` qui appelle `batch` sans `--force`) — si le schéma
de sortie évolue (ex. ajout d'un nouveau champ), les centrales déjà
onboardées ne le reçoivent pas tant qu'on ne relance pas explicitement
`batch --force` (ou `single --dossier ...`, qui écrase toujours).

**`centrales/REFERENCE/shapefiles/` n'est pas versionné** (contrairement à
`bv_rules.json`/`centrales_calibration.json`) — comme le MNT, c'est un
dossier local à peupler manuellement sur chaque machine qui exécute
`onboarding-bv.py`. `config/bv_mapping.yaml`, lui, est bien versionné (petit
fichier texte).

## Modules

- `rules.py` — formules de calage (`kc_unit`, `K_base`, `exposition`), calées
  sur `centrales/REFERENCE/centrales_calibration.json`.
- `delineation.py` — mesure de la physiographie, par shapefile
  (`measure_from_polygon`) ou par délimitation MNT (`delineate_and_measure`),
  et sélection des points de grille météo NWP représentatifs.
- `bv_builder.py` — orchestration : découverte des centrales, résolution du
  shapefile via `config/bv_mapping.yaml`, dédoublonnage par site physique
  (`centrale_uuid`), assemblage et écriture du JSON.

## Ajouter une centrale au mapping shapefile

Si un shapefile BV existe pour une nouvelle centrale : le déposer dans
`centrales/REFERENCE/shapefiles/BV_<Nom>.{shp,shx,dbf,prj,cpg}` et ajouter une
ligne `dossier: Nom` dans `config/bv_mapping.yaml` — **vérifier d'abord** que
le polygone contient bien les coordonnées `adresse_lat`/`adresse_lng` de la
centrale (cf. script de vérification dans le plan d'implémentation, Task 4) ;
ne pas se fier au nom seul. Sans shapefile connu, ne rien ajouter — la
centrale utilisera automatiquement le repli MNT.

## Schéma de sortie (`centrales/<dossier>/bv.json`)

```json
{
  "centrale": "apas_G1_G4",
  "source": {"generated_utc": "2026-07-07T08:30:00Z"},
  "exutoire": {"lat": 43.1312, "lon": 0.922689},
  "bassin_versant": {
    "surface_km2": 2052.3,
    "altitude_moyenne_m": 1201,
    "altitude_min_m": 250,
    "altitude_max_m": 2400,
    "gravelius": 1.55,
    "aspect_moyen_deg": 190
  },
  "parametres_calage": {
    "K_base": 3.0,
    "exposition": 0.98,
    "kc_unit": 0.878,
    "kbase_needs_review": true,
    "exposition_needs_review": true
  },
  "stations_meteo_nwp": [
    {"id": 1, "lat": 43.1, "lon": 0.9}
  ],
  "stations_hydrometriques": [
    {"code": "O020002001", "role": "reference", "lat": 43.098, "lon": 0.706, "altitude": 357.0},
    {"code": "O001531001", "role": "amont", "lat": 42.867, "lon": 0.748, "altitude": 552.0,
     "transit_vers_reference_h": {"DJF": 9, "MAM": 10, "JJA": 7, "SON": 6}}
  ],
  "transit_vers_centrale_h": {"DJF": 5.1, "MAM": 5.9, "JJA": 4.9, "SON": 4.5}
}
```

`stations_hydrometriques` : coordonnées Hub'Eau (`referentiel/stations`) de la
station de référence et des stations amont (`station_vigicrue_reference`/
`stations_vigicrue_amont`). `transit_vers_reference_h` (par station amont,
débit vs débit) : cross-corrélation saisonnière, aucun seuil de fiabilité,
utilise tout l'historique disponible. `transit_vers_centrale_h` (racine du
JSON, calculé **par dossier** pas par site) : cross-corrélation débit vs
`power_output` nettoyé (`puissance_horaire.csv`) si corrélation directe ≥ 0.9,
sinon repli géométrique par ratio de distances sur `transit_vers_reference_h`
(médiane entre stations amont si plusieurs — voir `transit.py`). Années ≥ 2026
exclues du calcul direct (R2 activée, `power_output` non représentatif du
potentiel hydraulique). Robustesse : échec Hub'Eau ou calcul → conserve la
valeur déjà présente dans le `bv.json` existant plutôt que d'écraser par du
vide ou une liste vide.

`kbase_needs_review` : altitude moyenne du BV dans la bande 1000-1300 m
(règle ambiguë, ex. `apas_G1_G4`). `exposition_needs_review` : toujours
`true` — l'aspect n'explique qu'environ 66 % de la variance, à ajuster
manuellement si besoin. Le garde-fou d'altitude (`adresse_alt` OneGate vs
`altitude_min_m` mesuré, écart > 150 m) n'est **pas** un champ du JSON — il
est loggé en avertissement (`logger.warning`) pour rester fidèle au schéma
exact attendu.

Une seule mesure est calculée par site physique (`centrale_uuid`) : les
centrales qui partagent le même site (ex. raccordements RTE scindés par
groupe) reçoivent un `bv.json` identique dans leurs dossiers respectifs.

## Tests

```bash
pytest tests/preprocessing/bv/ -v
```

Les tests de calage (`test_rules.py`, `test_bv_builder.py`) comparent les
formules aux valeurs réelles calées à la main pour les centrales Previ_v2
(`centrales/REFERENCE/centrales_calibration.json`), pas à des données inventées.
Aucun test n'exécute la mesure réelle (shapefile ou MNT — trop lent, nécessite
les fichiers volumineux) — `measure_from_polygon`/`delineate_and_measure`/
`select_meteo_points` sont mockés. La validation end-to-end se fait
manuellement avec les vrais shapefiles + MNT.
