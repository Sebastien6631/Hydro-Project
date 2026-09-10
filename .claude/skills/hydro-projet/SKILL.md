---
name: hydro-projet
description: |
  Architecture, pipeline DVC et pièges connus de previ-R2-D2 — VERSION PROJET
  DE COURS MLOps (dépôt DagsHub, 2 centrales, hors réseau de l'entreprise).
  Utilise ce skill quand l'utilisateur parle de : previ-R2-D2, ce projet de
  cours, apas_G1_G4/touzac_g2_G2, dvc.yaml, source=live|frozen,
  DagsHub, promote_model, onboarding BV, entraînement/prédiction hybride
  LightGBM+BiLSTM+Stacking, ou de tout script sous cron/scripts/.
---

# Skill : Hydro-Projet — version projet de cours (dépôt DagsHub)

## Ce que c'est

Ce dépôt est une version **réduite et publique** du pipeline previ-R2-D2
(prévision débit → puissance de centrales hydroélectriques, initialement un
projet de production chez Barthe EnR). Simplifié le 2026-07-23 pour un
projet de cours MLOps, réalisé hors du réseau de l'entreprise — voir
`docs/superpowers/specs/2026-07-23-simplification-projet-cours-design.md`
et `docs/superpowers/plans/2026-07-23-simplification-projet-cours.md`
(non versionnés, `docs/` est gitignoré — lisibles seulement en local sur la
machine qui a fait la simplification).

**Ce qui a été retiré** (inaccessible hors serveur de production) :
- **OneGate/memorandum** — API interne de structure des centrales.
- **automate** (rsync/SSH, `HAUTE_CHUTE`) — aucune centrale gardée ne
  l'utilise (les 2 sont `flex_strategy: DEFAULT`).
- **hydrospot_stream** (import réel de puissance) — seule la résolution de
  mapping (`preprocessing/puissance/puissance_store.py::find_source_folder`/
  `load_puissance_mapping`) a survécu, car `preprocessing/onboarding/
  validation.py` (gardé) en dépend.
- **FTP météo NWP** (`nwp_ftp.py`/`retention.py`) — seul
  `preprocessing/meteo/nwp_reader.py` (parseur pur, sans réseau) a survécu,
  utilisé par `bv_builder.py` et `data_preparation/dossier_window.py`.
- **Mail/digest quotidien** (`mailer.py`/`daily_report.py`) et **MLflow**
  (`mlflow_tracker.py`) — retirés entièrement. MLflow est **volontairement**
  à refaire proprement comme partie du travail de cours, pas juste "pas
  encore fait".

**Ce qui marche toujours réellement** : le débit (Hub'Eau/eaufrance, API
publique), l'onboarding BV (idempotent, no-op pour les 2 centrales déjà
onboardées), l'entraînement (LightGBM+BiLSTM+Stacking, intact), la
prédiction (avec un nouveau mode `source="frozen"` 100% offline en plus du
mode `"live"` historique).

## Périmètre : 2 centrales seulement

`apas_G1_G4`, `touzac_g2_G2` — toutes `flex_strategy: DEFAULT`.
Les 10 autres centrales de la version production (bonneval_G1/G2,
campagne_G1_G2, clairac_rd_G2_G1, counozouls_G1, la_bastide_G1_G2_G3,
meb_G2_amont, melles, morgane_G1_G2, rebouc_G1) ont été retirées du disque
et des fichiers de config versionnés (`config-general.json`,
`centrales_calibration.json`, `puissance_mapping.yaml`, `bv_mapping.yaml`,
`shapefiles/`).

## Architecture (package `previ_r2d2`)

```
Hydro-Project/
├── pyproject.toml               # dépendances réelles (mlflow retiré ; rasterio/geopandas/
│                                #   pysheds PAS installés dans l'env -- lazy-import only,
│                                #   jamais exercés pour les 2 centrales déjà onboardées)
├── run.py                        # CLI manuelle expés (--train --dossier/--all-dossiers)
├── models/                        # source de vérité modèles en PRODUCTION, versionné DVC+git
│                                  #   <dossier>/h<horizon>/{version.json, meta_config.json, ...}
├── weights/hybrid{,_candidate}/   # zone de travail manuelle / candidat en cours d'évaluation
├── src/previ_r2d2/
│   ├── common/                  # config.py, dvc_markers.py, onegate.py/mailer.py/
│   │                            #   daily_report.py/mlflow_tracker.py SUPPRIMÉS
│   ├── preprocessing/
│   │   ├── debit/                # eaufrance.py, hubeau.py, hydro_export.py, hydro_update.py,
│   │   │                         #   station_store.py, debit_csv.py (read_debit_csv)
│   │   ├── puissance/             # puissance_store.py TRIMMÉ (mapping seulement,
│   │   │                         #   export_puissance_csv/cleaning.py/consignes.py supprimés)
│   │   ├── meteo/                 # open_meteo.py SEUL (API Météo-France ; nwp_reader.py
│   │   │                         #   supprimé avec les fichiers NWP, cf. section Météo)
│   │   ├── data_preparation/      # dossier_window.py (build_dossier), data_preparation_csv.py
│   │   ├── onboarding/            # validation.py (missing_fields, DEFAULT seulement -- pas
│   │   │                         #   de HAUTE_CHUTE, load_sync_config retiré)
│   │   (bv/ SUPPRIMÉ -- toute la chaîne de création du bv.json, figée hors prod)
│   ├── model/                     # features/, architectures/{lightgbm,bilstm,stacking},
│   │                             #   pipeline/{orchestrator,predict_orchestrator,promotion,
│   │                             #     hydraulic, split, oof_cache, stacking_fit, ...}
│   │                             #   -- PLUS "intact" : cf. « Chantier meta-learner » ci-dessous.
│   │                             #   model/tracking/ SUPPRIMÉ (vestige MLflow vide)
│   ├── cli.py
│   └── postprocessing/           # archive.py (ARCHIVE_ROOT, plus NAS_ARCHIVE_ROOT)
│                                  #   api/ et archiving/ SUPPRIMÉS (packages vides jamais importés)
├── cron/scripts/                 # maj-data.py, onboarding-check.py,
│                                  #   build-data-preparation.py, train.py, predict-archive.py
│                                  #   (majdata-memo/maj-automate/maj-puissance/maj-meteo/
│                                  #    clean-meteo/daily-sync-report SUPPRIMÉS ; cron/wrappers/
│                                  #    SUPPRIMÉ -- lancement manuel uniquement)
├── dvc/
│   ├── preprocessing/dvc.yaml    # 3 stages : debit -> onboarding_check -> data_preparation
│   └── postprocessing/dvc.yaml   # predict_archive (inchangé)
├── config/
│   ├── centrales/<dossier>/      # vide, .gitkeep (jamais peuplé)
│   ├── bv_mapping.yaml           # 2 entrées (apas_G1_G4, touzac_g2_G2)
│   └── puissance_mapping.yaml    # 1 entrée (touzac_g2_G2 -- apas via heuristique)
├── centrales/                    # gitignoré (sauf .dvc) -- DVC-tracké, remote DagsHub
│   ├── REFERENCE/                # config-general.json.dvc, shapefiles.dvc, bv_rules.json,
│   │                             #   centrales_calibration.json (git-trackés directement),
│   │                             #   files/ (gros GIS annexes, MNT France -- NI git NI DVC,
│   │                             #   jamais utilisé : les 2 centrales ont toutes un shapefile)
│   └── <dossier>/                # <dossier>.dvc -- config-raccordement.json, bv.json, *.csv,
│                                  #   data_preparation.csv, prevision.json, enchere.json
└── ARCHIVE/                       # local, gitignoré (remplace l'ancien NAS_ARCHIVE_ROOT)
```

`NAS_DATA_ROOT` (`config.py`) est maintenant un simple alias de
`CENTRALES_DIR` (plus de NAS, un seul niveau de stockage) —
`station_store.ensure_local_symlink` a été **supprimé** (créer un symlink
d'un fichier vers lui-même l'aurait détruit). `ARCHIVE_ROOT = ROOT /
"ARCHIVE"` remplace `NAS_ARCHIVE_ROOT`.

## Environnement Python

Env conda dédié `projet-mlops` (Python 3.11, conda-forge) — pas
`requirements.txt`/`pip install -r` (vestige non à jour). `rasterio`/
`geopandas`/`pysheds` volontairement **pas installés** (lazy-import only
dans `delineation.py`, jamais exercés). `.venv/bin/python` est un shim
POSIX local pointant vers l'interpréteur conda — nécessaire pour que les
sous-process lancés depuis du code Python (ex. `promote_model` →
`subprocess.run(["dvc", ...])`) retrouvent le bon interpréteur/PATH sans
shell déjà activé. **Piège Windows** : ce shim (shebang POSIX) ne peut être
invoqué que depuis un shell qui le sait interpréter (Git Bash) — `dvc
repro` échoue si son moteur d'exécution interne passe par `cmd.exe`
(observé : `dvc.lock` a dû être régénéré en exécutant chaque stage via Git
Bash + `dvc commit -f`, pas via `dvc repro` directement, sur cette machine).
Voir README `## Installation` pour les commandes exactes.

## Travail en équipe (DagsHub)

Dépôt hébergé sur `https://dagshub.com/Sebastien6631/Hydro-Projet` — git
ET remote DVC au même endroit. `git clone` + `dvc pull` pour tout
récupérer (code + données statiques + modèles). Chaque personne configure
son propre token DagsHub dans `.dvc/config.local` (gitignoré, jamais
partagé) : `dvc remote modify dagshub --local user ... --local password
...`.

**`promote_model` (appelé par `train_one`) committe/tague en LOCAL
uniquement** (`dvc add` + `git add` + `git commit` + `git tag`, jamais de
push automatique, cf. `model/pipeline/promotion.py:82-85`) — après un
entraînement, il faut penser à `git push` + `dvc push` à la main pour
partager avec l'équipe. Les tests `slow`
(`tests/integration/test_train_predict_e2e.py`) exercent ce vrai mécanisme
de promotion (pas un mock) : relancer `pytest -m slow` crée un nouveau
commit+tag réel (v2, v3, ...) sur la branche courante à chaque fois — voulu
pédagogiquement, pas un bug à corriger.

Un remote git peut avoir plusieurs URLs de push simultanées
(`git remote set-url --add --push origin <url2>`) — utilisé ici pour que
`git push origin <branche>` mette à jour GitHub ET DagsHub en un seul
geste. Le remote `company` (`ssh://git@git.cap.dev.hydrospot.fr:...`,
serveur de l'entreprise) reste configuré séparément dans ce dépôt local —
ne jamais y pousser depuis ce contexte projet de cours.

Les onglets spécialisés "Data"/"Models" de l'UI DagsHub ne reflètent pas
forcément tout ce qui est DVC-tracké : "Models" semble lié à leur registre
MLflow hébergé (vide ici, MLflow retiré du projet) ; "Data" a souvent une
fonctionnalité "Data Engine" séparée nécessitant un enregistrement
explicite. L'explorateur de fichiers classique + `dvc pull` restent la
source de vérité fonctionnelle, indépendamment de ces onglets.

## Mode de prédiction `source="live"|"frozen"`

`run_prediction(..., source="live")` (`predict_orchestrator.py`) et
`load_prediction_window(..., source="live")` (`predict_window.py`) :
- `"live"` (défaut, comportement historique) : `build_dossier` — débit à
  jour via Hub'Eau, météo dégradée à 0/NaN (plus d'acquisition FTP, en
  attendant un futur sous-projet API météo publique). `find_record(dossier)`
  est appelé, donc un dossier absent de `config-general.json` lève bien
  `ValueError` (comme avant).
- `"frozen"` : relit `centrales/<dossier>/data_preparation.csv` tel quel,
  ancre `now` sur `df.index.max()` (dernière ligne connue) plutôt que sur
  l'horloge murale — 100% reproductible, aucun appel réseau. `find_record`
  n'est PAS appelé dans cette branche (divergence mineure acceptée : un
  dossier absent de `config-general.json` ne lève pas d'erreur en mode
  frozen s'il a un `data_preparation.csv`). Utilisé par les tests
  d'intégration `slow`.

`predict-archive.py` (production) garde `source="live"` par défaut, ne
passe jamais `source=` explicitement.

## Onboarding BV — idempotence

`bv.json` déjà calculé pour les 2 centrales (shapefile connu pour chacune,
`bv.json` existe déjà (`--force` pour recalculer). Le repli géométrique pur
(plus de préférence pour la grille NWP 2021, fonction retirée) s'applique
systématiquement si on relance en mode `--force` sans shapefile.

## Pipeline DVC (`dvc/preprocessing/dvc.yaml`, 3 stages)

```
debit (Hub'Eau, always_changed) ──> onboarding_check (always_changed) ──> data_preparation (always_changed)
```

`data_preparation` ne dépend plus du marker `puissance` (stage supprimé,
donnée figée sans remplacement de suivi). Dépend de
`src/previ_r2d2/preprocessing/meteo/nwp_reader.py` spécifiquement (pas tout
le dossier `meteo/`, qui n'a plus que ce fichier). Les colonnes météo de
`data_preparation.csv` restent figées (plus de FTP pour les rallonger) ;
seules débit/amont sont réellement rafraîchies par ce stage.

`dvc/model/dvc.yaml` a été SUPPRIMÉ le 2026-09-09 (ses 2 seuls stages,
train_new/train_monthly, retirés faute de cron pour les déclencher) --
l'entraînement se lance désormais uniquement à la main via
`cron/scripts/train.py --dossier`. `dvc/postprocessing/dvc.yaml`
(predict_archive) reste structurellement inchangé (leurs stages/deps/outs restaient valides
tels quels), seuls des commentaires stales mentionnant `puissance`/
`debit_automate` y ont été corrigés.

**`dvc.lock` doit rester synchronisé avec la réalité** — vérifier après
tout changement de `dvc.yaml` que `dvc.lock` ne référence plus de stage
supprimé (piège réel rencontré : `dvc.lock` a longtemps décrit encore
l'ancien pipeline à 7 stages après la réduction à 4, jusqu'à la revue
finale de la simplification).

## Chantier meta-learner + contexte BiLSTM (2026-09-09) — PORTÉ PUIS REPLIÉ

Deux fixes du skill d'origine (`previ-r2d2`) ont été portés ici, mesurés, puis
**retirés parce qu'ils dégradaient ou n'apportaient rien**. Ne pas les re-porter
sans relire cette section.

### Résultat 1 : calibration de l'alpha Ridge — NUISIBLE, retirée

Le projet d'origine calibre `alpha` sur la tranche val (`RIDGE_ALPHA_GRID`) puis
refit sur fit+val, au lieu de l'alpha 1.0 figé. Porté puis mesuré par une
**ablation à modèles de base identiques** (caches OOF/LGBM partagés entre les
deux bras, donc BiLSTM au bit près identique ; le Ridge étant une solution
fermée, la comparaison est déterministe, pas bruitée) :

| `apas_G1_G4` h8 | Stacking |
|---|---|
| alpha 1.0 figé, fit 100% (origine) | **0.9442** |
| alpha calibré sur val, fit fit+val | 0.9208 |

`touzac_g2_G2` : 0.9067 → 0.9029. Même sens. L'alpha retenu était en plus très
instable d'un run à l'autre (0.1, 10.0, 30.0). Une tranche val contiguë unique
est un signal de sélection trop faible ici. **`fit_stacking` est donc revenu au
comportement d'origine** : scaler et Ridge fittés sur 100%, `alpha=1.0`.

Ce qui a été GARDÉ de ce chantier : `training_curve` (`kge_fit`/`kge_val`/
`kge_test`) calculé par un **modèle diagnostic jetable** fitté sur la seule
tranche fit — les métriques restent hors échantillon sans que le modèle déployé
change. `PerStepLGBMMeta` reste en place pour `meta_type="lgbm"` (qui a besoin
d'une tranche val pour son early stopping) mais **n'a pas été mesuré**.

### Résultat 2 : contexte long du BiLSTM — VERDICT IMPOSSIBLE, retiré

Ajouter `lgbm_result["top_features"]` aux colonnes de séquence (24 → 62-68
colonnes) a d'abord semblé nuisible, puis s'est révélé **non concluant** :

**Le BiLSTM variait de ~0.14 KGE d'un run à l'autre à configuration identique**
(`apas` : 0.8392 / 0.8393 / 0.6973 ; `touzac` : 0.8744 / 0.8744 / 0.7408) —
aucune graine n'était fixée. Les valeurs mesurées avec contexte (0.7733 /
0.6563) tombaient DANS cette bande : l'effet était noyé dans le bruit.

Retiré par YAGNI (38 colonnes d'entrée en plus pour un gain non démontré), pas
parce qu'il serait prouvé nuisible. **Les graines sont fixées depuis
(`model/seeding.py`), donc cette piste est désormais mesurable** — la rouvrir
demande juste de comparer deux runs à graine identique.

Deux pièges rencontrés en le portant, à connaître si on y revient :
- Les features d'ingénierie ne sont PAS dans `data_preparation.csv` (elles
  vivent dans `X_train`) : il faut les joindre côté entraînement
  (`build_features`) ET côté prédiction (`build_future_features`, qui garde les
  `horizon` dernières lignes même NaN). Le skill d'origine affirme « aucun
  changement côté predict_orchestrator » — faux ici, c'est un `KeyError` garanti.
- Le bloc futur de la fenêtre ne neutralise que `target_col`. Y laisser entrer
  une dérivée du débit CIBLE (`debit_baseflow_7j`, `max_debit_vu`...) est une
  FUITE : elle y encode la réponse. L'amont et ses gradients, eux, sont des
  covariables futures légitimes (décalés du transit).

### Ce qui est resté du chantier

- `model/pipeline/split.py::train_val_test_indices` (utilisé par `fit_final` et
  le diagnostic meta).
- `fit_final` : fit sur 80% avec early stopping et `eval_sample_weight`
  explicite. **Effet non isolé** — jamais comparé dans une ablation dédiée.
- `results.json` : `training_curves`, `meta_alpha`, `seed`, et
  `evaluation_window` (`data_rows`/`n_train`/`n_test`/`test_start`/`test_end`).
- `model/seeding.py::set_seeds()` — appelé en tête de `run_training`, fixe
  random/numpy/torch (CPU et CUDA), surchargeable par `PREVI_SEED`. Ne force
  PAS `torch.use_deterministic_algorithms(True)` : les noyaux LSTM cuDNN n'ont
  pas d'implémentation déterministe et lèveraient une erreur. Pour MESURER la
  variance résiduelle plutôt que la subir, relancer avec `PREVI_SEED` différent.

### Piège de méthode : comparer deux `results.json`

`split_train_test` est POSITIONNEL (20% de fin). Deux entraînements sur des CSV
de longueurs différentes ne mesurent pas la même période — et les `results.json`
antérieurs au 2026-09-09 n'enregistrent AUCUNE trace de leur fenêtre. Ne jamais
comparer un KGE de prod à un KGE fraîchement entraîné sans vérifier
`evaluation_window` des deux côtés. Avant le 2026-09-09 s'y ajoutait la variance BiLSTM
ci-dessus ; depuis que les graines sont fixées, deux runs à configuration
identique donnent un résultat identique au bit près (vérifié).

En revanche, la comparaison candidat/production faite par
`evaluate_candidate_vs_production` a TOUJOURS été rigoureuse : elle réévalue le
modèle de prod sur le `_eval_context` du candidat, donc sur exactement les
mêmes lignes de test. C'est la lecture manuelle des `kge_stacking` figés dans
d'anciens `results.json` qui ne l'était pas.

## GPU (ajouté 2026-09-09)

`previ_r2d2/model/device.py::resolve_device()` est le **seul** endroit qui
choisit le device — entraînement (`fit_oof`) et prédiction
(`load_trained_models`) l'appellent tous les deux, pour qu'ils ne puissent pas
diverger. Détection auto, surchargeable par `PREVI_DEVICE=cpu|cuda` (repli sur
CPU avec avertissement si `cuda` est demandé sans GPU utilisable). Le device
retenu est loggé au lancement.

- **Le build torch compte plus que le GPU.** `torch==2.12.1+cpu` n'a AUCUN
  kernel CUDA (`torch.version.cuda is None`) : `torch.cuda.is_available()` est
  `False` même avec une carte présente. Il faut réinstaller le build CUDA.
- **Blackwell (RTX 50xx, sm_120) exige cu130.** Pour torch 2.12.1/cp311/Windows
  seuls `cu126` et `cu130` existent ; `cu126` ne contient pas les kernels
  sm_120 et donnerait `no kernel image is available for execution on the
  device`. Vérifier avec `torch.cuda.get_device_capability()` (doit rendre
  `(12, 0)`), pas seulement `is_available()`.
  ```powershell
  pip install --force-reinstall torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
  ```
- **`fit_oof` laisse le modèle sur son device d'entraînement.** Tout appel
  direct à `model(tensor)` doit donc aligner le tenseur (sémantique torch
  normale). Les chemins du projet sont device-aware :
  `BiLSTMHydro._forward_batched` résout `next(self.parameters()).device` et
  ramène le résultat sur CPU ; `predict()` et `get_attention_weights()` passent
  par là. Sans ça : `RuntimeError: Input and parameter tensors are not at the
  same device` — invisible tant que tout reste CPU, systématique dès qu'un GPU
  est présent.
- **Évaluation par batches** (`EVAL_BATCH_SIZE=512`) : évaluer plusieurs
  milliers de séquences en un seul tenseur saturait la VRAM d'un GPU 8 Go.
  Aucun effet sur les résultats (l'inférence est indépendante d'une séquence à
  l'autre). Pic mesuré : ~200 Mo.
- **Comparer un résultat CPU et un résultat GPU** demande une tolérance
  (`rtol=1e-4, atol=1e-6`) : cuDNN/cuBLAS n'ont pas le même ordre de sommation
  flottante que le BLAS CPU, écart ~1e-7. Normal, pas un bug.
- **LightGBM reste 100% CPU** — jamais buildé pour GPU ici. Le GPU n'accélère
  que le BiLSTM.

## Météo : API Météo-France (2026-09-10)

Les fichiers NWP ECMWF n'étaient plus alimentés (FTP retiré) : `build_dossier`
rendait une fenêtre SANS aucune colonne `_S`, `station_count` valait 0 et
`compute_meteo_hydro_features` mourait sur `pd.concat([])`. La prédiction
`source="live"` était donc cassée ; seul `"frozen"` marchait.

`preprocessing/meteo/open_meteo.py` interroge Open-Meteo (modèles
Météo-France), sans clé d'API ni nouvelle dépendance, **un appel par point**
des 7 `stations_meteo_nwp` de `bv.json`.

**Trois pièges, tous vérifiés par test** :
- `temperature_S{i}` doit être en **KELVIN** (`et0.py` fait `t - 273.15`) ;
  l'API rend des °C.
- `precipitation_S{i}` doit être **CUMULÉE** (`snow.py` fait
  `.diff(1).clip(lower=0)`) ; l'API rend des incréments horaires.
- **Ne PAS utiliser l'archive ERA5** malgré ses 5,6 ans : mesurée sur 504 h de
  recouvrement, elle donne TROIS FOIS plus de pluie que Météo-France
  (0.136 vs 0.049 mm/h, corr 0.21) pour une température quasi identique
  (+0.31 °C, corr 0.95). On utilise `historical-forecast-api` en Météo-France,
  homogène, depuis le **2022-11-15** (bissecté) : 3,8 ans sans trou.

**`niveau0` (isotherme 0°) n'existe plus** : aucun modèle Météo-France ne
l'expose, ni l'archive ERA5, donc l'historique serait irreconstituable. Il
servait à la partition pluie/neige ET à `t_moyen`. Les deux passent désormais
par `snow.bv_temperature` : température 2 m réelle corrigée de l'écart
d'altitude point -> bassin (`altitude_S{i}`, rendue par l'API). Sémantique
identique (`t_moyen > 0` <=> ancien « isotherme au-dessus du BV »), et les
tests à valeurs calculées à la main passent inchangés, ce qui le prouve.

**Toute la chaîne de création du `bv.json` a été supprimée** (`preprocessing/bv/`,
`centrales_calibration.json`, et les dépendances `pysheds`/`rasterio`/
`geopandas`/`shapely`/`pyproj`) : les `bv.json` des 2 centrales sont figés, on
est hors production. `bv.json` reste LU par l'entraînement et la prédiction.
Deux fonctions ont été rapatriées avant la suppression : `haversine_km` dans
`model/features/meteo_hydro.py` et `bv_json_path` inliné dans `dossier_window.py`.

### Pièges rencontrés en rebranchant la chaîne (2026-09-10)

- **Console Windows en cp1252** : `maj-data.py` mourait sur son propre `print`
  d'en-tête (`→`, `—`, `✗` sont hors cp1252), donc AUCUN débit n'était importé.
  `common/console.py::force_utf8()` est appelé en tête du `main()` des 5
  scripts cron. Corriger les caractères un par un ne tiendrait pas.
- **`build-data-preparation` FUSIONNE avec le CSV existant** : les anciennes
  colonnes survivent (`niveau0` réapparaissait) et un point météo en échec
  garde ses valeurs du run précédent. Pour un changement de schéma, SUPPRIMER
  les `data_preparation.csv` avant `--full-history`.
- **L'API rend l'axe temporel demandé EN ENTIER**, avec des `null` là où le
  modèle n'a rien (`past_days=92` sur un modèle qui n'archive que 60 jours).
  Sans `dropna` avant fusion, ces `null` gagnent le recouvrement (`keep="last"`)
  et EFFACENT les vraies valeurs de l'archive -- 32 jours perdus en silence.
- **`FULL_HISTORY_START` doit valoir la date de début de la météo.** Le débit
  remonte à 2021, mais une ligne sans météo est inexploitable : démarrer avant
  décale le split 80/20 vers un passé vide. Mesuré : meta-learner à 1175
  échantillons au lieu de ~19 650, `kge_stacking` à **-6.6e7**.
- **`dvc repro` ne nettoie pas `dvc.lock`** : il met à jour les stages présents
  dans `dvc.yaml` mais laisse les entrées orphelines (le stage `bv` supprimé y
  survivait). Retirer le bloc à la main après toute suppression de stage.
- **`dvc repro` marche à nouveau sur Windows** depuis que les `cmd` utilisent
  `python` et non le shim `.venv/bin/python` -- à condition que l'env conda
  soit activé (le sous-shell de DVC résout `python` via le PATH).
- **Changer le schéma météo casse les anciens modèles** : le BiLSTM attend un
  `n_features` figé (`[256, 29]` contre `[256, 22]`), donc
  `evaluate_candidate_vs_production` lève un `size mismatch`. Retirer les
  modèles incompatibles de `models/` fait basculer en `first_training`.

## Pièges connus (toujours valides après simplification)

- **`_read_source` (`puissance_store.py`, conservé)** : vérifier
  `pd.api.types.is_datetime64_any_dtype(df["Date"])`, pas `dtype ==
  "object"` — pandas 3.x peut retomber sur un dtype `str` (pas `object`)
  quand `parse_dates` échoue sur une ligne corrompue.
- **Horodatages `Z` (UTC)** : toujours relire avec
  `pd.to_datetime(col, utc=True).dt.tz_localize(None)` pour revenir
  tz-naive (comparer tz-aware/tz-naive lève `TypeError`). Vrai pour
  `read_debit_csv`, `read_data_preparation_csv`, etc.
- **Hub'Eau/eaufrance.fr a des pannes fréquentes** (timeouts, 500/504) — un
  échec de `maj-data.py` n'est pas forcément un bug.
- **`flex_strategy` vide (`""`) ou absent** → traité comme invalide par
  `validation.py` (`VALID_FLEX_STRATEGIES = ("DEFAULT",)` seulement
  désormais, plus de `HAUTE_CHUTE`) — un raccordement mal formé est ignoré
  silencieusement par les scripts qui filtrent sur `DEFAULT`.
- **`config.ROOT`** (`common/config.py`) se calcule via
  `Path(__file__).resolve().parents[N]` — recalculer l'index si ce fichier
  est déplacé.
- **`.gitignore` : `**/secret_config*.py`** ignore aussi un futur
  `secret_config.example.py` — vérifier avant d'en créer un.
- **Fidélité stacking** : l'ordre des colonnes de `meta_X`/`meta_X_future`
  doit correspondre EXACTEMENT à `base_cols` de `build_meta_features`
  (`[lgbm(H), lstm(H), divergence, q_now_log, sin, cos, crue_flag, trend,
  meteo]`) — `model/architectures/stacking.py`.
- **Fuseau horaire** : tout le stockage interne (CSV débit/puissance,
  calculs) reste en UTC — la conversion Europe/Paris ne se fait qu'à
  l'affichage final (`to_display_timezone` dans `predict_orchestrator.py`),
  jamais en amont.
- **Ne jamais créer un symlink d'un chemin vers lui-même** — piège
  découvert en supprimant la couche NAS : si deux variables de config
  finissent par pointer vers le même chemin réel (ex. ancien
  `NAS_DATA_ROOT` vs `CENTRALES_DIR`), une fonction de symlink "au cas où"
  peut supprimer le fichier réel puis le remplacer par un lien mort vers
  lui-même.

- **La promotion est EXPLICITE depuis le 2026-09-09** — `train.py` entraîne,
  évalue et écrit le candidat dans `weights/hybrid_candidate/`, mais ne promeut
  QUE si `--promote` est passé. `--dossier` est désormais obligatoire (les
  modes automatisés `--mode new|monthly` ont été retirés en même temps que
  `dvc/model/dvc.yaml`), donc AUCUN entraînement ne promeut sans geste explicite.
  `promote_model` refuse en plus de tourner si l'arbre git n'est pas propre
  (sinon son commit de promotion embarquerait des modifications sans rapport).
- **`promote_model` invoque `python -m dvc`, pas `dvc`** — sur Windows,
  `dvc.exe` vit dans le `Scripts\` de l'env conda et n'est sur le PATH que si
  l'environnement est activé. Un `dvc` nu lève `FileNotFoundError (WinError 2)`
  au message opaque dès qu'on lance le script autrement (shell non activé,
  ordonnanceur). L'interpréteur courant, lui, est toujours le bon.
- **`git stash --include-untracked` avale les données DVC** — `centrales/<dossier>/`
  n'est ni suivi par git ni ignoré au sens strict : un stash `-u` l'emporte et le
  `pop` ne le rend pas toujours. Utiliser `git stash push` nu (fichiers suivis
  seulement) ; réparer avec `git checkout -- centrales/<d>.dvc && dvc pull
  centrales/<d>.dvc`.
- **Comparer un KGE sur une tranche quasi constante** — `np.std(...) == 0` est un
  test trop strict après `expm1` : une cible constante laisse un std résiduel
  d'arrondi (~1e-15) qui passe le test et produit un KGE fini de l'ordre de
  −1e6, assez pour piloter une calibration et fuiter dans `results.json`.
  Utiliser un seuil RELATIF (`std <= 1e-9 * max(1, |moyenne|)`), cf.
  `stacking_fit._kge_m3s`.
- **`.venv/bin/python` n'existe pas sur un clone frais** — c'est un shim créé à
  la main, pas versionné. Sur une machine neuve, activer l'env conda
  (`conda activate projet-mlops`) et utiliser `python` tout court ; les chemins
  `.venv/bin/python` des `dvc.yaml` supposent que le shim a été recréé.

## Commandes de test rapide

```bash
python cron/scripts/maj-data.py --dossier apas_G1_G4      # test ciblé débit (Hub'Eau réel)
python cron/scripts/onboarding-check.py                   # tous les raccordements (pas de --dossier)
python cron/scripts/build-data-preparation.py --dossier apas_G1_G4
python cron/scripts/train.py --dossier touzac_g2_G2 --horizon 8 --force  # entraînement réel réduit
python cron/scripts/predict-archive.py                     # prédit + archive toutes les centrales avec modèle en prod
python -m pytest tests/ -q                                 # suite rapide (342 passed, 2 deselected)
python -m pytest tests/integration/ -v -m slow             # tests réels lents (entraînement + prédiction, ~30 min)
```

`onboarding-check.py`/`predict-archive.py` n'ont **aucun** flag `--dossier`
— ils bouclent sur tous les raccordements découverts.

## Références

- Design/plan de la simplification (non versionnés,
  `docs/superpowers/{specs,plans}/2026-07-23-simplification-projet-cours*.md`)
  — historique complet des décisions, à lire en local si besoin de contexte
  sur *pourquoi* telle chose a été retirée/gardée.
- README.md — installation, pipeline DVC, description script par script,
  travail en équipe DagsHub.
- Hub'Eau hydrométrie : https://hubeau.eaufrance.fr/page/api-hydrometrie
