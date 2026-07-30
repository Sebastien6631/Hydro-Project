---
name: hydro-projet
description: |
  Architecture, pipeline DVC et pièges connus de previ-R2-D2 — VERSION PROJET
  DE COURS MLOps (dépôt DagsHub, 3 centrales, hors réseau de l'entreprise).
  Utilise ce skill quand l'utilisateur parle de : previ-R2-D2, ce projet de
  cours, apas_G1_G4/nancy_A/touzac_g2_G2, dvc.yaml, source=live|frozen,
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
  l'utilise (les 3 sont `flex_strategy: DEFAULT`).
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
publique), l'onboarding BV (idempotent, no-op pour les 3 centrales déjà
onboardées), l'entraînement (LightGBM+BiLSTM+Stacking, intact), la
prédiction (avec un nouveau mode `source="frozen"` 100% offline en plus du
mode `"live"` historique).

## Périmètre : 3 centrales seulement

`apas_G1_G4`, `nancy_A`, `touzac_g2_G2` — toutes `flex_strategy: DEFAULT`.
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
│                                #   jamais exercés pour les 3 centrales déjà onboardées)
├── run.py                        # CLI manuelle expés (--train --dossier/--all-dossiers)
├── models/                        # source de vérité modèles en PRODUCTION, versionné DVC+git
│                                  #   <dossier>/h<horizon>/{version.json, meta_config.json, ...}
├── weights/hybrid{,_candidate}/   # zone de travail manuelle / candidat en cours d'évaluation
├── src/previ_r2d2/
│   ├── common/                  # config.py, dvc_markers.py, onegate.py/mailer.py/
│   │                            #   daily_report.py/mlflow_tracker.py SUPPRIMÉS
│   ├── preprocessing/
│   │   ├── debit/                # eaufrance.py, hubeau.py, hydro_export.py, station_store.py,
│   │   │                         #   debit_csv.py (read_debit_csv, relocalisé depuis automate/)
│   │   ├── puissance/             # puissance_store.py TRIMMÉ (mapping seulement,
│   │   │                         #   export_puissance_csv/cleaning.py/consignes.py supprimés)
│   │   ├── meteo/                 # nwp_reader.py SEUL (nwp_ftp.py/retention.py supprimés)
│   │   ├── data_preparation/      # dossier_window.py (build_dossier), data_preparation_csv.py
│   │   ├── onboarding/            # validation.py (missing_fields, DEFAULT seulement -- pas
│   │   │                         #   de HAUTE_CHUTE, load_sync_config retiré)
│   │   └── bv/                   # rules.py, delineation.py, bv_builder.py, transit.py
│   │                             #   (historical_grid_points/2021-grid RETIRÉ, géométrique pur)
│   ├── model/                     # INTACT -- features/, architectures/{lightgbm,bilstm,stacking},
│   │                             #   pipeline/{orchestrator,predict_orchestrator,promotion,...}
│   ├── cli.py
│   └── postprocessing/           # archive.py (ARCHIVE_ROOT, plus NAS_ARCHIVE_ROOT)
├── cron/scripts/                 # maj-data.py, onboarding-bv.py, onboarding-check.py,
│                                  #   build-data-preparation.py, train.py, predict-archive.py
│                                  #   (majdata-memo/maj-automate/maj-puissance/maj-meteo/
│                                  #    clean-meteo/daily-sync-report SUPPRIMÉS ; cron/wrappers/
│                                  #    SUPPRIMÉ -- lancement manuel uniquement)
├── dvc/
│   ├── preprocessing/dvc.yaml    # 4 stages : debit -> onboarding_check -> bv -> data_preparation
│   ├── model/dvc.yaml            # train_new, train_monthly (inchangé)
│   └── postprocessing/dvc.yaml   # predict_archive (inchangé)
├── config/
│   ├── centrales/<dossier>/      # vide, .gitkeep (jamais peuplé)
│   ├── bv_mapping.yaml           # 3 entrées (apas_G1_G4, nancy_A, touzac_g2_G2)
│   └── puissance_mapping.yaml    # 2 entrées (nancy_A, touzac_g2_G2 -- apas via heuristique)
├── centrales/                    # gitignoré (sauf .dvc) -- DVC-tracké, remote DagsHub
│   ├── REFERENCE/                # config-general.json.dvc, shapefiles.dvc, bv_rules.json,
│   │                             #   centrales_calibration.json (git-trackés directement),
│   │                             #   files/ (gros GIS annexes, MNT France -- NI git NI DVC,
│   │                             #   jamais utilisé : les 3 centrales ont toutes un shapefile)
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

`bv.json` déjà calculé pour les 3 centrales (shapefile connu pour chacune,
jamais le repli MNT). `onboarding-bv.py batch` est un no-op tant qu'un
`bv.json` existe déjà (`--force` pour recalculer). Le repli géométrique pur
(plus de préférence pour la grille NWP 2021, fonction retirée) s'applique
systématiquement si on relance en mode `--force` sans shapefile.

## Pipeline DVC (`dvc/preprocessing/dvc.yaml`, 4 stages)

```
debit (Hub'Eau, always_changed) ──> onboarding_check (always_changed) ──> bv ──> data_preparation (always_changed)
```

`data_preparation` ne dépend plus du marker `puissance` (stage supprimé,
donnée figée sans remplacement de suivi). Dépend de
`src/previ_r2d2/preprocessing/meteo/nwp_reader.py` spécifiquement (pas tout
le dossier `meteo/`, qui n'a plus que ce fichier). Les colonnes météo de
`data_preparation.csv` restent figées (plus de FTP pour les rallonger) ;
seules débit/amont sont réellement rafraîchies par ce stage.

`dvc/model/dvc.yaml` (train_new/train_monthly) et
`dvc/postprocessing/dvc.yaml` (predict_archive) : structurellement
inchangés par la simplification (leurs stages/deps/outs restaient valides
tels quels), seuls des commentaires stales mentionnant `puissance`/
`debit_automate` y ont été corrigés.

**`dvc.lock` doit rester synchronisé avec la réalité** — vérifier après
tout changement de `dvc.yaml` que `dvc.lock` ne référence plus de stage
supprimé (piège réel rencontré : `dvc.lock` a longtemps décrit encore
l'ancien pipeline à 7 stages après la réduction à 4, jusqu'à la revue
finale de la simplification).

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

## Commandes de test rapide

```bash
.venv/bin/python cron/scripts/maj-data.py --dossier apas_G1_G4      # test ciblé débit (Hub'Eau réel)
.venv/bin/python cron/scripts/onboarding-check.py                   # tous les raccordements (pas de --dossier)
.venv/bin/python cron/scripts/onboarding-bv.py single --dossier apas_G1_G4  # no-op si bv.json déjà présent
.venv/bin/python cron/scripts/build-data-preparation.py --dossier apas_G1_G4
.venv/bin/python cron/scripts/train.py --dossier touzac_g2_G2 --horizon 8 --force  # entraînement réel réduit
.venv/bin/python cron/scripts/predict-archive.py                     # prédit + archive toutes les centrales avec modèle en prod
.venv/bin/python -m pytest tests/ -q                                 # suite rapide (315 passed, 2 deselected)
.venv/bin/python -m pytest tests/integration/ -v -m slow             # tests réels lents (entraînement + prédiction, ~30 min)
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
