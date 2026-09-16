# projet_hydro (pipeline IA)

> **Version projet de cours MLOps** — ce dépôt est une version réduite de
> previ-R2-D2 (pipeline de production chez Barthe EnR), adaptée pour un
> projet de cours réalisé hors du réseau de l'entreprise. Périmètre réduit
> à 2 centrales (`apas_G1_G4`, `touzac_g2_G2`, toutes
> `flex_strategy: DEFAULT`). Les modules suivants, inaccessibles hors
> serveur de production, ont été retirés :
> - **OneGate** (`memorandum.py`) — API interne de structure des centrales ;
>   `config-general.json`/`config-raccordement.json` des 2 centrales sont
>   désormais des données statiques, versionnées via DVC.
> - **hydrospot_stream** (`preprocessing/puissance/`) — source de puissance
>   sur NAS ; `puissance.csv`/`puissance_horaire.csv` figés, idem.
>   le 2026-09-10 par l'API Météo-France (`preprocessing/meteo/open_meteo.py`),
>   qui rend le même contrat de colonnes sans fichiers locaux ; ancien sous-projet
>   séparé de remplacement par une API météo publique.
> - **automate** (rsync/SSH, `HAUTE_CHUTE`) — aucune des 2 centrales
>   gardées n'utilise cette stratégie.
> - **Mail/digest quotidien** — hors périmètre (lancement manuel des scripts).
>   **MLflow** a été refait proprement en phase 2 : voir la section dédiée.
>
> Reste pleinement fonctionnel : le débit (Hub'Eau/eaufrance, API
> publique), l'onboarding BV, l'entraînement et la prédiction (mode
> `source="frozen"` disponible pour une prédiction 100% reproductible sans
> réseau, en plus du mode `source="live"` par défaut).

Pipeline IA de prévision hydrologique : collecte des débits (eaufrance /
Hub'Eau), caractérisation du bassin versant, entraînement et prédiction du
modèle hybride (LightGBM + BiLSTM + stacking), orchestrés via DVC.

## Architecture

```
projet_hydro/
├── pyproject.toml               # package installable (pip install -e .)
├── requirements.txt
├── run.py                        # CLI entraînement/prédiction du modèle hybride (--train ...)
├── config/
│   ├── centrales/                # réservé (vide, .gitkeep) -- inutilisé dans cette version
│   └── puissance_mapping.yaml   # dossier -> nom de dossier hydrospot_stream (repli explicite, cf. ci-dessous)
├── src/projet_hydro/
│   ├── common/                  # config, secret_config (3 secrets restants), dvc_markers
│   ├── preprocessing/
│   │   ├── onboarding/          # validation.py -- complétude config-raccordement.json avant bv
│   │   ├── debit/                # eaufrance, Hub'Eau, stockage local (centrales/<dossier>/) + dédup
│   │   ├── puissance/            # puissance_store.py seul : résolution du dossier hydrospot_stream
│   │   │                        #   (mapping/heuristique, utilisée par la validation d'onboarding) --
│   │   │                        #   import/fusion réels (NAS) retirés, puissance*.csv figés
│   │   ├── meteo/                 # open_meteo.py seul (API Météo-France)
│   │   ├── data_preparation/      # Data_Preparation : débit + météo API + amont brut
│   │   └── bv/                   # onboarding BV : bassin versant, stations hydrométriques, transit
│   ├── model/
│   │   ├── features/             # et0, snow, meteo_hydro, debit_autoregressif, amont —
│   │   │                         #   feature engineering du modèle hybride, port fidèle de feature()
│   │   ├── architectures/
│   │   │   ├── lightgbm/          # metrics, features, training (Optuna+OOF+final), predict, explain (SHAP)
│   │   │   ├── bilstm/            # model.py (BiLSTMHydro, torch), sequences, metrics, explain (attention)
│   │   │   └── stacking.py        # meta-learner Ridge/LGBM (fonctions pures)
│   │   └── pipeline/              # orchestration cross-architecture :
│   │                              #   bv_config, data_loading, oof_cache, stacking_fit, metrics,
│   │                              #   predict, artifacts, plots, orchestrator (run_training),
│   │                              #   predict_orchestrator (run_prediction), predict_window,
│   │                              #   eligibility (12 mois / réentraînement hebdo),
│   │                              #   promotion (comparaison KGE + promotion versionnée DVC+git)
│   ├── cli.py                     # point d'entrée de run.py (--train/--predict, MANUEL uniquement)
│   └── postprocessing/           # archive.py (archivage horaire, local ./ARCHIVE) -- API FastAPI Previ_v2 non portée
├── models/                        # SOURCE DE VÉRITÉ modèles en PRODUCTION, versionné DVC+git
│                                  #   <dossier>/h<horizon>/{version.json, meta_config.json, bilstm.pt, ...}
├── weights/
│   ├── hybrid/<dossier>/h<horizon>/           # zone de travail manuelle (run.py, expés)
│   └── hybrid_candidate/<dossier>/h<horizon>/ # candidat en cours d'évaluation par train.py (auto)
├── ARCHIVE/                        # archive locale des prévisions horaires (gitignored)
├── cron/
│   └── scripts/                  # CLI minces (maj-data, onboarding-check, build-data-preparation,
│                                  #   validate-data, check-meteo, train, predict-archive) --
│                                  #   appelés par dvc repro ET par les DAGs Airflow
├── dvc/
│   ├── preprocessing/dvc.yaml     # debit -> onboarding_check -> data_preparation -> validate
│   ├── model/dvc.yaml             # train (--promote)
│   └── postprocessing/dvc.yaml    # predict_archive (horaire)
├── airflow/
│   ├── Dockerfile                 # FROM projet_hydro + apache-airflow (build app PUIS build airflow)
│   └── dags/                      # hydro_predict (horaire), hydro_train (lundi 02:00) -- cf. §Orchestration
├── outputs/                       # sorties de prédiction/entraînement (gitignored)
├── tests/                         # miroir de src/projet_hydro/
├── centrales/                     # données des 2 centrales -- <dossier>/ versionné via DVC
│   │                              #   (remote local ../remote_dvc, cf. <dossier>.dvc à la racine)
│   ├── REFERENCE/                 # config-general.json, versionné via DVC ;
│   │                              #   files/ (gitignoré, non utilisé dans cette version)
│   └── <dossier>/                 # config-raccordement.json, *.csv, bv.json, data_preparation.csv,
│                                  #   prevision.json
└── data/                          # inutilisé pour l'instant (réservé à un usage futur)
```

## Installation

L'environnement réel est un env conda dédié (`projet-mlops`, Python 3.11,
créé via conda-forge) — pas un simple `venv` + `pip install -r
requirements.txt` (`requirements.txt` est un vestige minimal, non à jour).

```bash
conda create -n projet-mlops python=3.11 -c conda-forge -y
conda activate projet-mlops

pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
pip install requests pandas "numpy<2.4" scikit-learn scipy pyyaml \
    lightgbm==4.6.0 optuna==4.6.0 joblib matplotlib shap shapely pyproj \
    dvc pytest
pip install -e . --no-deps   # --no-deps : cf. note ci-dessous

cp .env.example .env         # même fichier pour l'hôte et Docker -- cf. Configuration
dvc pull   # données statiques des 2 centrales (config-general.json,
           # centrales/<dossier>/...) depuis le remote DVC local
```

> **Note** : les dépendances GIS ont disparu avec la chaîne de délimitation de
> BV (supprimée le 2026-09-10, les `bv.json` étant figés). Les `dvc/*/dvc.yaml`
> invoquent `python` : l'env conda doit donc être activé, `dvc repro` résolvant
> uniquement depuis un shell qui lit un shebang (Git Bash, pas `cmd.exe`).

## Travail en équipe (DagsHub)

Le dépôt (code + données) est partagé via [DagsHub](https://dagshub.com/Sebastien6631/Hydro-Projet)
— un seul endroit pour le git et le remote DVC.

```bash
git clone https://dagshub.com/Sebastien6631/Hydro-Projet.git
cd Hydro-Projet
dvc pull   # récupère les données (config-general.json,
           # centrales/<dossier>/..., modèles entraînés)
```

**Configuration une fois par personne** : chacun crée son propre token
DagsHub (Settings → Tokens sur dagshub.com), puis :
```bash
dvc remote modify dagshub --local user <votre_pseudo_dagshub>
dvc remote modify dagshub --local password <votre_token>
```
(`.dvc/config.local` est gitignoré — jamais commité/partagé.)

**Après un entraînement ou une nouvelle donnée** : `promote_model` fait un
`git commit`/`git tag` **local uniquement**. Pour que l'équipe le
récupère, il faut pousser les deux à la main :
```bash
git push origin <branche>
dvc push
```

## Configuration

Les secrets vivent dans `src/projet_hydro/common/secret_config.py` (Python
local, **non versionné**) :

```python
PREVI_PUISSANCE_SOURCE_ROOT = "..." # racine hydrospot_stream (utilisée seulement pour la résolution
                                     #   du mapping puissance -- import/fusion réels retirés)
PREVI_NAS_METEO = "..."             # racine des fichiers météo NWP bruts (acquisition FTP retirée --
                                     #   dossier vide/absent = colonnes météo vides, dégradation gracieuse)
```

Priorité de lecture : `secret_config.py` > variables d'environnement > défauts. Les secrets
propres aux modules retirés dans cette version (cf. note en tête de fichier) ont disparu
avec eux.

### `.env` — un seul fichier pour l'hôte et Docker

`cp .env.example .env`, puis renseigner. **Tout le monde passe par là**, env
conda comme conteneur : `config.py` charge `.env` au premier import (parseur
stdlib, pas de `python-dotenv`), et `docker compose` lit le même fichier pour
ses substitutions `${VAR}`. Aucune variable à exporter à la main.

Une valeur déjà présente dans l'environnement n'est jamais écrasée
(`setdefault`) : le shell peut surcharger ponctuellement, et dans un conteneur
c'est `x-env` de `docker-compose.yml` qui a le dernier mot.

**Conflit `localhost` / `mlflow`** — `MLFLOW_TRACKING_URI` vaut
`http://localhost:5000` vu de l'hôte, mais dans le réseau compose le serveur
s'appelle `mlflow`. Le conflit est résolu avec la syntaxe `${VAR:+valeur}` de
Compose (« si non vide, remplace par »), qui permet à `.env` de fonctionner
pour Python **et** pour Docker Compose sans variable supplémentaire :

```yaml
MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI:+http://mlflow:5000}
```

`.env` non vide → hôte `localhost:5000`, conteneurs `mlflow:5000`. `.env`
vide ou absent → vide partout, MLflow inerte (défaut des tests et de la CI).

## Pipeline (DVC)

```bash
dvc dag dvc/preprocessing/dvc.yaml       # debit -> onboarding_check -> bv -> data_preparation (manuel)
dvc dag dvc/postprocessing/dvc.yaml      # predict_archive (horaire)
dvc repro dvc/preprocessing/dvc.yaml     # exécute tout ce pilier, dans l'ordre
```

```
debit ──> onboarding_check ──> bv
                                 │
                                 ▼
              (data_preparation : manuel uniquement,
               train.py le rafraîchit lui-même par dossier)
                                 │
                                          │
       dvc/postprocessing/dvc.yaml : predict_archive
```

`debit`, `onboarding_check` et `bv` déclarent chacun un `outs:` minimal
(`logs/dvc_markers/<stage>.json`, `cache: false`) — pas une vraie sortie mise
en cache, juste un marqueur horodaté écrit en une ligne
(`projet_hydro.common.dvc_markers.write(...)`) à la fin de chaque script, pour
donner une vraie arête DAG entre stages (sans ça, DVC n'a rien à quoi
accrocher une dépendance). `debit` et `onboarding_check` gardent en plus
`always_changed: true` (comme `data_preparation` plus bas) — leurs
dépendances déclarées ne suffisent pas à elles seules à détecter un
changement réel (ex. nouveau point Hub'Eau), donc l'exécution est forcée à
chaque `dvc repro`.
`data_preparation` **n'est appelé par rien d'automatisé** — `train.py`
(`dvc/model/dvc.yaml`) le rafraîchit lui-même, dossier par dossier, juste
avant de vérifier l'éligibilité de CE dossier (décision actée : un `foreach`
DVC par dossier aurait nécessité une liste de dossiers maintenue hors du yaml
+ restructurer `train` en stages par dossier pour un lien réellement
significatif — jugé disproportionné). Le stage reste un outil manuel
(`dvc repro dvc.yaml:data_preparation`, backfill groupé), lui aussi
`always_changed: true` (les colonnes débit/amont sont réellement
rafraîchies ; les colonnes météo restent figées, cf. note en tête de fichier).

## Les scripts

Les raccordements (`config-general.json` + `config-raccordement.json` par
dossier) sont désormais des données statiques versionnées via DVC pour les 3
centrales gardées — il n'y a plus de script pour les régénérer depuis une API
externe (cf. note en tête de fichier).

### `maj-data.py` — import / mise à jour des débits

Pour chaque raccordement `flex_strategy == "DEFAULT"` (les 2 centrales gardées
le sont toutes), on traite **la station de référence**
(`station_vigicrue_reference`) **et toutes les stations amont**
(`stations_vigicrue_amont`, codes séparés par des virgules). Pour chaque station :

- code jamais vu ailleurs → **import** complet `01/01/2021 → aujourd'hui` (eaufrance) ;
- code déjà porté par un autre dossier → **symlink** local vers ce fichier réel (dédup, pas de nouvel appel API) ;
- CSV déjà présent pour ce dossier → **mise à jour** incrémentale : Hub'Eau `observations_tr` si plus
  récent que eaufrance (agrégé en moyenne horaire), sinon eaufrance seule.

Sortie : `centrales/<dossier>/<station>.csv` (référence) ou `amont_<station>.csv`
(stations amont) — stockage local direct (`config.NAS_DATA_ROOT ==
config.CENTRALES_DIR` dans cette version, plus de NAS distant).

```bash
python cron/scripts/maj-data.py                  # import/MAJ (API Hub'Eau/eaufrance réelle)
python cron/scripts/maj-data.py --dossier apas_G1_G4   # test ciblé
python cron/scripts/maj-data.py --end 03/07/2026
```

### `build-data-preparation.py` — Data_Preparation (débit + météo + amont brut)

Construit/met à jour, pour chaque centrale, un CSV historique
`data_preparation.csv` combinant débit brut + météo NWP brute + amont brut,
alignés par horodatage horaire — aucune feature (lags, gradients,
transit_amont saisonnier) n'est calculée ici, ça reste un sous-projet
ultérieur (entraînement/prédiction du modèle hybride meta).

**Lancement manuel** avant un entraînement (la cadence dépend des dates
d'entraînement, pas d'une fréquence fixe). Stage DVC `data_preparation` (cf.
section Pipeline), avec une vraie dépendance sur `debit`/`bv` (via
marqueurs) — `dvc repro dvc.yaml:data_preparation` régénère aussi ces
dépendances amont au passage. Seules les colonnes débit/amont sont vraiment
rafraîchies à chaque exécution ; les colonnes météo restent figées (plus
d'acquisition FTP, cf. note en tête de fichier) tant qu'un sous-projet API
météo publique n'est pas ajouté.

- Débit : station de référence Hub'Eau (les 2 centrales gardées sont toutes
  en `flex_strategy == "DEFAULT"`).
- Amont : colonne `debit_amont` (un seul) ou `debit_amont_{code}` (plusieurs,
  dédupliqués), nommage repris de `lightgbm_model.py` (Previ_v2).
- Météo : points de `bv.json.stations_meteo_nwp` (déjà calculés par
  fichiers NWP bruts -- température/précipitation/niveau0° gardés bruts
  (Kelvin, cumul non diffé), la transformation en feature est hors périmètre.
- Historique confirmé (J-1 et avant) uniquement -- la fenêtre temps
  réel/prévision (arbitrage entre runs météo qui se superposent) est hors
  périmètre, réservée à un sous-projet `predict_future_meta` futur.
- Reprise incrémentale : repart juste après le dernier point déjà écrit pour
  cette centrale (pas de fenêtre fixe) ; `--full-history` force un backfill
  complet depuis le début de chaque source.
- `start` est relevé au premier point réel du débit (si postérieur à la
  fenêtre demandée) avant de lire amont/météo -- le débit est la variable
  cible, une donnée météo sans débit en face n'a aucun intérêt pour
  l'entraînement, et ça évite de scanner des années de fichiers NWP pour rien.

**Lecture sans effet de bord** — `debit_source.py`/`amont_source.py` lisent
via le symlink local déjà posé par l'import réel
(`config.CENTRALES_DIR / dossier / station_store.filename_for(...)`),
jamais via `station_store.resolve_nas_path` (qui écrit `_index.json` et peut
créer un symlink local -- effet de bord inapproprié pour un module qui ne fait
que lire une donnée déjà importée par un autre script).

**Piège rencontré (rétention météo en retard)** — dans l'archive météo NWP
figée reprise de la production, un jour donné peut garder toutes ses
échéances (000-360) au lieu de 000-023 seulement (retard ponctuel de
l'ancien pipeline de rétention, retiré dans cette version, cf. note en tête
de fichier), et ses échéances longues (ex. 024) pointent alors vers des
`flow_date` déjà couvertes par les jours suivants -- doublon d'horodatage
bien réel dans l'archive brute (deux runs différents, pas une corruption).
(moyenne les doublons plutôt que de tenter un arbitrage, hors périmètre pour
l'historique confirmé).

```bash
python cron/scripts/build-data-preparation.py                            # les 2 centrales
python cron/scripts/build-data-preparation.py --dossier apas_G1_G4       # test ciblé
python cron/scripts/build-data-preparation.py --full-history             # backfill complet
```

### `onboarding-check.py` — validation d'un raccordement

Pour chaque raccordement sans `bv.json` encore (pas onboardé), valide son
`config-raccordement.json` (`preprocessing/onboarding/validation.py::missing_fields`)
et journalise les infos manquantes. **Idempotent** : peut être relancé sans
risque (relancé manuellement tant que la config n'est pas complète — plus de
détection automatique de nouveaux raccordements dans cette version, cf. note
en tête de fichier). **Isolation par raccordement** : un enregistrement cassé
ne bloque jamais la validation des autres (`try/except` par item).

```bash
python cron/scripts/onboarding-check.py   # tous les raccordements pas encore onboardés (pas de --dossier)
```

### `train.py` — entraînement (éligibilité + promotion conditionnelle)

Sans `--dossier` : boucle sur les centrales onboardées (h8) et n'entraîne que
celles que `is_eligible_for_training` retient — **12 mois d'historique** pour
un premier modèle, **7 jours** depuis le dernier essai pour un réentraînement
(`RETRAIN_INTERVAL_DAYS`, aligné sur le DAG Airflow hebdo). Rien d'éligible →
« Aucune centrale éligible aujourd'hui. » en quelques secondes. La règle vit
dans le code testé, pas dans un planificateur : Airflow (ci-dessous) ne fait
que sonner. `--promote` ne promeut que si le KGE du candidat bat la production
(même holdout, cf. `model/pipeline/promotion.py`).

```bash
python cron/scripts/train.py --dossier apas_G1_G4 --horizon 8            # candidat seul, aucune promotion
python cron/scripts/train.py --dossier apas_G1_G4 --horizon 8 --promote  # promeut si meilleur que la prod
python cron/scripts/train.py --dossier apas_G1_G4 --horizon 8 --force   # test manuel ciblé, ignore l'éligibilité
```

Le modèle en PRODUCTION vit dans `models/<dossier>/h<horizon>/`, versionné
DVC + tag git (`<dossier>-h<horizon>-v<N>`, rollback = `git checkout <tag> &&
dvc pull`) — jamais dans `weights/hybrid/` (zone de travail manuelle) ni
`weights/hybrid_candidate/` (candidat en cours d'évaluation, jamais lu par la
prédiction).

> **Attention (tests `slow`)** — `promote_model` fait un vrai `dvc add` +
> `git add` + `git commit` + `git tag` sur le dépôt courant, pas une
> simulation. Les tests d'intégration marqués `slow`
> (`tests/integration/test_train_predict_e2e.py`) appellent le vrai
> `train_one` -> `promote_model` : les lancer pour de vrai (`pytest -m slow`)
> crée un commit + tag réels sur la branche courante à chaque exécution
> (nouvelle version `v2`, `v3`, ... à chaque relance). C'est volontaire
> (démonstration pédagogique du mécanisme de promotion réel), pas un mock à
> corriger — mais soyez-en conscient avant de lancer la suite `slow` sur une
> branche que vous ne voulez pas polluer de commits. Pensez aussi à
> `git push`/`dvc push` après (cf. section « Travail en équipe » en tête de
> fichier) pour partager le nouveau modèle.

### `predict-archive.py` — prédiction + archivage horaire

Pour chaque (dossier, horizon) ayant un modèle en production : archive le
JSON de prévision existant (`postprocessing/archive.py::archive_previous_json`
— l'heure de production, pas l'heure d'archivage, dans le nom de fichier)
vers `./ARCHIVE/<dossier>/<AAAA>/<MM>/<JJ>/` (local, `config.ARCHIVE_ROOT`),
puis prédit la nouvelle heure et écrit `centrales/<dossier>/prevision.json`.
Isolation par (dossier, horizon).

`run_prediction`/`load_prediction_window` acceptent un paramètre
`source="live"` (défaut, assemble la fenêtre dynamiquement comme
`build-data-preparation.py`) ou `source="frozen"` (relit directement
`data_preparation.csv` tel quel, sans aucun appel réseau) — utile pour une
prédiction 100% reproductible, par exemple en test.

```bash
python cron/scripts/predict-archive.py   # toutes les centrales avec un modèle en prod (pas de --dossier)
```

### `src/projet_hydro/model/` — portage du modèle hybride meta

Fonctions pures (pas de classe à état, sauf `BiLSTMHydro` qui est un
`nn.Module` PyTorch — contrainte du framework, pas un choix indépendant).
Convention de placement : logique propre à une seule architecture (LGBM ou
BiLSTM) → `model/architectures/<nom>/` ; logique qui combine plusieurs
architectures (Stacking, prédiction test set, plots) → `model/pipeline/`.

- `model/features/` — feature engineering, port fidèle de `feature()`
  (`lightgbm_model.py` Previ_v2) : `et0.py`, `snow.py`, `meteo_hydro.py`
  (météo/hydrologie), `debit_autoregressif.py` (lags/gradients/récession/
  baseflow du débit cible), `amont.py` (stations amont + transit saisonnier,
  `shift_amont_columns` réutilisée par l'orchestrateur pour les séquences BiLSTM).
- `model/architectures/lightgbm/` — sous-projet LightGBM (flux "meta" actif
  uniquement, pas le flux "v4 legacy") : `metrics.py` (KGE, quantiles,
  poids, post-traitement), `features.py` (`build_features`), `training.py`
  (sélection de features, tuning Optuna, OOF, entraînement final —
  dépendances `lightgbm`/`optuna`), `predict.py` (`predict_lgbm_full` —
  prédiction sur un DataFrame complet), `explain.py` (SHAP beeswarm/bar).
- `model/architectures/bilstm/` — sous-projet BiLSTM : `metrics.py` (KGE
  différentiable + wrapper numpy), `sequences.py` (fenêtres glissantes,
  sélection des colonnes de séquence), `model.py` (classe `BiLSTMHydro` —
  LSTM bidirectionnel + attention + `fit_oof` + `predict` — dépendance
  `torch`, CPU-only), `explain.py` (heatmaps d'attention sur les pics de crue).
- `model/architectures/stacking.py` — meta-learner Stacking/Ridge : fonctions
  pures (pas de classe — l'API classe de Previ_v2 s'est révélée non utilisée
  en prod), combinent les sorties OOF LightGBM/BiLSTM + features
  contextuelles pour le meta-learner final.
- `model/pipeline/` — **orchestration entraînement**, assemble toutes les
  briques ci-dessus (port de `train_meta.py`/`run_one`, Previ_v2) :
  `bv_config.py` (assembleurs `bv.json` → `bv_params`/`transit_amont`),
  `data_loading.py` (chargement/split train-test), `oof_cache.py`
  (cache disque OOF LGB/BiLSTM), `stacking_fit.py` (fit du meta-learner),
  `predict.py` (`predict_test_set` — prédiction test set sans dupliquer
  `StackingHydroModel.predict()`), `metrics.py` (KGE par pas/régime/saison,
  interprétabilité), `artifacts.py` (assemblage + écriture `results.json`/
  `meta_config.json`/CSV), `plots.py` (7 plots : SHAP, attention, comparaison
  test, KGE-by-step, coefficients Ridge, KGE radar), `orchestrator.py`
  (`run_training` — point d'entrée qui appelle tout dans l'ordre).
- **`model/pipeline/predict_orchestrator.py`** — prédiction opérationnelle
  (`run_prediction`, port de `predict_future_meta.py` Previ_v2) : lit le
  modèle depuis `models/<dossier>/h<horizon>/` (production, jamais
  `weights/hybrid/`), débit station + conversion turbine (`hydraulic.py` —
  puissance/chute/rendement).
- **`model/pipeline/{eligibility,promotion}.py`** — éligibilité (12 mois /
  réentraînement hebdomadaire, 7 j) et promotion versionnée (comparaison KGE candidat
  vs production sur le même holdout, `dvc add`+`git tag`, sûr face à un échec
  partiel). Cf. section `train.py` ci-dessus pour le détail opérationnel.

Voir le skill `projet_hydro` pour le détail pièce par pièce et le skill
`hybrid-meta-ops` pour l'architecture du modèle hybride.

```bash
.venv/bin/python -m pytest tests/model/ -v   # tests du portage modèle
```

### `run.py` — entraînement/prédiction MANUELS (expés, pas le chemin automatisé)

CLI mince (`src/projet_hydro/cli.py`) qui appelle `run_training`/`run_prediction`
pour un dossier+horizon donné, ou pour toutes les centrales × 3 horizons
(8h/48h/72h). **Le chemin opérationnel automatisé passe par
`cron/scripts/train.py`/`predict-archive.py`** (cf. sections dédiées
ci-dessus), pas par `run.py` — celui-ci reste réservé aux tests/expés
manuels (entraîne toujours dans `weights/hybrid/`, jamais `models/`, donc
jamais lu par la prédiction opérationnelle sans passer par une promotion
explicite).

```bash
python run.py --train --dossier apas_G1_G4 --horizon 8
python run.py --train --all-dossiers                      # toutes les centrales (h8)
python run.py --train --dossier apas_G1_G4 --horizon 8 --force-lgbm --force-lstm  # recalcul complet
python run.py --train --dossier apas_G1_G4 --horizon 8 --meta lgbm --epochs 60
```

Sorties : `weights/hybrid/<dossier>/h<horizon>/` (`results.json`,
`meta_config.json`, poids/caches OOF, `plots/`) et
`outputs/hybrid/<dossier>/h<horizon>/` (CSV + PNG de comparaison test).
Chaque paire (dossier, horizon) échoue indépendamment en mode
`--all-dossiers` (log + continue) — code de sortie `1` si au moins un échec.

## Flux normal (bout en bout, débit → prédiction)

```bash
python cron/scripts/maj-data.py                             # 1. importe/complète les débits (Hub'Eau)
python cron/scripts/onboarding-check.py                     # 2. valide les raccordements pas encore onboardés
python cron/scripts/train.py --dossier apas_G1_G4 --horizon 8 --promote   # 4. entraînement + promotion conditionnelle
python cron/scripts/predict-archive.py                      # 6. archive + prédit la nouvelle heure
```

## Notifications

Retirées dans cette version (cf. note en tête de fichier) — lancement
manuel des scripts, pas de digest mail. Le suivi d'expériences, lui, est
revenu : voir « Suivi d'expériences (MLflow) — Phase 2 » en fin de fichier.

## Références

- Hub'Eau hydrométrie : https://hubeau.eaufrance.fr/page/api-hydrometrie

## Conteneurisation (Docker) — Phase 1

Tout tourne dans un conteneur — plus besoin d'installer conda/Python en local.

```bash
cp .env.example .env      # renseigner DAGSHUB_USER + DAGSHUB_TOKEN + GIT_AUTHOR_*
docker compose build     # construit l'image (~5-8 min la 1re fois)

docker compose run --rm app                          # lance la suite de tests
docker compose run --rm app dvc pull                 # données + modèles (DagsHub)
docker compose run --rm app python run.py --train --dossier touzac_g2_G2 --horizon 8
docker compose run --rm app bash                     # shell interactif
```

L'image contient Python 3.11, PyTorch CPU, LightGBM, DVC et le package
`projet_hydro`.

Le code est **bind-monté** : une modif locale est vue immédiatement dans le
conteneur, pas de rebuild sauf changement de dépendances.

## API d'inférence (FastAPI) — Phase 1

Expose le modèle promu en HTTP. Prévision **h8 uniquement**, sur données
figées (`source="frozen"`, 100 % reproductible).

```bash
docker compose up -d nginx        # api n'expose pas de port direct -- via nginx, cf. phases 2.5/3.6
curl -k https://localhost:8443/health
curl -k https://localhost:8443/models
curl -k -X POST https://localhost:8443/predict \
     -H 'content-type: application/json' \
     -d '{"dossier": "touzac_g2_G2"}'
```

| Endpoint | Rôle |
|---|---|
| `GET /health` | état + centrales servies |
| `GET /models` | modèles promus (version, KGE) |
| `POST /predict` | `{dossier}` → prévision débit h8 (série `q_stacking_m3s` / `q_entrant_m3s`) |
| `GET /docs` | Swagger UI |

Le modèle est chargé depuis `models/<dossier>/h8/` (versionné DVC) — faire
`docker compose run --rm app dvc pull` au préalable. Sécurisation (auth,
rate-limit, logs structurés) : phase 3.

## Validation des données (contrat) — Phase 1

`data_preparation.csv` doit respecter un contrat avant d'entraîner. La
vérification est *hand-rolled* (`preprocessing/data_preparation/validation.py`)
et sépare **erreurs** (donnée inexploitable → le pipeline s'arrête) et
**avertissements** (donnée acceptée, à surveiller).

| Niveau | Règles |
|---|---|
| Erreur | fichier vide · colonne `debit_m3s` absente / non numérique / vide · index non temporel, non trié, ou avec doublons · débit négatif |
| Warning | > 10 % de cible manquante · historique < 180 j · trou > 24 h dans l'index horaire · colonne entièrement vide (météo figée) |

```bash
docker compose run --rm app python cron/scripts/validate-data.py
docker compose run --rm app python cron/scripts/validate-data.py --dossier apas_G1_G4 --strict
```

Stage DVC `validate` (dans `dvc/preprocessing/dvc.yaml`, après `data_preparation`) :
le stage `train` en dépend, donc `dvc repro` s'arrête avant l'entraînement si
une erreur de contrat est détectée. Rapport JSON par centrale sous `logs/validation/`.

## Suivi d'expériences (MLflow) — Phase 2

> **État : fait** (tâches 2.1 et 2.2, branche `phase2/mlflow-tracking`).
> Validé le 2026-09-11 sur un entraînement réel : run `touzac_g2_G2-h8` loggé
> avec ses métriques et artefacts, KGE stacking 0,824. Le Model Registry se
> remplit au premier `--promote` qui aboutit.

Chaque entraînement enregistre ses paramètres, ses métriques KGE et ses
artefacts dans MLflow, pour comparer deux entraînements autrement qu'en
diffant deux `results.json` à la main.

Lancer le serveur (UI sur <https://localhost:5443>, cf. phase 3.6 HTTPS) :

```bash
docker compose up -d mlflow
```

Les entraînements le trouvent via `MLFLOW_TRACKING_URI` dans `.env` (cf.
§Configuration — la même ligne sert à l'hôte et aux conteneurs). **Sans cette
variable, MLflow est inerte** : l'entraînement tourne normalement, aucun run
n'est enregistré — c'est le mode par défaut des tests et de la CI, qui n'ont
donc jamais besoin d'un serveur.

### Ce qui est enregistré

| | Contenu |
|---|---|
| Paramètres | centrale, horizon, `meta_type`, `seq_len`, `n_splits`, `epochs`, `n_trials_*`, graine, **et la fenêtre d'évaluation** |
| Métriques | `kge_lgbm` / `kge_lstm` / `kge_stacking`, KGE par régime hydrologique et par saison, KGE par pas d'échéance (en courbe) |
| Artefacts | `results.json`, `meta_config.json`, tous les plots d'entraînement |
| Tags | centrale, horizon, commit git, graine |

La fenêtre d'évaluation (`eval_test_start`, `eval_n_train`…) est un paramètre
et pas un détail : `split_train_test` est **positionnel** (20 % de fin), donc
deux entraînements sur des CSV de longueurs différentes ne mesurent pas la
même période. Comparer leurs KGE sans regarder cette fenêtre n'a pas de sens.

### MLflow *et* DVC — pourquoi les deux

Ils ne répondent pas à la même question. **DVC** versionne les octets
(données et poids) et permet le retour arrière : `git checkout <tag> && dvc
pull`. **MLflow** indexe et compare les expériences, et sert de registre
lisible par un humain. Les deux se pointent mutuellement : le `run_id` MLflow
est écrit dans `meta_config.json`, et le commit git est un tag du run.

### Model Registry — promotion

`train.py --promote` promeut un candidat qui bat la production, puis
`promote_model` copie les artefacts, les versionne (`dvc add`), pose un commit
et un tag git — et enregistre la version au registry MLflow sous le nom
`projet_hydro-<centrale>-h<horizon>`, avec l'alias `@production`.

Les deux versions se répondent : `version.json` porte le `mlflow_run_id`, et
la version du registry porte le tag git en tag MLflow.

L'enregistrement se fait **après** le commit et le tag git, hors du bloc de
rollback : à ce stade la promotion est actée, et un registry injoignable ne
doit pas annuler un modèle correctement promu. On perd le lien, pas le modèle.

Ce qui est enregistré est le **répertoire d'artefacts**, pas une saveur MLflow
chargeable : le modèle est un trio (LightGBM + BiLSTM + méta) plus ses
scalers, et le chargement passe par `load_trained_models`. Le registry sert
d'index et d'historique des promotions ; DVC porte les octets et le rollback.

## Reverse proxy + stockage objet (NGINX + MinIO) — Phase 2

`api` et `mlflow` n'exposent plus de port directement : **nginx** est
l'unique point d'entrée réseau (`resolver` + résolution DNS paresseuse :
nginx démarre même si `api`/`mlflow` ne sont pas encore prêts).

```bash
docker compose up -d nginx      # démarre aussi api + mlflow (+ minio via mlflow)
curl -k https://localhost:8443/health   # HTTPS, cf. section suivante
open https://localhost:5443             # UI MLflow
```

**MinIO (stockage objet)** — remplace le volume local des artefacts MLflow
par un vrai stockage S3 (bucket `mlflow-artifacts`, créé automatiquement par
`minio-setup` au démarrage). Reste accessible **directement** (pas derrière
nginx) : la console MinIO est une SPA qui suppose être servie à la racine —
la proxifier sous un sous-chemin casserait ses assets, pas justifié pour un
outil d'admin interne à l'équipe.

```bash
docker compose up -d minio
open http://localhost:9001       # console MinIO (identifiants : .env MINIO_ROOT_USER/PASSWORD)
```

Pas de Postgres pour MLflow : backend SQLite sur volume, suffisant à 2.

## HTTPS (NGINX) — Phase 3.6

nginx est l'unique point d'entrée réseau (phase 2.5) — c'est donc là que le
chiffrement se met en place, une fois, pour les 3 services (API, MLflow,
Airflow). Chaque port historique redirige (301) vers son équivalent HTTPS ;
**aucun trafic en clair n'est servi** :

| Service | Port HTTP (redirection) | Port HTTPS (service réel) |
|---|---|---|
| API | 8000 | **8443** |
| MLflow | 5000 | **5443** |
| Airflow | 8080 | **8843** |

```bash
docker compose up -d nginx
curl http://localhost:8000/health        # 301 -> https://localhost:8443/health
curl -k https://localhost:8443/health    # -k : certificat auto-signé, cf. limite ci-dessous
```

**Certificat auto-signé** (`infrastructure/nginx/generate-cert.sh`, exécuté
une fois par le service compose `nginx-cert-init` — idempotent, ne régénère
pas si déjà présent) : `CN=localhost`, valable 825 jours. Pas de certificat
signé par une autorité reconnue (Let's Encrypt ou équivalent) car cela
suppose un **nom de domaine réel**, absent ici (projet de démonstration,
`localhost`). **Limite assumée et documentée** : navigateur et curl doivent
accepter explicitement ce certificat (`-k` en curl, avertissement "connexion
non privée" à valider manuellement en navigateur) — en production, la même
configuration nginx fonctionnerait telle quelle derrière un vrai domaine, il
suffirait de remplacer le certificat auto-signé par un certificat signé.

## CI (GitHub Actions) — Phase 3

Sur chaque PR et push vers `dev`/`main` : `ruff check .` puis `pytest -q`
(suite rapide, sans les tests `slow`, sans données DagsHub — les tests
tournent sur des fixtures synthétiques, aucun secret nécessaire).

Job unique `lint-test`, Python 3.11 natif (pas de conteneur) : plus rapide
qu'un rebuild Docker à chaque run, et le but de la CI est de vérifier le
code, pas l'image.

## Sécurisation de l'API — Phase 3.3

- **Clé API optionnelle** (`API_KEY` dans `.env`, vide par défaut = désactivée
  y compris tests/CI) : à fournir dans l'en-tête `X-API-Key` pour `/models` et
  `/predict`. `/health` reste public (convention monitoring).
- **Logs structurés** : une ligne JSON par requête (`request_id`, méthode,
  route, statut, latence), `request_id` renvoyé dans l'en-tête `X-Request-ID`.
- **Timeouts + rate-limit** côté nginx (déjà le point de passage unique,
  phase 2.5) plutôt que dans l'app : 10 req/s/IP (rafale 20), 30s de lecture
  max sur `/predict`.
- Pas de stack trace exposée (déjà en place depuis la phase 1) : toute
  exception de `run_prediction` devient un message court, jamais la trace.

```bash
# .env : API_KEY=ma-cle
curl -k https://localhost:8443/models                              # 401
curl -k -H "X-API-Key: ma-cle" https://localhost:8443/models        # 200
```

## BentoML — écarté (Phase 3.4)

Un service BentoML alternatif à l'API FastAPI a été construit puis retiré,
après validation avec le tuteur du projet. Raisons :

- **BentoML apporte de la valeur quand plusieurs modèles/images doivent être
  packagés et servis séparément** (versionnement de bundle, routage entre
  plusieurs services). Ici, tout tient dans **une seule image**
  (`projet_hydro:latest`) avec deux centrales et un seul horizon : ce
  problème ne se pose pas.
- **FastAPI (phase 1), déjà en place, sécurisé (phase 3.3) et suffisant**,
  couvre exactement le besoin (`/health`, `/models`, `/predict`).
- Coder un deuxième chemin de serving qui ne sert à rien en production allait
  à l'encontre du principe du projet (au plus simple, chaque outil justifié
  par un besoin réel) — retiré plutôt que maintenu comme simple démo.

## Kubernetes (Helm) — Phase 3.5

Chart minimal `infrastructure/helm/projet-hydro/` : **Deployment + Service +
Ingress + HPA** pour l'API (le service exposé au public — mlflow/minio/nginx
restent en docker-compose, usage interne équipe, pas le sujet du critère
scalabilité). Même image que docker-compose (`projet_hydro:latest`).

Validé par `helm lint` / `helm template` (pas de cluster réel requis pour la
validation statique) :

```bash
docker run --rm -v "$(pwd)/infrastructure/helm/projet-hydro:/chart" \
  --entrypoint helm alpine/helm:3.16.3 lint /chart
docker run --rm -v "$(pwd)/infrastructure/helm/projet-hydro:/chart" \
  --entrypoint helm alpine/helm:3.16.3 template test /chart
```

**Limite assumée** (cf. `templates/NOTES.txt`) : le chart ne peuple pas
`models/`/`centrales/` dans le conteneur (bind-montés depuis l'hôte en
docker-compose, après `dvc pull`). Sans données, l'API démarre et répond
normalement (dégradation gracieuse, déjà le comportement testé) mais ne sert
aucune prévision réelle — un PVC + initContainer `dvc pull` est le next step
documenté, pas implémenté à l'aveugle sans cluster pour le valider.

## Orchestration (Airflow) — Phase 3.1

Deux DAGs, deux cadences, dans `airflow/dags/`. Les tâches sont des
`BashOperator` qui lancent les **scripts `cron/scripts/`** — les mêmes que
`dvc repro` — dans le conteneur Airflow, sur le repo bind-monté.

```
hydro_predict  (toutes les heures, h+5)
  sources ─┬─ debit   maj-data.py          Hub'Eau, débit dans le store local
           └─ meteo   check-meteo.py       Open-Meteo répond-il pour [now, now+8h] ?
        ──▶ predict_archive  predict-archive.py

hydro_train    (lundi 02:00)
  data_preparation ──▶ validate --strict ──▶ train --promote
```

**Mêmes stages que DVC, groupés autrement.** Les `dvc.yaml` sont découpés par
domaine (preprocessing / model / postprocessing), les DAGs par cadence
(servir / entretenir) ; les `task_id` reprennent les noms de stages pour que
la correspondance se lise. Les DAGs appellent les scripts et **pas
`dvc repro`** : chaque repro réécrit `dvc.lock`, l'arbre git devient sale et
`promote_model` refuse de committer.

**Pas de `data_preparation` dans le DAG horaire** : la prédiction en
`source="live"` assemble sa fenêtre elle-même (débit du store local + météo en
direct). `data_preparation.csv` ne sert qu'à l'entraînement.

**Les deux sources bloquent.** Sans débit frais on ne prédit pas ; sans météo
`build_dossier` n'assemble rien. `meteo` existe parce que `read_points` ignore
un point Open-Meteo en échec sans lever : sept points en échec = DataFrame vide
sans erreur. `check-meteo.py` rend l'échec visible **avant** `predict_archive`,
et nomme la centrale. Chaque source a 2 retries à 5 min ; `predict_archive`
aucun (un échec là est un bug ou une donnée absente, pas un réseau qui tousse).
La vue Grid de l'UI devient de fait le journal de disponibilité des deux API.

**Le DAG hebdo ne pousse pas.** `train --promote` commite et tague en local
si le KGE bat la production ; le `git push` / `dvc push` restent humains —
la décision (repo bind-monté ou clone dédié, token en écriture dans un
conteneur) est en attente, cf. skill `hydro-mlops` §Propositions.

### Lancer

L'image Airflow dérive de l'image projet (`airflow/Dockerfile` :
`FROM projet_hydro:latest` + `apache-airflow==3.3.1`, +200 Mo de couche
propre). **Toujours dans cet ordre, jamais les deux dans la même commande** :
compose construit en parallèle et `airflow` partirait de l'ancienne
`projet_hydro`.

```bash
docker compose build app
docker compose build airflow
docker compose up -d nginx airflow
```

UI : <https://localhost:8843> (via nginx, cf. phase 3.6 HTTPS — pas de login en local,
`SIMPLE_AUTH_MANAGER_ALL_ADMINS`, même logique que `API_KEY` vide ; à durcir
avant toute exposition). `airflow standalone` = webserver + scheduler + SQLite
dans un processus : suffisant pour deux DAGs. Un nouveau DAG apparaît **en
pause** : l'activer dans l'UI (interrupteur) ou

```bash
docker compose exec airflow airflow dags unpause hydro_train
```

Modifier un script ou un DAG **ne demande pas de rebuild** (bind mount, le
scheduler relit `airflow/dags/` toutes les 30 s). Une dépendance dans
`pyproject.toml` en demande un (`build app` puis `build airflow`).

### Tests

`tests/dags/test_dags.py` charge les DAGs sans lancer Airflow : import sans
erreur, `task_id` = stages DVC, `catchup=False`, cron hebdo. Il tourne là où
Airflow est installé :

```bash
docker compose exec airflow python -m pytest tests/dags -q
```

Ailleurs (env conda, CI) il est *skipped*, pas rouge — installer 70 paquets
pour 4 asserts n'a pas de sens.

### Pièges rencontrés

- **CRLF** : `core.autocrlf=true` (défaut Git for Windows) transforme
  `entrypoint.sh` en CRLF au checkout → `exec entrypoint.sh: no such file or
  directory` dans **tous** les conteneurs qui l'utilisent (`app`, `api`,
  `airflow`). `.gitattributes` force LF sur `*.sh`.
- **Fuseau** : un conteneur est en UTC ; `predict_orchestrator` prend l'heure
  murale du système pour `generation-date`, qui reculait de 2 h par rapport au
  même script lancé depuis conda. `TZ=Europe/Paris` dans `x-env`.
- **`airflow/` est un homonyme Python** : vu depuis la racine, ce dossier est
  un paquet `airflow` vide. Le test fait `importorskip("airflow.models")`, pas
  `("airflow")`.
- **Build parallèle** : cf. ci-dessus, `build app` **puis** `build airflow`.

## Monitoring (Prometheus + Grafana) — Phase 4.1

`GET /metrics` sur l'API (format Prometheus, `serving/metrics.py`) : compteur
de requêtes par méthode/route/statut, histogramme de latence, et le **KGE du
modèle promu par centrale** (recalculé à chaque scrape depuis les
`version.json` -- pas de thread de rafraîchissement, 2 centrales, YAGNI).
Public comme `/health` (jamais de clé API), scrapé en interne par Prometheus
directement sur `api:8000`, pas besoin de passer par nginx.

```bash
docker compose up -d grafana        # démarre aussi prometheus + node-exporter (+ api)
curl http://localhost:8000/metrics  # via nginx -- ou direct : docker compose up -d api puis port-forward
open http://localhost:9090          # Prometheus (requêtes PromQL, onglet Alerts)
open http://localhost:3001          # Grafana (identifiants : .env GRAFANA_ADMIN_PASSWORD, admin par défaut)
```

Dashboard provisionné automatiquement (`infrastructure/grafana/dashboards/hydro-overview.json`) :
requêtes/s par statut, latence P95, KGE par centrale, CPU hôte.

**Seuils d'alerte** (`infrastructure/prometheus/alert_rules.yml`) évalués par
Prometheus lui-même (onglet *Alerts*) : API injoignable (1 min), KGE < 0.5
(5 min), latence P95 > 5s (5 min). Pas d'Alertmanager (routage
email/Slack) : ça demanderait un canal de notification réel qu'on n'a pas
pour ce projet de cours -- extension documentée, pas implémentée à l'aveugle.

Prometheus/Grafana/node-exporter restent **hors nginx** (outils d'admin
internes à l'équipe, même raisonnement que la console MinIO).

## Détection de dérive (Evidently) — Phase 4.2

Compare une fenêtre récente de `data_preparation.csv` (30 derniers jours)
à tout l'historique d'entraînement qui la précède, colonne par colonne
(test de Kolmogorov-Smirnov, `monitoring/drift.py`). **Signal de
surveillance, jamais bloquant** (contrairement au contrat `validate-data.py`,
phase 1.5) : une dérive détectée est loggée, jamais une erreur de pipeline.

```bash
docker compose run --rm app python cron/scripts/check-drift.py
docker compose run --rm app python cron/scripts/check-drift.py --dossier touzac_g2_G2
```

Seuil : **40 % des colonnes en dérive** déclenche `dataset_drift=true`
(pas une seule colonne isolée). Rapport JSON par centrale sous
`logs/drift/`, relu par `GET /metrics` (`data_drift_share`,
`data_drift_detected`) -- jamais recalculé en direct au scrape, un rapport
Evidently prend de vraies secondes, trop lent pour Prometheus. Alerte
`DataDrift` dans `infrastructure/prometheus/alert_rules.yml`.

## Déploiement cloud — stratégie documentée — Phase 4.4

**Pas de crédits cloud pour ce projet de cours** : rien n'est déployé en
continu chez un fournisseur. Ce qui suit décrit *comment* le stack actuel
s'y déploierait, brique par brique -- chaque service compose a un
équivalent managé direct, sans changement de code (mêmes images Docker,
mêmes variables d'environnement).

| Brique locale (docker-compose) | Équivalent cloud | Changement de code |
|---|---|---|
| `api`, `mlflow`, `airflow` (conteneurs) | Conteneurs managés (AWS ECS Fargate / Cloud Run / Azure Container Apps) | Aucun -- mêmes images `projet_hydro:latest` / `projet_hydro-airflow:latest` |
| `nginx` (reverse proxy + TLS auto-signé, phase 3.6) | Load balancer managé du fournisseur (ALB / Cloud Load Balancing) + certificat géré (ACM ou équivalent) | Aucun côté app -- juste un vrai certificat à la place de celui auto-signé, la limite documentée en phase 3.6 disparaît d'elle-même |
| `minio` (S3-compatible, phase 2.6) | Stockage objet natif (S3 / GCS / Azure Blob) | Aucun -- MLflow parle déjà l'API S3 via `boto3`/`MLFLOW_S3_ENDPOINT_URL`, changer l'URL et les credentials suffit |
| MLflow backend SQLite | Base managée (RDS Postgres / Cloud SQL) si plusieurs utilisateurs concurrents | `--backend-store-uri` seulement -- SQLite choisi ici volontairement, suffisant à 2 (cf. phase 2) |
| `airflow` en mode `standalone` | Service Airflow managé (MWAA / Cloud Composer) ou déploiement multi-nœud (webserver+scheduler+worker+DB) | Aucun sur les DAGs -- `standalone` est une simplification déjà documentée (phase 3.1), pas une impasse |
| Chart Helm (`infrastructure/helm/projet-hydro/`, phase 3.5) | Déploiement direct sur un cluster managé (EKS / GKE / AKS) | Aucun -- le chart est déjà écrit et validé (`helm lint`/`template`), il ne manque qu'un vrai cluster + les credentials cloud pour le `dvc pull` en initContainer (limite déjà documentée en phase 3.5) |
| `.env` (secrets : `DAGSHUB_TOKEN`, `API_KEY`, `MINIO_ROOT_*`) | Gestionnaire de secrets managé (AWS Secrets Manager / GCP Secret Manager), injecté en variables d'environnement | Aucun -- le code lit déjà des variables d'environnement (`pydantic-settings`), peu importe leur origine |
| CI GitHub Actions (phase 3.2) | + un job `deploy` (build, push vers un registre, rolling update) déclenché sur tag/merge `main` | Extension du workflow existant, pas une réécriture |

**DVC/DagsHub ne change pas** : le remote est déjà distant (DagsHub),
indépendant d'où tourne le calcul.

**Pourquoi documenté plutôt qu'implémenté** : faire tourner ce stack 24/7
chez un fournisseur pour un projet de cours n'a pas de justification
(coût récurrent pour zéro utilisateur réel) -- même raisonnement que pour
BentoML (phase 3.4) : ne pas coder/déployer pour rien. Le fait que docker-
compose reproduise exactement l'architecture cible (mêmes images, mêmes
variables d'environnement) démontre la portabilité sans dépenser.
