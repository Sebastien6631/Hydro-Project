# previ-R2-D2 (pipeline IA)

> **Version projet de cours MLOps** — ce dépôt est une version réduite de
> previ-R2-D2 (pipeline de production chez Barthe EnR), adaptée pour un
> projet de cours réalisé hors du réseau de l'entreprise. Périmètre réduit
> à 3 centrales (`apas_G1_G4`, `nancy_A`, `touzac_g2_G2`, toutes
> `flex_strategy: DEFAULT`). Les modules suivants, inaccessibles hors
> serveur de production, ont été retirés :
> - **OneGate** (`memorandum.py`) — API interne de structure des centrales ;
>   `config-general.json`/`config-raccordement.json` des 3 centrales sont
>   désormais des données statiques, versionnées via DVC.
> - **hydrospot_stream** (`preprocessing/puissance/`) — source de puissance
>   sur NAS ; `puissance.csv`/`puissance_horaire.csv` figés, idem.
> - **FTP météo NWP** (`nwp_ftp.py`/`retention.py`) — seul `nwp_reader.py`
>   (parseur pur, sans réseau) est conservé ; les colonnes météo de
>   `data_preparation.csv` restent figées en attendant un sous-projet
>   séparé de remplacement par une API météo publique.
> - **automate** (rsync/SSH, `HAUTE_CHUTE`) — aucune des 3 centrales
>   gardées n'utilise cette stratégie.
> - **Mail/digest quotidien** et **MLflow** — hors périmètre (lancement
>   manuel des scripts ; le tracking d'expériences est à refaire proprement
>   comme partie du travail de cours).
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
previ-R2-D2/
├── pyproject.toml               # package installable (pip install -e .)
├── requirements.txt
├── run.py                        # CLI entraînement/prédiction du modèle hybride (--train ...)
├── config/
│   ├── centrales/                # réservé (vide, .gitkeep) -- inutilisé dans cette version
│   ├── bv_mapping.yaml          # dossier -> nom de shapefile BV (centrales/REFERENCE/shapefiles/)
│   └── puissance_mapping.yaml   # dossier -> nom de dossier hydrospot_stream (repli explicite, cf. ci-dessous)
├── src/previ_r2d2/
│   ├── common/                  # config, secret_config (3 secrets restants), dvc_markers
│   ├── preprocessing/
│   │   ├── onboarding/          # validation.py -- complétude config-raccordement.json avant bv
│   │   ├── debit/                # eaufrance, Hub'Eau, stockage local (centrales/<dossier>/) + dédup
│   │   ├── puissance/            # puissance_store.py seul : résolution du dossier hydrospot_stream
│   │   │                        #   (mapping/heuristique, utilisée par la validation d'onboarding) --
│   │   │                        #   import/fusion réels (NAS) retirés, puissance*.csv figés
│   │   ├── meteo/                 # nwp_reader.py seul (parseur, FTP retiré)
│   │   ├── data_preparation/      # Data_Preparation : débit + météo NWP brute + amont brut
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
│   │                              #   eligibility (12 mois / réentraînement mensuel),
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
│   └── scripts/                  # CLI minces (maj-data, onboarding-check, onboarding-bv,
│                                  #   build-data-preparation, train, predict-archive) --
│                                  #   lancement manuel, pas de cron/wrappers/ (retiré, cf. note en tête)
├── dvc/
│   ├── preprocessing/dvc.yaml     # debit -> onboarding_check -> bv -> data_preparation (manuel)
│   ├── model/dvc.yaml             # train_new (quotidien, nouvelles centrales), train_monthly (mensuel)
│   └── postprocessing/dvc.yaml    # predict_archive (horaire)
├── outputs/                       # sorties de prédiction/entraînement (gitignored)
├── tests/                         # miroir de src/previ_r2d2/
├── centrales/                     # données des 3 centrales -- <dossier>/ versionné via DVC
│   │                              #   (remote local ../remote_dvc, cf. <dossier>.dvc à la racine)
│   ├── REFERENCE/                 # bv_rules.json + centrales_calibration.json (git-tracké) ;
│   │                              #   config-general.json (présent localement, gitignoré, non DVC) ;
│   │                              #   shapefiles/, files/ (gitignorés)
│   └── <dossier>/                 # config-raccordement.json, *.csv, bv.json, data_preparation.csv,
│                                  #   prevision.json, enchere.json
└── data/                          # inutilisé pour l'instant (réservé à un usage futur)
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Configuration

Les secrets vivent dans `src/previ_r2d2/common/secret_config.py` (Python
local, **non versionné**) :

```python
PREVI_PUISSANCE_SOURCE_ROOT = "..." # racine hydrospot_stream (utilisée seulement pour la résolution
                                     #   du mapping puissance -- import/fusion réels retirés)
PREVI_MNT = "..."                   # GeoTIFF MNT France entière (repli délimitation BV)
PREVI_NAS_METEO = "..."             # racine des fichiers météo NWP bruts (acquisition FTP retirée --
                                     #   dossier vide/absent = colonnes météo vides, dégradation gracieuse)
```

Priorité de lecture : `secret_config.py` > variables d'environnement > défauts. Les secrets
propres aux modules retirés dans cette version (cf. note en tête de fichier) ont disparu
avec eux.

## Pipeline (DVC)

```bash
dvc dag dvc/preprocessing/dvc.yaml       # debit -> onboarding_check -> bv -> data_preparation (manuel)
dvc dag dvc/model/dvc.yaml               # train_new (quotidien), train_monthly (mensuel)
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
       dvc/model/dvc.yaml : train_new  ──┐
       dvc/model/dvc.yaml : train_monthly┤
                                          │
       dvc/postprocessing/dvc.yaml : predict_archive
```

`debit`, `onboarding_check` et `bv` déclarent chacun un `outs:` minimal
(`logs/dvc_markers/<stage>.json`, `cache: false`) — pas une vraie sortie mise
en cache, juste un marqueur horodaté écrit en une ligne
(`previ_r2d2.common.dvc_markers.write(...)`) à la fin de chaque script, pour
donner une vraie arête DAG entre stages (sans ça, DVC n'a rien à quoi
accrocher une dépendance). `debit` garde en plus `always_changed: true` : sa
vraie source (Hub'Eau) est externe à DVC et change en permanence.
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

Pour chaque raccordement `flex_strategy == "DEFAULT"` (les 3 centrales gardées
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

### `onboarding-bv.py` — caractérisation du bassin versant

Pour chaque centrale (regroupées par site physique quand plusieurs dossiers
partagent le même `centrale_uuid`), mesure le bassin versant amont — via
shapefile connu (`config/bv_mapping.yaml`) ou repli par délimitation MNT
depuis le point exutoire — calcule les paramètres de calage (`K_base`,
`exposition`, `kc_unit`), les points météo NWP représentatifs, les
coordonnées Hub'Eau des stations hydrométriques (référence + amont) et le
temps de transit hydraulique par cross-corrélation saisonnière
(amont→référence : débit vs débit ; référence→centrale : débit vs
`power_output` nettoyé, avec repli géométrique par ratio de distances si la
corrélation directe est peu fiable). Écrit `centrales/<dossier>/bv.json`.
**Idempotent** : un `bv.json` déjà présent n'est jamais retraité sans
`--force` — voir `src/previ_r2d2/preprocessing/bv/README.md` pour le détail
du schéma de sortie et de la logique de repli.

**Sélection des points météo NWP (`stations_meteo_nwp`)** — la grille NWP
réelle s'est densifiée avec le temps (la grille 2021-2024 est un sous-ensemble
strict de la grille actuelle, aucun point supprimé, seulement ajoutés).
`select_meteo_points` préfère désormais les points de grille déjà couverts en
2021 (`bv_builder.historical_grid_points`, lu depuis un fichier NWP de
référence sur le NAS) et ne retombe sur la sélection purement géométrique que
si aucun candidat du polygone n'a de couverture historique (région
nouvellement couverte). **Limite connue** : certaines régions entières
(sud, ex. `campagne_G1_G2`, `la_bastide_G1_G2_G3`, `counozouls_G1`) n'étaient
tout simplement pas dans la grille avant le 2024-12-06 — pour elles, aucun
point du polygone n'a de couverture historique, le repli s'applique, et le
`data_preparation.csv` correspondant aura ~70% de son historique débit sans
météo (gap réel côté fournisseur, rien à corriger côté code).

```bash
python cron/scripts/onboarding-bv.py batch                              # toutes les centrales
python cron/scripts/onboarding-bv.py batch --force                      # recalcule même l'existant
python cron/scripts/onboarding-bv.py single --dossier apas_G1_G4        # test ciblé
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

- Débit : station de référence Hub'Eau (les 3 centrales gardées sont toutes
  en `flex_strategy == "DEFAULT"`).
- Amont : colonne `debit_amont` (un seul) ou `debit_amont_{code}` (plusieurs,
  dédupliqués), nommage repris de `lightgbm_model.py` (Previ_v2).
- Météo : points de `bv.json.stations_meteo_nwp` (déjà calculés par
  `onboarding-bv.py`), matching exact `(latitude, longitude)` contre les
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
`nwp_reader.read_points` resample chaque point à l'heure avant de combiner
(moyenne les doublons plutôt que de tenter un arbitrage, hors périmètre pour
l'historique confirmé).

**Piège rencontré (format NWP brut a changé dans le temps)** — les fichiers
2021-2024 ont un header différent de 2025+ (`Latitude`/`Longitude`/`2t`
capitalisés + une colonne `hour_index` en plus, vs `latitude`/`longitude`/`2T`
minuscules côté récent) -- `nwp_reader.parse_nwp_file` normalise la casse des
colonnes avant le rename pour accepter les deux formats.

```bash
python cron/scripts/build-data-preparation.py                            # les 3 centrales
python cron/scripts/build-data-preparation.py --dossier apas_G1_G4       # test ciblé
python cron/scripts/build-data-preparation.py --full-history             # backfill complet
```

### `onboarding-check.py` — validation d'un raccordement avant `bv`

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

### `train.py` — entraînement automatisé (2 cadences)

- `run_new_dossiers()` (quotidien) : pour chaque horizon **sans** modèle en
  production, rafraîchit `data_preparation.csv` pour ce dossier puis entraîne
  dès que 12 mois d'historique sont atteints.
- `run_monthly_retrain()` (mensuel) : pour chaque horizon **déjà** en
  production, réentraîne inconditionnellement (l'invocation mensuelle est
  l'échéance) — promeut seulement si le nouveau modèle est meilleur (KGE sur
  le même holdout que le modèle en prod, cf. `model/pipeline/promotion.py`).

```bash
python cron/scripts/train.py --mode new       # stage DVC train_new
python cron/scripts/train.py --mode monthly   # stage DVC train_monthly
python cron/scripts/train.py --dossier apas_G1_G4 --horizon 8 --force   # test manuel ciblé, ignore l'éligibilité
```

Le modèle en PRODUCTION vit dans `models/<dossier>/h<horizon>/`, versionné
DVC + tag git (`<dossier>-h<horizon>-v<N>`, rollback = `git checkout <tag> &&
dvc pull`) — jamais dans `weights/hybrid/` (zone de travail manuelle) ni
`weights/hybrid_candidate/` (candidat en cours d'évaluation, jamais lu par la
prédiction).

### `predict-archive.py` — prédiction + archivage horaire

Pour chaque (dossier, horizon) ayant un modèle en production : archive le
JSON de prévision existant (`postprocessing/archive.py::archive_previous_json`
— l'heure de production, pas l'heure d'archivage, dans le nom de fichier)
vers `./ARCHIVE/<dossier>/<AAAA>/<MM>/<JJ>/` (local, `config.ARCHIVE_ROOT`),
puis prédit la nouvelle heure et écrit `centrales/<dossier>/prevision.json`
(h8) ou fusionne dans `enchere.json` (clés `J2`/`J3`, h48/h72). Isolation par
(dossier, horizon).

`run_prediction`/`load_prediction_window` acceptent un paramètre
`source="live"` (défaut, assemble la fenêtre dynamiquement comme
`build-data-preparation.py`) ou `source="frozen"` (relit directement
`data_preparation.csv` tel quel, sans aucun appel réseau) — utile pour une
prédiction 100% reproductible, par exemple en test.

```bash
python cron/scripts/predict-archive.py   # toutes les centrales avec un modèle en prod (pas de --dossier)
```

### `src/previ_r2d2/model/` — portage du modèle hybride meta

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
  réentraînement mensuel) et promotion versionnée (comparaison KGE candidat
  vs production sur le même holdout, `dvc add`+`git tag`, sûr face à un échec
  partiel). Cf. section `train.py` ci-dessus pour le détail opérationnel.

Voir le skill `previ-r2d2` pour le détail pièce par pièce et le skill
`hybrid-meta-ops` pour l'architecture du modèle hybride.

```bash
.venv/bin/python -m pytest tests/model/ -v   # tests du portage modèle
```

### `run.py` — entraînement/prédiction MANUELS (expés, pas le chemin automatisé)

CLI mince (`src/previ_r2d2/cli.py`) qui appelle `run_training`/`run_prediction`
pour un dossier+horizon donné, ou pour toutes les centrales × 3 horizons
(8h/48h/72h). **Le chemin opérationnel automatisé passe par
`cron/scripts/train.py`/`predict-archive.py`** (cf. sections dédiées
ci-dessus), pas par `run.py` — celui-ci reste réservé aux tests/expés
manuels (entraîne toujours dans `weights/hybrid/`, jamais `models/`, donc
jamais lu par la prédiction opérationnelle sans passer par une promotion
explicite).

```bash
python run.py --train --dossier apas_G1_G4 --horizon 8
python run.py --train --all-dossiers                      # toutes les centrales x h8/h48/h72
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
python cron/scripts/onboarding-bv.py batch                  # 3. caractérise le BV + stations/transit (no-op si déjà fait)
python cron/scripts/train.py --mode new                     # 4. 1er entraînement des nouvelles centrales
python cron/scripts/train.py --mode monthly                 # 5. réentraînement mensuel (promotion conditionnelle)
python cron/scripts/predict-archive.py                      # 6. archive + prédit la nouvelle heure
```

## Notifications / MLflow

Retirés dans cette version (cf. note en tête de fichier) — lancement
manuel des scripts, pas de digest mail ni de tracking d'expériences pour
l'instant.

## Références

- Hub'Eau hydrométrie : https://hubeau.eaufrance.fr/page/api-hydrometrie
