# previ-R2-D2 (pipeline IA)

Reprise automatisée de Previ_v2 : structure des centrales depuis l'API
**OneGate** (`/hydrogrid`), collecte des débits (eaufrance / Hub'Eau) et de
la puissance (hydrospot_stream), orchestrée via DVC.

## Architecture

```
previ-R2-D2/
├── pyproject.toml               # package installable (pip install -e .)
├── requirements.txt
├── run.py                        # CLI entraînement/prédiction du modèle hybride (--train ...)
├── config/
│   ├── centrales/<dossier>/     # structure à venir (tokapi_config.yaml, puissance_config.yaml)
│   ├── bv_mapping.yaml          # dossier -> nom de shapefile BV (centrales/REFERENCE/shapefiles/)
│   └── puissance_mapping.yaml   # dossier -> nom de dossier hydrospot_stream (repli explicite)
├── src/previ_r2d2/
│   ├── common/                  # config, mailer, daily_report (digest quotidien), dvc_markers,
│   │                            #   mlflow_tracker (file-based, cf. section MLflow), client OneGate
│   ├── preprocessing/
│   │   ├── onegate/             # memorandum -> liste des raccordements
│   │   ├── onboarding/          # validation.py -- complétude config-raccordement.json avant bv
│   │   ├── debit/                # eaufrance, Hub'Eau, stockage NAS + dédup
│   │   ├── puissance/            # hydrospot_stream -> puissance.csv/puissance_horaire.csv
│   │   ├── automate/              # débit "automate" (ex-TokAPI) : sync rsync + calcul physique
│   │   ├── meteo/                 # FTP NWP + rétention + nwp_reader (parsing fichiers bruts)
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
│   └── postprocessing/           # archive.py (archivage horaire NAS) -- API FastAPI Previ_v2 non portée
├── models/                        # SOURCE DE VÉRITÉ modèles en PRODUCTION, versionné DVC+git
│                                  #   <dossier>/h<horizon>/{version.json, meta_config.json, bilstm.pt, ...}
├── weights/
│   ├── hybrid/<dossier>/h<horizon>/           # zone de travail manuelle (run.py, expés)
│   └── hybrid_candidate/<dossier>/h<horizon>/ # candidat en cours d'évaluation par train.py (auto)
├── cron/
│   ├── scripts/                  # CLI minces (majdata-memo, maj-data, maj-puissance, maj-automate,
│   │                              #   maj-meteo, clean-meteo, onboarding-bv, onboarding-check,
│   │                              #   build-data-preparation, train, predict-archive, daily-sync-report)
│   └── wrappers/                  # crontab_previ_r2d2.cron (séparé de celui de Previ_v2, chargés ensemble)
├── dvc/
│   ├── preprocessing/dvc.yaml     # memorandum -> {puissance, debit, debit_automate} -> onboarding_check
│   │                              #   -> bv -> data_preparation (manuel) + daily_sync_report (indépendant)
│   ├── model/dvc.yaml             # train_new (quotidien, nouvelles centrales), train_monthly (mensuel)
│   └── postprocessing/dvc.yaml    # predict_archive (horaire)
├── outputs/                       # sorties de prédiction/entraînement (gitignored)
├── tests/                         # miroir de src/previ_r2d2/
├── centrales/                     # sortie OneGate + bv.json + symlinks NAS par dossier (gitignored)
│   ├── REFERENCE/                 # config-general.json, bv_rules.json, centrales_calibration.json,
│   │                              #   automate_sync.yaml, automate_physics.yaml
│   │                              #   (git-tracké sauf shapefiles/ et files/), shapefiles/, files/
│   └── <dossier>/                 # *.csv, bv.json, data_preparation.csv, prevision.json, enchere.json
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
ONEGATE_BASE_URL = "..."            # URL de l'API OneGate
PREVI_BEARER_TOKEN = "..."          # jeton d'accès OneGate
REFRESCH = "..."                    # jeton de rafraîchissement (optionnel)

PREVI_NAS_DATA_ROOT = "..."         # racine NAS pour les CSV débit/puissance
PREVI_PUISSANCE_SOURCE_ROOT = "..." # racine hydrospot_stream
PREVI_CONSIGNES_ROOT = "..."        # racine des ordres de consigne EDF (.true/.false)
PREVI_MNT = "..."                   # GeoTIFF MNT France entière (repli délimitation BV)
PREVI_MLFLOW_URI = "..."            # ex. "file:///mnt/nasGB/sebastien/mlflow" (absent = tracking désactivé)

PREVI_NAS_METEO = "..."             # racine NAS pour les fichiers météo NWP (FTP)
PREVI_FTP_HOST = "..."              # hôte FTP météo NWP
PREVI_FTP_USER = "..."              # identifiant FTP
PREVI_FTP_PASS = "..."              # mot de passe FTP

MAIL_SMTP_HOST = "..."
MAIL_SMTP_PORT = 587
MAIL_SMTP_USER = "..."
MAIL_SMTP_PASSWORD = "..."
MAIL_FROM = "..."
MAIL_RECIPIENTS = ["..."]
```

Priorité de lecture : `secret_config.py` > variables d'environnement > défauts.

## Pipeline (DVC)

```bash
dvc dag dvc/preprocessing/dvc.yaml       # memorandum -> {puissance, debit, debit_automate}
                                          #   -> onboarding_check -> bv -> data_preparation (manuel)
dvc dag dvc/model/dvc.yaml               # train_new (quotidien), train_monthly (mensuel)
dvc dag dvc/postprocessing/dvc.yaml      # predict_archive (horaire)
dvc repro dvc/preprocessing/dvc.yaml     # exécute tout ce pilier, dans l'ordre
```

```
memorandum ──┬──> puissance ──┬──> debit ────────────┐
             │                │                       │
             │                └──> debit_automate ────┼──> onboarding_check ──> bv
             │                                         │                        │
             └─────────────────────────────────────────┘                        │
                                                                                  ▼
                                       (data_preparation : manuel uniquement,
                                        train.py le rafraîchit lui-même par dossier)
                                                                                  │
                                              dvc/model/dvc.yaml : train_new  ──┐ │
                                              dvc/model/dvc.yaml : train_monthly┤◄┘
                                                                                 │
                                       dvc/postprocessing/dvc.yaml : predict_archive

  daily_sync_report   (indépendant, dvc/preprocessing/dvc.yaml — dépend du code
                        des scripts horaires + de logs/daily_sync_state/)
```

`debit`, `puissance`, `debit_automate`, `onboarding_check` et `bv` déclarent
chacun un `outs:` minimal (`logs/dvc_markers/<stage>.json`, `cache: false`) —
pas une vraie sortie mise en cache, juste un marqueur horodaté écrit en une
ligne (`previ_r2d2.common.dvc_markers.write(...)`) à la fin de chaque script,
pour donner une vraie arête DAG entre stages (sans ça, DVC n'a rien à quoi
accrocher une dépendance). `debit`/`puissance`/`debit_automate` gardent en
plus `always_changed: true` : leur vraie source (Hub'Eau, rsync hydrospot)
est externe à DVC et change en permanence. `data_preparation` **n'est appelé
par rien d'automatisé** depuis 2026-07-17 — `train.py` (`dvc/model/dvc.yaml`)
le rafraîchit lui-même, dossier par dossier, juste avant de vérifier
l'éligibilité de CE dossier (décision actée : un `foreach` DVC par dossier
aurait nécessité une liste de dossiers maintenue hors du yaml + restructurer
`train` en stages par dossier pour un lien réellement significatif — jugé
disproportionné). Le stage reste un outil manuel
(`dvc repro dvc.yaml:data_preparation`, backfill groupé).

## Les scripts

### `majdata-memo.py` — OneGate → structure des centrales

1. `GET /hydrogrid/memorandum/select` (clés : noms, `flexibilite_*`,
   `flex_strategy`, `station_vigicrue_reference`, `stations_vigicrue_amont`, `uuid`,
   et les champs `puissance_config.yaml` Previ_v2 ci-dessous).
2. Aplatit l'arbre en raccordements, ne garde que `flexibilite_rte = true`.
3. Écrit `centrales/REFERENCE/config-general.json` + un dossier/`config-raccordement.json`
   par raccordement (nom = `nom_centrale` + groupes, espaces → `_`).

OneGate stocke tout en texte (BDD) : les nombres et listes arrivent en chaînes
JSON-encodées (ex. `"2"`, `"[80.07]"`), décodées via `decode_onegate_value`
(même principe que `_is_true` pour `flexibilite_rte="true"`). Une chaîne vide
signifie "absent" (clé omise, jamais de `null` trompeur). Certains champs
arrivent parfois mal formés côté OneGate (segments multiples sans crochets
extérieurs, `.inf` en syntaxe YAML) — `decode_chute_disponible` retente un
décodage tolérant (crochets extérieurs ajoutés, `.inf`/`-.inf` → chaîne
`"Inf"`/`"-Inf"`, jamais le token `Infinity` qui n'est pas du JSON standard).

Champs ajoutés au `config-raccordement.json`, repris de `puissance_config.yaml`
(Previ_v2) :
- **Niveau centrale** : `debit_reserve`, `debit_non_turbinable` (valeur unique,
  ou liste `[été, hiver]` — éclatée en `q_non_turbinable_ete`/
  `q_non_turbinable_hiver` + `periode_ete` calculé depuis `flex_strategy` :
  `HAUTE_CHUTE` → 1er juillet-31 octobre, sinon 1er juin-31 octobre).
- **Niveau raccordement** : `facteur_debit`.
- **Niveau groupe** (dans chaque entrée `groupes[]`) : `priorite`,
  `debit_armement_turbine`, `debit_max`, `rendement` (coefficients polynôme),
  `chute_disponible_polynome`/`chute_disponible_seuils` (liste de listes,
  un segment par tranche de débit).

Les champs `decalage_*`/`transit_amont` de `puissance_config.yaml` ne sont
**pas** repris ici : ils correspondent à `transit_vers_reference_h`/
`transit_vers_centrale_h`, déjà calculés dans `bv.json` (cf. `onboarding-bv.py`).

```bash
python cron/scripts/majdata-memo.py --check      # test de connexion (via /select)
python cron/scripts/majdata-memo.py              # génère centrales/
python cron/scripts/majdata-memo.py --input x.json --output ./out   # hors ligne
```

### `maj-data.py` — import / mise à jour des débits

Pour chaque raccordement `flex_strategy == "DEFAULT"`, on traite **la station de
référence** (`station_vigicrue_reference`) **et toutes les stations amont**
(`stations_vigicrue_amont`, codes séparés par des virgules). Pour chaque station :

- code jamais vu ailleurs → **import** complet `01/01/2021 → aujourd'hui` (eaufrance) ;
- code déjà porté par un autre dossier → **symlink** NAS vers ce fichier réel (dédup, pas de nouvel appel API) ;
- CSV déjà présent pour ce dossier → **mise à jour** incrémentale : Hub'Eau `observations_tr` si plus
  récent que eaufrance (agrégé en moyenne horaire), sinon eaufrance seule.

Sortie : `<NAS_DATA_ROOT>/<dossier>/<station>.csv` (référence) ou `amont_<station>.csv`
(stations amont), symlinké dans `centrales/<dossier>/`. **Ne mail plus rien
lui-même** — journalise son bilan (imports, MAJ, liens, erreurs groupées) via
`daily_report.record(...)` pour le mail unique du jour, cf. section Notifications.

```bash
python cron/scripts/maj-data.py                  # import/MAJ (tourne toutes les heures via DVC)
python cron/scripts/maj-data.py --dossier apas_G1_G4   # test ciblé
python cron/scripts/maj-data.py --end 03/07/2026
```

### `maj-puissance.py` — import de la puissance depuis hydrospot_stream

Pour chaque raccordement, retrouve le dossier `hydrospot_stream` correspondant
(mot-clé centrale + numéro de groupe, ou `config/puissance_mapping.yaml` en
repli explicite), fusionne ses 3 fichiers bruts (`date_power.csv`,
`date_power_fill_nan.csv`, `date_power_fill_nan_neg_price.csv`) et calcule
`power_output` en deux étages, comme Previ_v2 :

- **Étage 1** (`preprocessing/puissance/consignes.py`) : interpolation linéaire
  de `Puissance` sur les fenêtres de consigne EDF (`PREVI_CONSIGNES_ROOT`),
  sinon priorité `Puissance_neg_price` > `Puissance` > `0.0`. Écrit
  `<NAS_DATA_ROOT>/<dossier>/puissance.csv` (minute par minute), colonnes
  `Date;Puissance;Puissance_fill_nan;Puissance_neg_price;MA_baisse;power_output`.
- **Étage 2** (`preprocessing/puissance/cleaning.py`) : détection de chaos,
  interpolation, ré-échantillonnage horaire (médiane). Écrit
  `<NAS_DATA_ROOT>/<dossier>/puissance_horaire.csv` (une ligne/heure),
  colonnes `Date;power_output` — consommé par le calcul de transit
  hydraulique de `onboarding-bv.py`.

Les deux fichiers sont symlinkés dans `centrales/<dossier>/`. **Ne mail plus
rien lui-même** (journalise via `daily_report.record(...)`, cf. section
Notifications) — tourne une fois par jour (`01:00`, pas horaire comme
`debit`/`debit_automate` : la fusion hydrospot_stream n'a pas besoin d'une
cadence plus fine).

```bash
python cron/scripts/maj-puissance.py                        # tous les raccordements
python cron/scripts/maj-puissance.py --dossier apas_G1_G4   # test ciblé
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

### `maj-meteo.py` / `clean-meteo.py` — météo NWP (FTP)

`maj-meteo.py` mirrore quotidiennement le dossier météo NWP du jour depuis le
FTP (`$PREVI_FTP_HOST`) vers `<NAS_METEO>/<AAAA>/<MM>/<JJ>/`. Alerte mail si
connexion impossible, échecs de téléchargement, ou nombre de fichiers sous le
seuil minimum (24, un run complet ≈ 145 fichiers).

`clean-meteo.py` supprime, pour une année donnée, les fichiers d'échéance
24h-360h des jours passés (jour courant et échéances courtes conservés).

Scripts cron **autonomes, hors pipeline DVC** — cadence quotidienne
incompatible avec le `dvc repro` horaire du débit.

```bash
python cron/scripts/maj-meteo.py                    # téléchargement du jour
python cron/scripts/clean-meteo.py                  # nettoyage année courante
python cron/scripts/clean-meteo.py --annee 2025 --dry-run
```

### `maj-automate.py` — débit "automate" (ex-TokAPI, rsync/SSH)

Pour chaque dossier `flex_strategy == "HAUTE_CHUTE"` (pas de station
Vigicrues — débit calculé à partir des capteurs bruts de l'automate de
contrôle local de la centrale, remontés par rsync/SSH) : sync les capteurs
(`centrales/REFERENCE/automate_sync.yaml`, mêmes dossiers NAS que Previ_v2,
sync indépendant), calcule le débit entrant via la physique de la centrale
(`centrales/REFERENCE/automate_physics.yaml` — coefficients de calage, port
des anciens `Fichier_config_*.py`), fusionne avec l'historique déjà écrit
(pas d'écrasement) et écrit `<NAS_DATA_ROOT>/<dossier>/debit_automate.csv`,
symlinké dans `centrales/<dossier>/` (même format que les CSV débit station :
`Date (TU);Valeur (en m³/s)`).

Le calcul du débit turbiné a jusqu'à 4 méthodes candidates selon les capteurs
disponibles (puissance, pression, ouverture vanne/injecteurs) — la colonne
finale retenue priorise l'ouverture (`_ouv`) puis la puissance (`_P`). Si la
méthode retenue dépend réellement de `puissance_horaire.csv` (`Q_fct_P`/
`Q_fct_P_Hn_R`), la sortie est tronquée aux heures où la puissance est
disponible — sinon les heures sans puissance produiraient un résidu
déversoir-seul trompeur (pas un vrai zéro).

**Ne mail plus rien lui-même** — journalise via `daily_report.record(...)`
(cf. section Notifications).

**Stage DVC `debit_automate`** (`always_changed: true`) depuis 2026-07-13 —
appelé par le crontab avec `debit` en **un seul** `dvc repro` (jamais deux
`dvc repro` séparés au même horaire : lock exclusif projet entier, cf.
section Pipeline). Remplace le sync TokAPI Melles/Bonneval de Previ_v2, cf.
`cron/wrappers/crontab_previ_r2d2.cron`.

`automate_sync.yaml` peut déclarer un `source_names` (dossier distant hydrospot
→ dossier destination local) quand les deux noms diffèrent (cas de Melles :
`PosInj1` distant → `ML2_POS_STAB_INJEC1_G1` local) — absent = identité (cas
de Bonneval). Un rsync avec un chemin source erroné échoue silencieusement
si le code retour n'est pas vérifié : `sync_variable()` vérifie maintenant
`result.returncode` et lève une erreur explicite le cas échéant.

```bash
python cron/scripts/maj-automate.py                          # toutes les centrales automate
python cron/scripts/maj-automate.py --dossier bonneval_G2    # test ciblé
python cron/scripts/maj-automate.py --skip-sync              # calcul seul, sans rsync (test sans
                                                               # risque de collision avec le cron Previ_v2)
python cron/scripts/maj-automate.py --full-history            # backfill ponctuel : tout l'historique
                                                               # disponible, pas juste la fenêtre glissante
```

### `daily-sync-report.py` — mail unique quotidien

Relit les exécutions journalisées aujourd'hui (`logs/daily_sync_state/<date>.jsonl`,
alimenté par `maj-data.py`/`maj-automate.py`/`maj-puissance.py` via
`daily_report.record(...)`) et envoie **LE seul mail du jour** (bilan compact
si tout est OK, détail complet des exécutions en erreur) — toujours envoyé,
même sans erreur. Stage DVC `daily_sync_report`, cron quotidien à `23:55`
(après la dernière exécution horaire du jour). Voir section Notifications.

```bash
python cron/scripts/daily-sync-report.py
```

### `build-data-preparation.py` — Data_Preparation (débit + météo + amont brut)

Construit/met à jour, pour chaque centrale, un CSV historique
`data_preparation.csv` combinant débit brut + météo NWP brute + amont brut,
alignés par horodatage horaire — aucune feature (lags, gradients,
transit_amont saisonnier) n'est calculée ici, ça reste un sous-projet
ultérieur (entraînement/prédiction du modèle hybride meta).

**Pas de cron** — lancé à la demande avant un entraînement (la cadence
dépend des dates d'entraînement, pas d'une fréquence fixe). Mail
systématique via `mailer.send_report` (pas le digest quotidien de
`daily_report` : ce script n'est pas horaire, pas de bruit à réduire). Stage
DVC `data_preparation` (cf. section Pipeline), avec une vraie dépendance sur
`debit`/`debit_automate`/`puissance`/`bv` (via marqueurs) — absent du
crontab, mais `dvc repro dvc.yaml:data_preparation` régénère aussi ses 4
dépendances amont au passage.

- Débit : `flex_strategy == "DEFAULT"` -> station de référence Hub'Eau ;
  `"HAUTE_CHUTE"` -> `debit_automate.csv`.
- Amont : uniquement pour `DEFAULT` (seul cas où `maj-data.py` collecte des
  stations amont) -- colonne `debit_amont` (un seul) ou `debit_amont_{code}`
  (plusieurs, dédupliqués), nommage repris de `lightgbm_model.py` (Previ_v2).
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
  l'entraînement, et ça évite de scanner des années de fichiers NWP pour rien
  (ex. `debit_automate` qui démarre parfois des années après le début de
  l'archive météo, comme `bonneval_G2` qui ne démarre qu'en 2024-06).

**Lecture sans effet de bord** — `debit_source.py`/`amont_source.py` lisent
via le symlink local déjà posé par l'import réel
(`config.CENTRALES_DIR / dossier / station_store.filename_for(...)`),
jamais via `station_store.resolve_nas_path` (qui écrit `_index.json` et peut
créer un symlink NAS -- effet de bord inapproprié pour un module qui ne fait
que lire une donnée déjà importée par un autre script).

**Piège rencontré (rétention météo en retard)** — la rétention (sous-projet
météo, `retention.py`) peut avoir du retard sur un jour donné (cron en panne
temporaire) : ce jour garde alors toutes ses échéances (000-360) au lieu de
000-023 seulement, et ses échéances longues (ex. 024) pointent vers des
`flow_date` déjà couvertes par les jours suivants -- doublon d'horodatage
bien réel dans l'archive brute (deux runs différents, pas une corruption).
`nwp_reader.read_points` resample chaque point à l'heure avant de combiner
(moyenne les doublons plutôt que de tenter un arbitrage, hors périmètre pour
l'historique confirmé) -- même remède déjà utilisé pour les capteurs automate.

**Piège rencontré (format NWP brut a changé dans le temps)** — les fichiers
2021-2024 ont un header différent de 2025+ (`Latitude`/`Longitude`/`2t`
capitalisés + une colonne `hour_index` en plus, vs `latitude`/`longitude`/`2T`
minuscules côté récent) -- `nwp_reader.parse_nwp_file` normalise la casse des
colonnes avant le rename pour accepter les deux formats.

**Point ouvert (non résolu) — `bonneval_G1` sans débit propre** :
`bonneval_G1` n'a pas son propre `debit_automate.csv` (partage le débit
physique de `bonneval_G2`, mais aucun champ config ne l'exprime aujourd'hui —
ni `modele_source` dans `config-general.json`, ni entrée dans
`automate_sync.yaml`/`automate_physics.yaml`) : son `data_preparation.csv` a
`debit_m3s` entièrement vide. À corriger plus tard (alias explicite dans
`debit_source.py` ou nouveau champ OneGate — décision différée).

**Limite connue (fournisseur météo, pas un bug)** — `campagne_G1_G2`,
`la_bastide_G1_G2_G3` et `counozouls_G1` sont dans une zone (sud, lat
~42.6-43.0) absente de la grille NWP avant le 2024-12-06 -- pour les deux
premières (débit dispo depuis 2021), ~70% de leur historique débit n'aura
jamais de météo en face. Rien à corriger côté code (`select_meteo_points`
retombe déjà correctement sur la sélection géométrique, aucun point du
polygone n'a de couverture antérieure) — à prendre en compte dans le
découpage train/test de la pièce B/C.

```bash
python cron/scripts/build-data-preparation.py                            # toutes les centrales
python cron/scripts/build-data-preparation.py --dossier melles           # test ciblé
python cron/scripts/build-data-preparation.py --full-history             # backfill complet
```

### `onboarding-check.py` — validation d'un raccordement avant `bv`

Pour chaque raccordement sans `bv.json` encore (pas onboardé), valide son
`config-raccordement.json` (`preprocessing/onboarding/validation.py::missing_fields`)
et journalise les infos manquantes dans le digest quotidien. **Idempotent** :
retenté automatiquement le lendemain via `memorandum` tant que la config
n'est pas complète. **Isolation par raccordement** : un enregistrement cassé
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
vers `$PREVI_NAS_DATA_ROOT/ARCHIVE/<dossier>/<AAAA>/<MM>/<JJ>/`, puis prédit
la nouvelle heure et écrit `centrales/<dossier>/prevision.json` (h8) ou fusionne
dans `enchere.json` (clés `J2`/`J3`, h48/h72). Isolation par (dossier,
horizon) + digest quotidien.

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

## Flux normal (bout en bout, memorandum → prédiction)

```bash
python cron/scripts/majdata-memo.py                        # 1. rafraîchit centrales/ depuis OneGate
python cron/scripts/maj-data.py                             # 2. importe/complète les débits
python cron/scripts/maj-automate.py                         # 3. sync + calcule le débit "automate" (haute chute)
python cron/scripts/maj-puissance.py                        # 4. importe la puissance
python cron/scripts/onboarding-check.py                     # 5. valide les raccordements pas encore onboardés
python cron/scripts/onboarding-bv.py batch                  # 6. caractérise le BV + stations/transit
python cron/scripts/train.py --mode new                     # 7. 1er entraînement des nouvelles centrales
python cron/scripts/train.py --mode monthly                 # 8. réentraînement mensuel (promotion conditionnelle)
python cron/scripts/predict-archive.py                      # 9. archive + prédit la nouvelle heure
python cron/scripts/daily-sync-report.py                    # 10. mail unique de bilan (une fois par jour)
```

En production, c'est scindé sur des horaires différents (memorandum quotidien,
puissance quotidien, débit/prédiction horaires, entraînement quotidien +
mensuel) — cf. section Crontab ci-dessous, jamais un unique `dvc repro`
géant. `data_preparation` n'est jamais dans le crontab : `train.py`
(étapes 7/8) le rafraîchit lui-même, dossier par dossier, juste avant de
vérifier l'éligibilité de ce dossier précis.

## Crontab (`cron/wrappers/crontab_previ_r2d2.cron`)

Séparé de celui de Previ_v2 (`crontab_V1.cron`), chargés ensemble :
`cat crontab_V1.cron crontab_previ_r2d2.cron | crontab -`.

| Heure | Commande | Stages DVC |
|---|---|---|
| `00:00` quotidien | `dvc repro dvc.yaml:memorandum --force --keep-going` | détection nouveaux raccordements OneGate |
| `01:00` quotidien | `dvc repro dvc.yaml:puissance --force --keep-going` | fusion puissance hydrospot_stream |
| `01:15` quotidien | `dvc repro dvc.yaml:onboarding_check dvc.yaml:bv --force --keep-going` | validation + bv des nouveaux raccordements (avant `train_new`, même jour) |
| `02:00` quotidien | `dvc repro dvc/model/dvc.yaml:train_new --force --keep-going` | 1er entraînement des nouvelles centrales (12 mois d'historique atteints) |
| `03:00` le 1er du mois | `dvc repro dvc/model/dvc.yaml:train_monthly --force --keep-going` | réentraînement mensuel, promotion conditionnelle (KGE) |
| `08:00` quotidien | `maj-meteo.py && clean-meteo.py` (hors DVC) | météo NWP FTP + nettoyage |
| `:00` toutes les heures | `dvc repro dvc.yaml:debit dvc.yaml:debit_automate dvc/postprocessing/dvc.yaml:predict_archive --force --keep-going` | débit + débit automate + prédiction/archivage horaire (**un seul appel** — plusieurs `dvc repro` séparés au même horaire se bloquent sur le lock projet) |
| `23:55` quotidien | `dvc repro dvc.yaml:daily_sync_report --keep-going` | mail unique du jour |

`--keep-going`/`-k` partout : sans ça, une panne Hub'Eau sur `debit` arrêterait
aussi `debit_automate` dans le même appel, alors qu'il n'a rien à voir avec
Hub'Eau (cf. piège "`dvc repro` arrête tout dès qu'un stage échoue" plus bas).

`data_preparation` reste hors cron (`train.py` le rafraîchit lui-même,
dossier par dossier, cf. section Pipeline — outil manuel de backfill groupé).

## Notifications — digest quotidien

Avant : chaque script envoyait un mail à chaque exécution — problématique
pour `maj-data`/`maj-automate` qui tournent toutes les heures (24 mails/jour
noyant le signal utile). Depuis 2026-07-13 : `maj-data.py`, `maj-automate.py`
et `maj-puissance.py` **ne mailent plus rien eux-mêmes** — ils journalisent
leur bilan via `previ_r2d2.common.daily_report.record(script, subject, body,
has_errors)` dans `logs/daily_sync_state/<AAAA-MM-JJ>.jsonl`.
`daily-sync-report.py` (cron unique, `23:55`) relit ce journal et envoie
**LE seul mail du jour** — toujours envoyé, même sans erreur (confirme aussi
que le sync tourne). `majdata-memo.py` garde son comportement d'origine
(alerte immédiate en cas d'échec API, hors scope de ce changement). Sans
SMTP configuré, les scripts fonctionnent et avertissent seulement qu'aucun
mail n'a pu être envoyé (`previ_r2d2.common.mailer`, inchangé).

Depuis 2026-07-17, `onboarding-check.py`/`train.py`/`predict-archive.py`
journalisent aussi via `daily_report.record(...)` — **toujours ajouter un
nouveau script à `daily_report.SCRIPT_LABELS` en même temps que son premier
appel à `record()`** : un script absent de ce dict est journalisé
correctement dans le JSONL mais **jamais rendu dans le mail envoyé**
(`build_email()` n'itère que sur `SCRIPT_LABELS`) — piège réel rencontré.

## MLflow (`common/mlflow_tracker.py`)

Tracking file-based sur NAS (pas de serveur), comme Previ_v2 — circuit
breaker (`@safe`/`MLFLOW_OK`, jamais bloquant pour le cron). Comble 2 TODO
jamais faits côté Previ_v2 : `log_prediction_hybrid` (prédiction horaire) et
`log_verification_hybrid` (vérification a posteriori). **Observabilité
seule** — jamais lu par `predict_orchestrator` (source de vérité = fichier
DVC `models/`, cf. section `train.py`). Activé via `PREVI_MLFLOW_URI` dans
`secret_config.py`/l'environnement.

## Références

- Endpoints OneGate `/hydrogrid` : skill `call-hydrogrid`.
- Hub'Eau hydrométrie : https://hubeau.eaufrance.fr/page/api-hydrometrie
