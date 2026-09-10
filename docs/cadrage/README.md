# Document de cadrage — Plateforme MLOps *previ-R2-D2*

> Projet fil rouge — formation DataScientest, spécialité **MLOps**.
> Équipe : Xavier Henry, Sébastien G. — soutenance début novembre 2026.
> Ce document couvre les points 1.1 (objectifs & métrique), 1.7 (RGPD /
> sécurité / éthique) et 1.8 (cadrage) de la phase 1.

---

## 1. Contexte & cas d'usage

**previ-R2-D2** est la version « projet de cours » d'un pipeline de
production de **Barthe EnR** : prévision du **débit entrant** puis de la
**puissance turbinable** de centrales hydroélectriques au fil de l'eau.

Enjeu métier côté production :

- **Enchères marché J+1** — l'exploitant s'engage la veille sur un profil de
  production horaire ; une prévision de débit fiable évite les écarts
  pénalisés.
- **Anticipation hydraulique** — repérer une montée de débit (épisode
  pluvieux amont) plusieurs heures avant qu'elle n'atteigne la prise d'eau.

Le cas d'usage retenu pour le projet : **servir une prévision de débit
horaire à 8 heures (`h8`) pour 2 centrales**, via une API, à partir de
données publiques, avec toute la chaîne MLOps autour (versioning,
reproductibilité, orchestration, monitoring).

Ce dépôt est une version **réduite et publique**, réalisée **hors du réseau
de l'entreprise** : les modules dépendant du SI interne (OneGate,
hydrospot_stream, automate, digest mail) ont été retirés et remplacés par
des sources publiques ou des données figées.

---

## 2. Objectifs

### 2.1 Objectif ML

Prévoir la série de **débit entrant horaire** `q_entrant_m3s[t+1 … t+8]`
d'une centrale, à partir de :

- son historique de débit,
- le débit des stations **amont** (avec transit),
- la **météo** sur le bassin versant (précipitation, température, …).

Sortie exploitée : `q_stacking_m3s` (débit prévu par le modèle) et la
conversion en puissance turbinable.

### 2.2 Objectif MLOps

Construire une **plateforme de bout en bout** démontrant, sur ce cas
d'usage, la maîtrise des briques du référentiel : environnement
reproductible, données et modèles versionnés, suivi d'expériences, serving,
orchestration, CI, monitoring système **et** ML, détection de dérive,
scalabilité, sécurité, déploiement documenté.

Principe directeur : **au plus simple** — aucune obligation d'outil, on
retient le plus léger qui fait le travail et **on justifie chaque choix**
(critère du jury). Discipline `ponytail` sur le code (suppression > ajout).

---

## 3. Périmètre

| Élément | Dans le périmètre | Hors périmètre |
|---|---|---|
| Centrales | `apas_G1_G4`, `touzac_g2_G2` (toutes `flex_strategy: DEFAULT`) | les 10 autres centrales de la version production |
| Horizon | **h8** (8 pas horaires) | h48, h72 |
| Débit | Hub'Eau / eaufrance (API publique) | — |
| Météo | API Open-Meteo / Météo-France (`preprocessing/meteo/open_meteo.py`) | acquisition FTP NWP interne |
| Modèle | hybride LightGBM + BiLSTM + stacking, déjà porté | ré-architecture du modèle |
| LLM / agents | **aucun** (si un jour jugé utile → 2ᵉ évaluation obligatoire) | — |
| GPU | CPU par défaut (VM DataScientest sans GPU) ; `model/device.py` = bonus local | — |

**Données figées / temps réel.** Le pipeline sait ingérer en continu
(`source="live"` : Hub'Eau + API météo interrogées à chaque exécution).
Pour les besoins du projet on peut aussi rejouer une prévision **100 %
reproductible** sans réseau (`source="frozen"` : relit
`data_preparation.csv`, ancré sur sa dernière ligne connue). L'API
d'inférence utilise `frozen` pour garantir des résultats reproductibles en
démo ; le mécanisme temps réel existe et reste un choix d'exécution.

---

## 4. Données

### 4.1 Sources

| Source | Contenu | Nature | Accès |
|---|---|---|---|
| **Hub'Eau / eaufrance** | débit des stations hydrométriques (référence + amont) | série temporelle, pas horaire | API REST publique, sans clé |
| **Open-Meteo / Météo-France** | précipitation, température… sur les points du bassin versant | série temporelle horaire (prévision + historique) | API publique |

Historique de référence : depuis le **01/01/2021**. Chaque centrale a un
`bv.json` (bassin versant, stations amont, points météo) **figé** — la
chaîne de délimitation automatique a été retirée (les 2 bassins sont
connus).

### 4.2 Jeu d'entraînement — `data_preparation.csv`

Un CSV par centrale, indexé par horodatage horaire (UTC, tz-naïf), colonnes :

- `debit_m3s` — **cible** (débit de la station de référence) ;
- `debit_amont[_<code>]` — débit des stations amont ;
- colonnes météo (précipitation, température, …).

Le *feature engineering* (lags, gradients, transit saisonnier, ET0, neige…)
n'est **pas** dans ce CSV : il est recalculé à l'entraînement et à la
prédiction pour éviter toute fuite.

### 4.3 Contrat de données (validation — tâche 1.5)

`preprocessing/data_preparation/validation.py` vérifie le CSV avant
l'entraînement, en séparant **erreurs** (donnée inexploitable → le pipeline
DVC s'arrête avant `train`) et **avertissements** (donnée acceptée, à
surveiller).

| Niveau | Règle |
|---|---|
| Erreur | fichier vide · `debit_m3s` absent / non numérique / entièrement vide · index non temporel, non trié, ou avec doublons · débit négatif |
| Warning | > 10 % de cible manquante · historique < 180 j · trou > 24 h dans l'index · colonne entièrement vide |

Stage DVC `validate`, intercalé `data_preparation → validate → train`.
Rapport JSON par centrale sous `logs/validation/`.

### 4.4 Versioning

Données et modèles versionnés avec **DVC**, remote sur **DagsHub**
(`dagshub.com/Sebastien6631/Hydro-Projet`). Les pointeurs `.dvc` sont dans
git ; `git checkout <tag> && dvc pull` restaure un état exact.

### 4.5 RGPD

Les données sont **exclusivement environnementales et publiques** (débits de
rivière, météo). **Aucune donnée à caractère personnel** n'est collectée,
stockée ou traitée. Le projet n'entre donc pas dans le champ du RGPD sur le
traitement de données personnelles. Les seuls secrets manipulés sont des
**jetons d'accès API** (DagsHub), gérés hors du dépôt (`.env` +
`.dvc/config.local`, tous deux gitignorés).

---

## 5. Modèle

### 5.1 Architecture — hybride à 3 étages

| Étage | Rôle | Points clés |
|---|---|---|
| **LightGBM** | régression tabulaire sur features hydro/météo | sélection de features, tuning **Optuna**, prédictions **out-of-fold** (OOF) pour le meta-learner |
| **BiLSTM** | capture la dynamique temporelle (montée / récession) | LSTM bidirectionnel + **attention**, PyTorch, CPU |
| **Stacking (Ridge)** | combine les sorties OOF LGBM + BiLSTM + contexte (débit courant, tendance, saison, crue) | régression linéaire régularisée, `alpha` figé (calibration testée puis abandonnée — cf. skill technique) |

### 5.2 Justification

- La réponse hydrologique **mélange des régimes** (étiage lent, crues
  rapides) : un modèle tabulaire (LightGBM) capte bien les relations
  non-linéaires features → débit, un modèle séquentiel (BiLSTM) capte
  l'inertie temporelle. Le stacking arbitre selon le contexte.
- Le **stacking sur prédictions OOF** évite que le meta-learner apprenne le
  sur-apprentissage des modèles de base.
- Modèle **déjà porté et testé** (issu de la production) — le projet ne le
  ré-architecture pas, il le met en condition MLOps.

### 5.3 Features (résumé)

Autorégressif débit (lags, gradients, récession, baseflow) · débit amont +
transit saisonnier · météo/hydrologie (pluie cumulée, ET0) · neige · encodage
saison / heure.

---

## 6. Métrique clé — KGE (Kling-Gupta Efficiency)

### 6.1 Définition

Le **KGE** décompose l'erreur d'une prévision hydrologique en trois termes :

```
KGE = 1 − √( (r − 1)² + (α − 1)² + (β − 1)² )

  r = corrélation linéaire   (prévu vs observé)
  α = σ_prévu / σ_observé    (rapport de variabilité)
  β = μ_prévu / μ_observé    (rapport de biais)
```

`KGE = 1` = prévision parfaite. `KGE = 0` ≈ aussi bon que la moyenne des
observations. Négatif = pire que la moyenne.

### 6.2 Pourquoi KGE

- **Standard du domaine hydrologique** (comparabilité avec la littérature et
  les pratiques métier).
- **Diagnostique** : un KGE moyen se lit — problème de corrélation ? de
  dynamique (α) ? de biais (β) ? — là où un RMSE seul ne dit pas *pourquoi*.
- Moins sensible que le NSE à la sur-pondération des forts débits.

### 6.3 Repères

| KGE | Lecture |
|---|---|
| > 0.9 | excellent |
| 0.75 – 0.9 | bon, exploitable en production |
| 0.5 – 0.75 | correct, à surveiller |
| < 0.5 | insuffisant |

### 6.4 Résultats actuels (modèles promus, schéma météo Météo-France)

| Centrale | KGE stacking h8 |
|---|---|
| `apas_G1_G4` | **0.856** |
| `touzac_g2_G2` | **0.819** |

Le KGE est aussi calculé **par pas d'horizon**, **par régime** (étiage /
normal / crue) et **par saison** pour l'analyse fine
(`model/pipeline/metrics.py`).

---

## 7. Validation & sélection du modèle

- **Split chronologique** 80 / 20 (positionnel, jamais de fuite du futur
  vers le passé) ; une tranche de validation interne sert à l'early stopping.
- **Reproductibilité** : graines fixées (`model/seeding.py`, surchargeable
  par `PREVI_SEED`) — deux entraînements à configuration identique donnent le
  même résultat au bit près (CPU).
- **Promotion conditionnelle** : un candidat n'est promu que s'il **bat le
  modèle en production sur le même holdout**
  (`model/pipeline/promotion.py` → `dvc add` + tag git
  `<dossier>-h<horizon>-v<N>`). Rollback = `git checkout <tag> && dvc pull`.
- Le modèle **en production** vit dans `models/<dossier>/h8/` (versionné) —
  jamais la zone de travail `weights/`.

---

## 8. Architecture MLOps cible

### 8.1 Les 4 phases

| Phase | Contenu | Deadline (sem. formation) |
|---|---|---|
| **1 — Fondations & conteneurisation** | objectifs & métrique · **Docker** · collecte/prétraitement · modèle de base + tests · **API d'inférence** · amorce RGPD/sécu | sem. 8 |
| **2 — Microservices, suivi & versioning** | **MLflow** + Model Registry · versioning **DVC** · découpage microservices · **NGINX** · **MinIO** | sem. 12 |
| **3 — Orchestration & déploiement** | **Airflow** (entraînement auto) · **CI** · sécurisation API · **Kubernetes** (Helm) · **BentoML** · début drift | sem. 16-17 |
| **4 — Monitoring & maintenance** | **Prometheus / Grafana** · dérive **Evidently** · réentraînement + promotion auto · doc technique finale · déploiement cloud documenté | sem. 20-22 |

### 8.2 Stack retenue & alternatives écartées

| Besoin | Outil retenu | Écarté — pourquoi |
|---|---|---|
| Env reproductible | **Docker Compose** | conda local — pas portable |
| Données + modèles versionnés | **DVC + DagsHub** | déjà en place |
| Suivi d'expériences + registre | **MLflow** | Weights & Biases — payant / hébergé |
| Orchestration | **Airflow** | Prefect — essayé puis abandonné (Airflow est au programme, connu de l'équipe) |
| Serving | **FastAPI** puis **BentoML** | — (les deux au programme) |
| Reverse proxy | **NGINX** | — |
| Stockage artefacts | **MinIO** (ou FS local si mono-poste) | S3 cloud — pas de compte |
| Monitoring | **Prometheus + Grafana + Evidently** | — |
| CI | **GitHub Actions** | Jenkins — plus lourd à héberger |
| Scalabilité | **Kubernetes** (Helm) | — (critère jury) |
| Cloud | déploiement **documenté** a minima | — pas de crédits |

### 8.3 État d'avancement (phase 1)

| # | Tâche | État |
|---|---|---|
| 1.1 | Objectifs + métrique KGE documentés | ✅ (ce document) |
| 1.2 | Environnement Docker reproductible | ✅ mergé |
| 1.3 | Collecte + prétraitement des données | ✅ (cœur existant + API météo) |
| 1.4 | Modèle de base + évaluation + tests | ✅ |
| 1.5 | Validation des données (contrat) | ✅ mergé |
| 1.6 | API d'inférence (`/health`, `/models`, `/predict`) | ✅ mergé |
| 1.7 | RGPD / sécurité / éthique — amorce | ✅ (§ 4.5 et § 9) |
| 1.8 | Document de cadrage | ✅ (ce document) |

---

## 9. Sécurité & éthique

### 9.1 Sécurité de l'API

| Sujet | Phase 1 (maintenant) | Phase 3 (prévu) |
|---|---|---|
| Authentification | — (API locale, réseau privé) | jeton / clé API |
| Fuite d'information | pas de *stack trace* renvoyée au client (erreurs converties en `HTTPException` avec message court) | logs structurés, corrélation par `request-id` |
| Déni de service | — | rate-limiting, timeouts |
| Entrées | validation Pydantic du corps `{dossier}` ; refus propre (`404`) si centrale/modèle inconnu | — |
| Secrets | jamais dans le dépôt (`.env`, `.dvc/config.local` gitignorés) | *secrets* Kubernetes |

### 9.2 Limites du modèle (à communiquer avec la prévision)

- **Pas d'extrapolation fiable hors des régimes observés** — une crue
  d'ampleur jamais vue dans l'historique sera sous-estimée (les événements
  extrêmes sont rares dans les données d'entraînement).
- **Dépendance à la qualité des sources amont** — une panne ou un trou
  Hub'Eau / API météo dégrade la prévision ; le pipeline continue mais la
  fiabilité baisse (à monitorer en phase 4).
- **Horizon court** — au-delà de h8 la prévision n'est pas produite dans ce
  périmètre.
- **Modèle figé entre deux réentraînements** — dérive possible si le régime
  hydrologique change (aménagement amont, sécheresse pluriannuelle).

### 9.3 Éthique

- Le système est un **outil d'aide à la décision** pour l'exploitant
  (engagement de production, gestion de la ressource). Il **ne pilote aucun
  organe de sécurité** : la gestion des vannes et la sûreté hydraulique
  restent sous responsabilité humaine.
- Données publiques, pas de profilage, pas d'impact sur des personnes.
- Transparence : métrique (KGE) et limites documentées, résultats
  reproductibles et versionnés.

---

## 10. Risques & mitigation

| Risque | Impact | Mitigation |
|---|---|---|
| Panne API amont (Hub'Eau / Open-Meteo) | prévision dégradée ou impossible | `source="frozen"` (rejoue le dernier état connu) ; dégradation gracieuse ; alerte monitoring (phase 4) |
| Dérive données / concept | KGE se dégrade silencieusement | drift **Evidently** + suivi KGE en ligne (phase 4) ; réentraînement + promotion conditionnelle |
| Perte de reproductibilité | impossible de rejouer un résultat du jury | Docker + DVC + tags git + graines fixées |
| Dépendance DagsHub | données/modèles inaccessibles | dépôt git canonique sur GitHub ; cache DVC restaurable ; `dvc pull` documenté (auth `basic` explicite) |
| Équipe réduite (2 pers.) | goulot sur une brique | branches `phaseN/<sujet>` indépendantes, revue croisée, tableau de suivi partagé (skill `hydro-mlops`) |

---

## 11. Organisation d'équipe

- **Dépôts** : git canonique `github.com/Sebastien6631/Hydro-Project`,
  miroir + remote DVC `dagshub.com/Sebastien6631/Hydro-Projet`. Push sur les
  deux en un geste (multi-URL).
- **Branches** : `main` (prod) ← `dev` (intégration) ← `phaseN/<sujet>`
  (une tâche = une branche). PR vers `dev`, re-test sur `dev`, merge
  `dev → main` quand le livrable de phase est validé à deux.
- **README** mis à jour à chaque phase (commandes + explication).
- **Suivi** : tableau « fait / pas fait » dans le skill `hydro-mlops`
  (`.claude/skills/hydro-mlops/SKILL.md`).
- **Soutenance** : 20 min de présentation + démo live + 10 min de Q&A,
  visée début novembre 2026 ; points hebdomadaires avec le mentor.

---

## Annexe — glossaire

| Terme | Définition |
|---|---|
| **KGE** | Kling-Gupta Efficiency — métrique hydrologique décomposant l'erreur (corrélation, variabilité, biais) |
| **OOF** | Out-Of-Fold — prédictions d'un modèle sur des données qu'il n'a pas vues à l'entraînement, servant d'entrée au stacking |
| **Stacking** | méta-modèle qui apprend à combiner les sorties de plusieurs modèles de base |
| **BV** | bassin versant — surface qui draine vers un point de la rivière |
| **Transit** | délai de propagation d'un débit d'une station amont vers la prise d'eau |
| **Débit entrant** | débit d'eau arrivant à la centrale (`q_entrant_m3s`) |
| **h8** | horizon de prévision de 8 pas horaires |
| **DVC** | Data Version Control — versioning de gros fichiers (données, modèles) couplé à git |
| **Promotion** | passage d'un modèle candidat au statut « production » après comparaison KGE sur le même holdout |
