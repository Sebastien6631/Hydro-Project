---
name: hydro-mlops
description: |
  Cadrage, roadmap et SUIVI D'AVANCEMENT du projet MLOps Hydro-Project
  (projet_hydro), à deux — Xavier + Sébastien. Soutenance début novembre 2026,
  points hebdo. Utilise ce skill quand quelqu'un demande : où on en est,
  quoi faire ensuite, qui fait quoi, le plan / la roadmap, l'état
  d'avancement, les décisions de cadrage, les deadlines, la correspondance
  avec le cours DataScientest / la grille du jury, le workflow d'équipe
  (branches, dépôts GitHub + DagsHub), ou pour cocher une tâche faite dans le
  tableau de suivi. Le skill technique du code (architecture projet_hydro,
  pipeline DVC, pièges connus) est `hydro-projet`, séparé.
---

# Projet MLOps Hydro-Project — cadrage & suivi

## 1. Objectif & périmètre

Construire une **plateforme MLOps de bout en bout** autour de **projet_hydro**
(prévision de débit → puissance de centrales hydroélectriques ; modèle
hybride **LightGBM + BiLSTM + stacking** ; métrique **KGE** — Kling-Gupta
Efficiency), comme projet fil rouge de la formation DataScientest, spécialité
**MLOps**.

| Élément | Décision |
|---|---|
| Centrales | **2** : `apas_G1_G4`, `touzac_g2_G2` (`nancy_A` retirée au nettoyage) |
| Horizon | **h8 uniquement** (h48 / h72 hors périmètre — « pour ne pas se compliquer ») |
| Données | ✅ **API Open-Meteo / Météo-France + Hub'Eau** branchées (`preprocessing/meteo/open_meteo.py`) — ingestion temps réel possible (`source="live"`). `source="frozen"` (ancré sur la dernière ligne de `data_preparation.csv`) reste dispo pour une prédiction 100 % reproductible (utilisé par l'API). |
| LLM / agents | **aucun** (si un jour jugé utile → 2ᵉ évaluation obligatoire pour suivre ses perfs) |
| GPU | non disponible sur les VM DataScientest — CPU par défaut ; `model/device.py` (support GPU) = local / bonus |

**À expliquer en soutenance** : le pipeline ingère en continu via des **API
publiques** (Open-Meteo/Météo-France pour la météo, Hub'Eau pour le débit) —
plus aucune dépendance au SI de l'entreprise. Pour la démo et les tests,
l'API d'inférence rejoue une prévision **100 % reproductible** depuis le
dataset versionné (`source="frozen"`) ; le temps réel reste disponible
(`source="live"`).

## 2. Principe directeur : AU PLUS SIMPLE

- **Aucune obligation d'outil** (validé avec le prof). On prend le plus
  simple qui fait le job ; on **documente** et **justifie** chaque choix
  (c'est un critère du jury).
- **Discipline `ponytail`** pour le **code** : YAGNI → déjà dans le repo →
  stdlib → feature native → dépendance déjà installée → une ligne → sinon le
  strict minimum. **Suppression > ajout.** Marquer une simplification
  volontaire avec un commentaire `ponytail:`. *(Ne s'applique PAS aux docs
  de cadrage / slides : là, complétude > concision.)*
- **On repart de zéro** sur `main` (= cœur ML nettoyé de Sébastien) pour la
  couche MLOps, dans l'ordre : **Docker d'abord**, puis le reste. Une archive
  des prototypes (07–09/09) est sur le Bureau
  (`mlops-prototypes-archive.bundle`) — à consulter ponctuellement si on
  bloque sur un point déjà résolu là-dedans, jamais une obligation.
- **RGPD / sécurité / éthique** : à intégrer **dès la phase 1** (critère
  jury). Ici : données publiques Hub'Eau (pas de données perso), sécuriser
  l'API (auth, pas de stack trace exposée), noter les limites du modèle.

## 3. Dépôts & remotes

| | URL | Rôle |
|---|---|---|
| GitHub | `github.com/Sebastien6631/Hydro-Project` | git canonique + CI GitHub Actions |
| DagsHub | `dagshub.com/Sebastien6631/Hydro-Projet` | **miroir git** + **remote DVC** (données + modèles) |

**On pousse sur les DEUX** en un seul geste (multi-URL push) :

```bash
git remote set-url origin https://github.com/Sebastien6631/Hydro-Project.git
git remote set-url --add --push origin https://github.com/Sebastien6631/Hydro-Project.git
git remote set-url --add --push origin https://dagshub.com/Sebastien6631/Hydro-Projet.git
# git push origin <branche>  ->  met à jour GitHub ET DagsHub
```

`.dvc/config` garde `dagshub` comme remote DVC. Chacun configure son token
DagsHub en local : `dvc remote modify dagshub --local user <pseudo> --local
password <token>` (ou via `.env` + `entrypoint.sh` dans le conteneur).

## 4. Workflow d'équipe

```
main   ← prod : uniquement du code validé à deux
 └─ dev ← intégration commune (copie de main)
     ├─ phase1/docker        phase1/api ...
     └─ phase2/mlflow        phase3/airflow ...
```

- Branches nommées **`phaseN/<sujet>`** (une tâche = une branche), partant de `dev`.
- Flux : `git push origin phaseN/<sujet>` → **PR vers `dev`** → re-test sur
  `dev` → merge `dev → main` quand le livrable de phase est validé à deux.
- Tests verts avant merge. Revue croisée quand c'est possible.
- **À chaque phase implémentée : mettre le `README.md` à jour** — commandes +
  explication de ce que la phase apporte (une section par phase).
- Après un entraînement qui promeut un modèle : `git push` **et** `dvc push`
  (`promote_model` committe/tague en local — cf. skill `hydro-projet`).
- Commits : `type(scope): résumé` (`feat` `fix` `docs` `chore` `test`
  `refactor`). Si Claude a aidé, finir par
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- **Historique Git propre + répartition des tâches visible** = critère jury.

## 5. Stack cible (minimale, justifiée)

| Besoin | Outil retenu | Alternative écartée & pourquoi |
|---|---|---|
| Env partagé reproductible | **Docker Compose** | conda local (pas portable) |
| Données + modèles versionnés | **DVC + DagsHub** | déjà en place, imposé de fait |
| Reverse proxy | **NGINX** | — (au programme) |
| Stockage objet artefacts | **MinIO** (ou FS local si mono-poste) | S3 cloud (pas de compte) |
| Suivi d'expériences + registre | **MLflow + Model Registry** | Weights & Biases (optionnel) |
| Orchestration | **Airflow** | **Prefect** (essayé puis abandonné : Airflow est au programme et Sébastien le connaît) |
| Serving | **BentoML** et/ou **FastAPI** | — (BentoML au programme MLOps) |
| Monitoring | **Prometheus + Grafana** + **Evidently** (dérive) | — (au programme) |
| CI | **GitHub Actions** | Jenkins (optionnel) |
| Scalabilité | **Kubernetes** (Helm) | — (au programme, critère jury) |
| Cloud | déploiement documenté a minima | — (pas de crédits cloud) |

**Non retenu** : ZenML (spé optionnelle), LLM/agents.

---

## 6. Roadmap — 4 phases

Plan de l'équipe (condensé du fil rouge officiel) :

### Phase 1 — Fondations & Conteneurisation
- Définir les objectifs du projet et les métriques clés (KGE)
- Environnement de développement reproductible (**Docker**)
- Collecter et prétraiter les données
- Modèle ML de base + évaluation + **tests unitaires**
- **API d'inférence basique**
- *(+ RGPD / sécurité / éthique — à amorcer ici)*

### Phase 2 — Microservices, Suivi & Versioning
- Suivi des expériences avec **MLflow**
- Versioning des données et des modèles (**DVC**)
- Décomposer en **microservices** + orchestration simple
- *(+ **NGINX** reverse proxy, + **MinIO**)*

### Phase 3 — Orchestration & Déploiement
- **Orchestration de bout en bout** (Airflow)
- Pipeline **CI**
- Optimiser et **sécuriser l'API**
- Scalabilité **Docker / Kubernetes**
- *(+ **BentoML** serving, + début du **drift monitoring**)*

### Phase 4 — Monitoring & Maintenance
- Monitoring des performances **Prometheus / Grafana** *(+ seuils d'alerte)*
- Détection de dérive **Evidently**
- Mises à jour automatisées du modèle et des composants
  *(= réentraînement sur le dataset figé + promotion conditionnelle KGE)*
- **Documentation technique** finale *(+ cloud deployment documenté)*

## 7. Correspondance cours officiel + grille jury

**Livrables datés** (semaines de formation — à confirmer avec le mentor,
soutenance visée « début novembre ») :

| Phase | Livrable officiel | Deadline |
|---|---|---|
| 1 — Foundations | objectifs/métriques documentés · env Docker opérationnel · API REST fonctionnelle | sem. 8 |
| 2 — Versioning | tracking MLflow + modèle au registry · versioning DVC · infra MinIO | sem. 12 |
| 3 — Deployment | pipeline Airflow (entraînement auto) · service BentoML · dashboard Prometheus/Grafana | sem. 16–17 |
| 4 — Production | full monitoring · drift detection · déploiement cloud (Kubernetes) | sem. 20–22 |
| Soutenance | 20 min présentation + **démo live** + 10 min Q&A jury | sem. 24 |

**Grille d'évaluation du jury** (l'architecture doit couvrir la majorité) :
proposition claire + cas d'usage · doc technique complète (README + slides) ·
données propres / traitées / **versionnées** · sélection & validation du
modèle **justifiées** · **résultats reproductibles** · code versionné et
organisé sur GitHub · infra **conteneurisée + orchestrée** · **serving
opérationnel** · monitoring système **et** ML · **détection de dérive** ·
infra **scalable (Kubernetes)** · **sécurité des APIs** · **RGPD & éthique** ·
stratégie de déploiement documentée · **collaboration d'équipe** (historique
Git, répartition) · suivi d'expériences documenté (MLflow) · plan de
maintenance/évolution esquissé.

**Certification** : ≥ 8/10 modules tronc commun + ≥ 3/6 modules spé + majorité
de la grille projet + soutenance d'une solution fonctionnelle.

---

## 8. TABLEAU DE SUIVI

**Légende** : ✅ fait (`dev`/`main`) · 🔄 en cours · ⬜ à faire ·
⏸️ décidé plus tard / simplifié · 🧪 exemplaire dans l'archive Bureau (à
consulter, pas à copier tel quel).

### État de base — `main` (cœur ML nettoyé par Sébastien, 09/09/2026)

| Élément | État | Détail |
|---|---|---|
| Cœur ML (LightGBM + BiLSTM + stacking) | ✅ | `src/projet_hydro/model/` |
| Preprocessing (débit, BV, data_preparation, météo) | ✅ | `src/projet_hydro/preprocessing/` |
| Pipeline DVC complet (3 + 1 + 1 stages) | ✅ | `debit → onboarding_check → data_preparation` puis `train` puis `predict_archive` |
| Reproductibilité (graines fixées) | ✅ | `model/seeding.py` |
| Support GPU + early stopping | ✅ | `model/device.py` (GPU = local seulement) |
| Promotion de modèle (`dvc add` + tag git) | ✅ | `model/pipeline/promotion.py` |
| Tests unitaires | ✅ | `pytest -q` |
| Modes d'entraînement automatisés | ❌ retirés | à re-brancher via Airflow (phase 3) |
| **Docker / Compose** (env conteneurisé) | ✅ | `Dockerfile` + `docker-compose.yml` + `entrypoint.sh` — `docker compose run --rm app` → 340 tests verts. Section README « Conteneurisation ». |
| MLflow · API · validation données · CI · monitoring · k8s | ⬜ | **à construire** |

### Phase 1

| # | Tâche | Qui | État | Notes |
|---|---|---|---|---|
| 1.1 | Objectifs + métrique KGE documentés | xh | ✅ | `docs/cadrage/README.md` §2 et §6 |
| 1.2 | **Docker / Compose** (env reproductible) | xh | ✅ | mergé sur `dev`+`main`. `docker compose run --rm app` → tests verts. Doc : README §Conteneurisation. |
| 1.3 | Collecte + prétraitement des données | sg | ✅ | débit Hub'Eau + météo Open-Meteo/Météo-France (`preprocessing/meteo/open_meteo.py`) |
| 1.4 | Modèle de base + évaluation + tests | — | ✅ | |
| 1.5 | **Validation des données** (contrat `data_preparation.csv`) | xh | ✅ | `phase1/validation` → PR vers `dev`. `validation.py` hand-rolled (erreurs vs warnings), `cron/scripts/validate-data.py`, stage DVC `validate` intercalé `data_preparation → validate → train`. 13 tests. Doc : README §Validation des données. |
| 1.6 | **API d'inférence** (FastAPI) | xh | ✅ | Mergée sur `dev`+`main` (`6833661`). `/health`, `/models`, `/predict` (dossier, h8, `source="frozen"`). Service `api` dans compose (port 8000). 5 tests boîte noire. Doc : README §API d'inférence. |
| 1.7 | RGPD / sécurité / éthique — amorce | xh | ✅ | `docs/cadrage/README.md` §4.5 (RGPD : données publiques, 0 donnée perso) et §9 (sécu API, limites modèle, éthique) |
| 1.8 | Document de cadrage | xh | ✅ | `docs/cadrage/README.md` — contexte, périmètre, données, modèle, KGE, archi 4 phases, risques, équipe. À relire par sg. |

### Phase 2

| # | Tâche | Qui | État | Notes |
|---|---|---|---|---|
| 2.1 | Suivi d'expériences MLflow | libre | ⬜ 🧪 | tracking + params + métriques + artefacts |
| 2.2 | Model Registry + promotion (alias `@production`) | libre | ⬜ 🧪 | câblé sur `promotion.py` |
| 2.3 | Versioning données + modèles | — | ✅ | DVC + tags git |
| 2.4 | Découpage microservices | libre | ⬜ | services `api` / `mlflow` / `db` / `minio` dans compose |
| 2.5 | NGINX reverse proxy | libre | ⬜ 🧪 | point d'entrée unique |
| 2.6 | MinIO (artefacts MLflow) | libre | ⬜ | ou FS local si on simplifie |

### Phase 3

| # | Tâche | Qui | État | Notes |
|---|---|---|---|---|
| 3.1 | **Airflow** — orchestration bout-en-bout | libre | ⬜ 🧪 | DAGs `BashOperator` → `cron/scripts/` ; DAG entraînement auto |
| 3.2 | Pipeline CI (`ruff` + `pytest` sur PR) | libre | ⬜ 🧪 | GitHub Actions |
| 3.3 | Sécuriser + optimiser l'API | libre | ⬜ 🧪 | auth, logs, timeouts, pas de stack trace |
| 3.4 | BentoML — service de serving | libre | ⬜ | `bentoml build` + `containerize` |
| 3.5 | Scalabilité Docker / Kubernetes (Helm) | libre | ⬜ 🧪 | Deployment + Service + Ingress + HPA |

### Phase 4

| # | Tâche | Qui | État | Notes |
|---|---|---|---|---|
| 4.1 | Prometheus + Grafana + **seuils d'alerte** | libre | ⬜ 🧪 | dashboards provisionnés + règles d'alerte |
| 4.2 | Détection de dérive Evidently | libre | ⬜ 🧪 | dérive features d'entrée vs fenêtre d'entraînement |
| 4.3 | Mises à jour automatisées du modèle | libre | ⬜ | réentraînement + promotion KGE, planifié Airflow |
| 4.4 | Déploiement cloud (documenté a minima) | équipe | ⏸️ | pas de crédits cloud — stratégie décrite |
| 4.5 | Documentation technique finale | équipe | ⬜ 🧪 | `ARCHITECTURE.md` + `MLOPS.md` (brique → cours) |
| 4.6 | **API Météo France + Hub'Eau** — dé-figer les données | sg | ✅ | `feat/meteo` ; Open-Meteo (modèles Météo-France), **pas de clé d'API nécessaire** ; prédiction live validée. ⚠️ **A changé la forme de sortie** : `niveau0` a disparu (aucun modèle MF ne l'expose, ni ERA5) → 45 → 38 colonnes, anciens modèles incompatibles, réentraînement complet. Historique ramené à 3,8 ans (2022-11-15) pour rester homogène — l'archive ERA5 donne 3× plus de pluie que MF. |

### Transverse

| # | Tâche | Qui | État | Notes |
|---|---|---|---|---|
| T1 | Données DVC poussées sur DagsHub | sg | ✅ | vérifier avec un `dvc pull` propre |
| T2 | `main` = `dev` (base = cœur nettoyé) | sg | ✅ | |
| T3 | Ce skill de cadrage + tableau | xh | 🔄 | pousser sur GitHub + DagsHub |
| T4 | Remotes multi-URL (GitHub + DagsHub) configurés chez chacun | chacun | ⬜ | cf. §3 |
| T5 | Chacun : `.env` + token DagsHub + `dvc pull` OK | chacun | ⬜ | |
| T6 | Confirmer le calendrier des deadlines avec le mentor | équipe | ⬜ | sem. 8/12/16/22 vs réel |
| T7 | Slides de soutenance + démo live | équipe | ⬜ | démo = élément clé du jury |

---

## 9. Archive des prototypes

Une couche MLOps complète (Docker · MLflow + Registry · validation · API
FastAPI · CI · monitoring Evidently+Prometheus+Grafana · Helm · DAGs Airflow ·
`make demo` · `ARCHITECTURE.md` / `MLOPS.md`) a été écrite et **testée** lors
d'un run exploratoire (07–09/09/2026), puis mise de côté pour repartir propre.

`Bureau/mlops-prototypes-archive.bundle` — `git clone
mlops-prototypes-archive.bundle proto` pour la consulter.
**Ce n'est pas notre base.** À ouvrir uniquement si on bloque sur un point
déjà résolu là-dedans, et alors **re-porter au plus simple**, pas copier.

## 10. Comment mettre à jour ce tableau

1. Tu prends une ligne `⬜` → passe-la `🔄`, mets tes initiales dans **Qui**.
2. Tu la finis → `✅`, ajoute une note (branche / fichier clé).
3. Commit **ce fichier** (`docs(skill): suivi — 3.1 fait`), PR vers `dev`.
4. Doute sur qui fait quoi / sur une simplification → point hebdo ou canal
   Slack mentor.

## 11. Références

- **`hydro-projet`** (skill) — architecture technique de projet_hydro,
  pipeline DVC, pièges connus, script par script.
- **`README.md`** — installation (env conda `projet-mlops`), pipeline DVC.
- **Archive formation** : `~/Documents/DataScientest-Formation/` — cours par
  sprint, `10_Sprint_10.../01_Onboarding_MLOps/` = cadrage, modalités
  d'évaluation, projets fil rouge.
