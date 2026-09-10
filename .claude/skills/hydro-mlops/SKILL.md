---
name: hydro-mlops
description: |
  Cadrage, roadmap et SUIVI D'AVANCEMENT du projet MLOps Hydro-Project
  (previ-R2-D2), à deux — Xavier + Sébastien. Soutenance début novembre 2026,
  points hebdo. Utilise ce skill quand quelqu'un demande : où on en est,
  quoi faire ensuite, qui fait quoi, le plan / la roadmap, l'état
  d'avancement, le workflow d'équipe (branches main / dev / perso), les
  décisions de cadrage, ou pour cocher une tâche comme faite dans le tableau
  de suivi. Le skill technique du code (architecture previ-R2-D2, pipeline
  DVC, pièges connus) est `hydro-projet`, séparé.
---

# Projet MLOps Hydro-Project — cadrage & suivi

## Objectif

Construire une **plateforme MLOps de bout en bout** autour de **previ-R2-D2**
(prévision de débit → puissance de centrales hydroélectriques ; modèle
hybride LightGBM + BiLSTM + stacking ; métrique **KGE**), comme projet de
cours DataScientest.

- **Soutenance** : début novembre 2026. **Points de suivi** : hebdomadaires.
- **Équipe** : Xavier (`xh`) + Sébastien (`sg`).
- **2 centrales** : `apas_G1_G4`, `touzac_g2_G2` (toutes `flex_strategy: DEFAULT`).
  `nancy_A` retirée au nettoyage du 09/09.

## Dépôts

| | URL | Rôle |
|---|---|---|
| **Git canonique** | `dagshub.com/Sebastien6631/Hydro-Projet` | code — on clone et on pousse ici |
| **Remote DVC** | `dagshub.com/Sebastien6631/Hydro-Projet` (`.dvc` endpoint) | données + modèles (`dvc pull` / `dvc push`) |
| Archive prototypes | `Bureau/mlops-prototypes-archive.bundle` | phases 00–08 (Prefect) testées, mises de côté — `git clone` du bundle pour piocher un bout si besoin |

> ⚠️ La **CI GitHub Actions** ne tourne que sur GitHub. Si le canonique reste
> DagsHub : soit miroir git vers GitHub pour la CI, soit CI DagsHub. **À
> trancher au prochain point.**

## Principe directeur : AU PLUS SIMPLE

- **Aucune obligation d'outil** (validé avec le prof). On prend le plus
  simple qui fait le job, on documente le choix.
- **Pas de LLM ni d'agents.** Si un jour jugé utile → 2ᵉ évaluation
  obligatoire pour suivre ses perfs → non justifié ici.
- **Discipline `ponytail`** avant d'écrire du code : YAGNI → déjà dans le
  repo → stdlib → feature native → dépendance déjà installée → une ligne →
  sinon le strict minimum. **Suppression > ajout.** Marquer une
  simplification volontaire avec un commentaire `ponytail:`.
- **On repart de zéro** sur `sebastien/main` (cœur ML nettoyé), on refait la
  couche MLOps **au plus simple**, dans l'ordre : **Docker d'abord**, puis
  les autres phases. Une archive des prototypes (07–09/09) est sur le Bureau
  (`mlops-prototypes-archive.bundle`) — s'en inspirer ponctuellement si utile,
  jamais une obligation.

## Stack cible (minimale)

| Besoin | Outil | Note |
|---|---|---|
| Env partagé reproductible | **Docker Compose** | un `.env` chacun |
| Données + modèles versionnés | **DVC + DagsHub** | déjà en place |
| Suivi d'expériences + registre | **MLflow + Model Registry** | backend PostgreSQL ; artefacts FS local (MinIO seulement si besoin cloud) |
| API d'inférence | **FastAPI** | `/predict`, `/health`, `/metrics` |
| Orchestration | **Airflow** | DAGs = `BashOperator` sur les `cron/scripts/` existants |
| Monitoring | **Prometheus + Grafana + Evidently** | dérive + KGE en ligne + alertes |
| CI | **GitHub Actions** (ou DagsHub) | `ruff` + `pytest` sur PR |

**Écarté / plus tard** : Prefect (→ Airflow), Kubernetes (prototype Helm
existe, hors chemin critique), BentoML, ZenML, W&B, Jenkins.

## Workflow d'équipe

```
main   ← prod : uniquement du code validé à deux
 └─ dev ← copie de main, intégration commune
     ├─ feat/xh-<sujet>   (Xavier)
     └─ feat/sg-<sujet>   (Sébastien)
```

- Chacun bosse **sur sa branche** partant de `dev`.
- Partager l'avancement : `git push` sa branche + **PR vers `dev`**.
  `ruff` clean + tests verts avant merge.
- `dev → main` quand un jalon est validé ensemble.
- Après un entraînement qui promeut : `git push` **et** `dvc push`
  (`promote_model` committe/tague en local — cf. skill `hydro-projet`).
- Commits : `type(scope): résumé` (`feat` `fix` `docs` `chore` `test`
  `refactor`). Si Claude a aidé, finir par
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

## Répartition & ordre

1. **Xavier → Docker / conteneurisation** (`feat/xh-docker`) — c'est le
   préalable, on merge sur `dev` avant d'enchaîner.
2. Puis les phases suivantes en parallèle (MLflow, validation, API,
   **Airflow**, monitoring, CI) — répartition au fil des points hebdo via le
   tableau ci-dessous. Chacun prend une ligne `⬜`, la passe `🔄` + initiales.

---

## Roadmap (4 phases du cours) + TABLEAU DE SUIVI

Roadmap officiel : **1** Fondations & Conteneurisation · **2** Microservices,
Suivi & Versioning · **3** Orchestration & Déploiement · **4** Monitoring &
Maintenance.

**Légende** : ✅ fait (sur `dev`/`main`) · 🔄 en cours · ⬜ à faire ·
⏸️ optionnel / plus tard · 🧪 un exemplaire archivé existe (bundle Bureau) —
à consulter pour s'inspirer, pas à copier tel quel.

### État de base — `sebastien/main` (cœur ML nettoyé, 09/09/2026)

| Élément | État | Détail |
|---|---|---|
| Cœur ML (LightGBM + BiLSTM + stacking) | ✅ | `src/previ_r2d2/model/` — intact |
| Preprocessing (débit, BV, data_preparation, météo) | ✅ | `src/previ_r2d2/preprocessing/` |
| Pipeline DVC preprocessing | ✅ | `dvc/preprocessing/dvc.yaml` (4 stages) |
| Reproductibilité (graines fixées) | ✅ | `src/previ_r2d2/model/seeding.py` |
| Support GPU + early stopping | ✅ | `src/previ_r2d2/model/device.py` |
| Promotion de modèle | ✅ | `promotion.py` — `dvc add` + tag git, invocation DVC robuste |
| Tests unitaires | ✅ | `pytest -q` (env conda `projet-mlops`) |
| Modes d'entraînement automatisés (`--mode new/monthly`) | ❌ retirés | à re-décider avec l'orchestration |
| Docker / MLflow / API / validation données / CI | ⬜ | **à construire** (prototypes dispo, voir 🧪) |

### Phase 1 — Fondations & Conteneurisation

| # | Tâche | Qui | État | Notes |
|---|---|---|---|---|
| 1.1 | Objectifs + métrique clé (KGE) | équipe | ✅ | horizons h8/h48/h72 |
| 1.2 | Env de dev reproductible → **Docker Compose + `.env`** | xh | ⬜ 🧪 | proto : `Dockerfile.base`, `docker-compose.yml`, `Makefile`, `pydantic-settings` |
| 1.3 | Collecte + prétraitement des données | — | ✅ | cœur existant |
| 1.4 | Modèle de base + évaluation + tests unitaires | — | ✅ | |
| 1.5 | **Validation des données** (contrat `data_preparation.csv`) | libre | ⬜ 🧪 | proto : `validation.py` + stage DVC `validate` (erreurs vs warnings, hand-rolled) |
| 1.6 | API d'inférence basique (FastAPI) | libre | ⬜ 🧪 | proto : `src/previ_r2d2/serving/` complet |

### Phase 2 — Microservices, Suivi & Versioning

| # | Tâche | Qui | État | Notes |
|---|---|---|---|---|
| 2.1 | Suivi d'expériences MLflow | libre | ⬜ 🧪 | proto : `model/tracking/mlflow_tracking.py` (no-op si pas de serveur) |
| 2.2 | Model Registry + promotion (alias `@production`) | libre | ⬜ 🧪 | proto : câblé sur `promotion.py` |
| 2.3 | Versioning données + modèles | — | ✅ | DVC + tags git `<dossier>-h<h>-v<N>` |
| 2.4 | Découpage en microservices | libre | ⬜ | services séparés : `api` / `mlflow` / `db` (proto compose) |
| 2.5 | Orchestration simple | libre | ⬜ | → **Airflow** (voir 3.1) |

### Phase 3 — Orchestration & Déploiement

| # | Tâche | Qui | État | Notes |
|---|---|---|---|---|
| 3.1 | Orchestration bout-en-bout → **Airflow** | libre | ⬜ 🧪 | proto : 4 DAGs `BashOperator` (hourly/daily/weekly/monthly) + compose Airflow. Extraire `weekly-monitoring` en script `cron/scripts/` |
| 3.2 | Pipeline CI (`ruff` + `pytest` sur PR) | libre | ⬜ 🧪 | proto : `.github/workflows/ci.yml` (Python natif) + `release.yml` |
| 3.3 | Optimiser + sécuriser l'API | libre | ⬜ 🧪 | proto : logs JSON, `X-Request-ID`, JWT optionnel |
| 3.4 | Scalabilité (Docker / k8s) | — | ⏸️ 🧪 | proto : chart Helm. Hors chemin critique à 2 users |

### Phase 4 — Monitoring & Maintenance

| # | Tâche | Qui | État | Notes |
|---|---|---|---|---|
| 4.1 | Monitoring perfs (Prometheus + Grafana) | libre | ⬜ 🧪 | proto : exporter `:9200` + 3 dashboards provisionnés + 6 alertes |
| 4.2 | Détection de dérive (Evidently) | libre | ⬜ 🧪 | proto : `monitoring/drift.py` (DataDriftPreset + repli KS) + `online_perf.py` |
| 4.3 | Mises à jour automatisées du modèle | libre | ⬜ | réentraînement + promotion conditionnelle KGE, planifié via Airflow |
| 4.4 | Documentation technique finale | libre | ⬜ 🧪 | proto : `ARCHITECTURE.md` + `MLOPS.md` (brique → cours) |

### Transverse

| # | Tâche | Qui | État | Notes |
|---|---|---|---|---|
| T1 | Données DVC poussées sur DagsHub | sg | ✅ | vérifier avec un `dvc pull` propre |
| T2 | `main` = `dev` (base = cœur nettoyé) | sg | ✅ | |
| T3 | Ce skill de cadrage + tableau | xh | 🔄 | `.claude/skills/hydro-mlops/SKILL.md` |
| T4 | Chacun : `.env` + token DagsHub + `dvc pull` OK | chacun | ⬜ | |
| T5 | Canonique DagsHub vs GitHub + CI → trancher | équipe | ⬜ | cf. encadré « Dépôts » |
| T6 | Soutenance : slides + démo bout-en-bout | équipe | ⬜ | début novembre |

---

## Archive des prototypes

Une couche MLOps complète (phases 00–08, orchestration Prefect) a été écrite
et **testée** lors d'un run exploratoire (07–09/09/2026), puis mise de côté
pour repartir propre sur le cœur nettoyé de Sébastien.

`Bureau/mlops-prototypes-archive.bundle` — `git clone
mlops-prototypes-archive.bundle proto` pour la consulter. Contenu : Docker
socle · MLflow + Registry · validation données · API FastAPI · CI · monitoring
(Evidently + Prometheus + Grafana) · Helm · DAGs Airflow · `make demo` ·
`ARCHITECTURE.md` / `MLOPS.md` / `docs/mlops/PHASE-0X-*.md`.

**Ce n'est pas notre base** — on refait au plus simple. À ouvrir uniquement
si on bloque sur un point déjà résolu là-dedans.

## Comment mettre à jour le tableau

1. Tu prends une tâche `⬜` → passe-la `🔄`, mets tes initiales dans **Qui**.
2. Tu la finis → `✅`, ajoute une note (branche / fichier clé).
3. Commit **ce fichier** (`docs(skill): suivi — 3.1 fait`), PR vers `dev`.
4. En cas de doute sur qui fait quoi → point hebdo.

## Références

- **`hydro-projet`** (skill) — architecture technique de previ-R2-D2,
  pipeline DVC, pièges connus, script par script.
- **`README.md`** — installation (env conda `projet-mlops`), pipeline DVC.
- **Roadmap officiel du cours** — 4 phases ci-dessus, soutenance début nov.
