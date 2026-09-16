# Architecture technique — projet_hydro

> Documentation technique finale (phase 4.5). Décrit l'architecture telle
> qu'elle est aujourd'hui, pas l'historique de sa construction (voir
> `README.md`, une section par phase, pour le détail chronologique et les
> commandes). Complète `docs/cadrage/README.md` (objectifs, périmètre,
> modèle, RGPD) sans le redupliquer.

## 1. Vue d'ensemble

```
                         ┌──────────────────────────────────────────┐
                         │                  nginx                    │
                         │   point d'entrée réseau unique, HTTPS      │
                         │   (certificat auto-signé, phase 3.6)       │
                         └───┬──────────────┬──────────────┬─────────┘
                             │              │              │
                    :8443    │      :5443   │      :8843   │
                    (redir   │      (redir  │      (redir  │
                    8000)    │      5000)   │      8080)   │
                             ▼              ▼              ▼
                        ┌────────┐    ┌──────────┐   ┌──────────┐
                        │  api   │    │  mlflow  │   │ airflow  │
                        │FastAPI │    │ tracking │   │standalone│
                        └───┬────┘    │+registry │   └────┬─────┘
                            │         └────┬─────┘        │
                            │              │               │ déclenche
                            │              ▼               ▼
                            │         ┌─────────┐   cron/scripts/*.py
                            │         │  minio  │   (mêmes scripts que
                            │         │(S3 objet)│   les stages DVC)
                            │         └─────────┘
                            ▼
                   models/<dossier>/h8/          data (Hub'Eau, Open-Meteo)
                   (versionné DVC+git)                      │
                            ▲                                ▼
                            └──────── train.py ◄──── data_preparation.csv
                                    (LightGBM+BiLSTM+stacking)

        Observabilité (hors nginx, admin interne) :
        prometheus (scrape api:8000/metrics) → grafana (dashboards)
        + alert_rules.yml (APIDown, ModelKGELow, HighRequestLatency, DataDrift)
        monitoring/drift.py (Evidently, K-S test) → logs/drift/*.json → /metrics
```

Tout tourne en **Docker Compose**, une seule machine (pas de cluster réel
requis — cf. §5 pour le chemin vers un vrai cluster). Le code est
bind-monté : une modification locale est vue immédiatement dans les
conteneurs, sans rebuild sauf changement de dépendances.

## 2. Composants

| Composant | Rôle | Pourquoi ce choix |
|---|---|---|
| **app** | image de base (Python 3.11, PyTorch CPU, LightGBM, DVC, `projet_hydro`) | une seule image pour tous les services applicatifs (api, tests, entraînement manuel) — pas de duplication de build |
| **api** (FastAPI) | `GET /health`, `GET /models`, `POST /predict`, `GET /metrics` | léger, async natif, suffisant seul pour 2 centrales/1 horizon (cf. BentoML écarté, README §BentoML) |
| **mlflow** | suivi d'expériences + Model Registry (alias `@production`) | SQLite backend (suffisant à 2 personnes), artefacts sur MinIO (`--serve-artifacts`) |
| **minio** | stockage objet S3-compatible pour les artefacts MLflow | remplace un volume local par un vrai stockage objet, sans compte cloud |
| **airflow** | orchestration : `hydro_predict` (horaire, h+5) et `hydro_train` (hebdo, réentraînement + promotion) | mode `standalone` (webserver+scheduler+SQLite en un processus), suffisant pour 2 DAGs |
| **nginx** | reverse proxy, point d'entrée réseau unique, TLS | routage par port (pas par sous-chemin, évite la fragilité des assets des UI tierces) |
| **prometheus** + **grafana** | monitoring système + ML (latence, KGE par centrale, dérive) | scrape direct `api:8000/metrics`, dashboard provisionné automatiquement |
| **node-exporter** | métriques machine (CPU/RAM) | complète Prometheus, image officielle standard |

**Modèle ML** (inchangé par la couche MLOps, cf. `docs/cadrage/README.md` §5) :
hybride **LightGBM** (tabulaire, Optuna+OOF) + **BiLSTM** (dynamique
temporelle, attention, PyTorch CPU) + **stacking Ridge** (arbitrage entre
les deux). Métrique : **KGE** (Kling-Gupta Efficiency), décomposée en
corrélation/variabilité/biais. Résultats actuels : `apas_G1_G4` 0.8558,
`touzac_g2_G2` 0.8192.

## 3. Flux de données

1. **Ingestion** : Hub'Eau (débit, station de référence + amont) et
   Open-Meteo (précipitation, température) — deux API publiques, aucune
   donnée personnelle.
2. **`data_preparation.csv`** : assemble débit + météo + amont par centrale,
   horodaté (UTC, tz-naïf).
3. **`validate-data.py`** : contrat de données (erreurs bloquantes vs
   avertissements), stage DVC intercalé avant l'entraînement.
4. **`train.py`** : feature engineering (lags, ET0, neige, transit
   saisonnier…) → LightGBM+BiLSTM+stacking → comparaison KGE au modèle en
   production → promotion conditionnelle (DVC `add` + tag git).
5. **`predict-archive.py`** / **API `/predict`** : charge le modèle promu
   (`models/<dossier>/h8/`), prévoit 8 pas horaires (`source="live"` ou
   `"frozen"` pour la reproductibilité en démo).
6. **Monitoring** : chaque scrape Prometheus relit `version.json` (KGE) et
   `logs/drift/*.json` (dérive, écrits par `check-drift.py`, jamais
   recalculés au scrape car un rapport Evidently prend de vraies secondes).

## 4. Sécurité

- **Clé API optionnelle** (`config.API_KEY`, en-tête `X-API-Key`) sur
  `/models` et `/predict` — `/health` et `/metrics` publics (monitoring).
  Vide par défaut (tests/CI inchangés), à durcir avant toute exposition
  réelle.
- **HTTPS** (phase 3.6) : certificat auto-signé, tout le trafic applicatif
  passe par nginx en TLS, les ports historiques (8000/5000/8080)
  redirigent en 301 vers leur équivalent HTTPS. Limite assumée : pas de
  nom de domaine réel pour ce projet de cours (documenté dans le README).
- **Rate-limit + timeouts** côté nginx (10 req/s/IP, 30s read) plutôt que
  dans l'app — nginx est déjà le point de passage unique.
- **Secrets** (`DAGSHUB_TOKEN`, `API_KEY`, `MINIO_ROOT_*`) dans `.env`
  (gitignoré, jamais commité — cf. incident corrigé en phase 3, PR #11).
- **RGPD** : aucune donnée personnelle (débit de rivière, météo publique).
  Cf. `docs/cadrage/README.md` §4.5 pour le détail.
- **Logs structurés** (JSON, `request_id` par requête) — pas de stack
  trace exposée côté client, uniquement un message d'erreur court.

## 5. Scalabilité & déploiement cloud

- **Kubernetes (Helm)** : chart `infrastructure/helm/projet-hydro/`
  (Deployment+Service+Ingress+HPA) pour l'API seule. Validé statiquement
  (`helm lint`/`template`) sans cluster réel. Limite assumée : pas de
  PVC/initContainer `dvc pull` (documenté dans le chart, `NOTES.txt`).
- **Déploiement cloud documenté a minima** (README §Déploiement cloud) :
  chaque service compose a un équivalent managé direct (conteneurs
  managés, load balancer + TLS géré, stockage objet natif, DB managée,
  Airflow managé, le chart Helm déjà prêt) — sans changement de code.
  Non implémenté : pas de crédits cloud, pas de justification à faire
  tourner le stack 24/7 pour zéro utilisateur réel (même raisonnement que
  BentoML, phase 3.4).

## 6. Tests & CI

- **345+ tests** (`pytest`), fixtures synthétiques, aucun secret requis.
- **GitHub Actions** (`ci.yml`) : `ruff check .` + `pytest -q` sur chaque
  PR et push vers `dev`/`main`, Python natif (pas de conteneur — plus
  rapide, le but de la CI est de vérifier le code, pas l'image).
- **`tests/dags/`** : charge les DAGs Airflow sans lancer Airflow
  (`importorskip`), *skipped* ailleurs (env conda, CI) — pas rouge.

## 7. Limites connues (consolidées)

| Limite | Où | Pourquoi assumée |
|---|---|---|
| Certificat HTTPS auto-signé | nginx (3.6) | pas de domaine réel pour un projet de cours |
| Pas de login Airflow (`SIMPLE_AUTH_MANAGER_ALL_ADMINS`) | airflow (3.1) | usage local, à durcir avant exposition |
| `git push`/`dvc push` manuels après le DAG d'entraînement | `hydro_train.py` (4.3) | décision d'équipe en attente (repo bind-monté vs clone dédié, token en écriture) |
| Pas de PVC/initContainer `dvc pull` dans le chart Helm | K8s (3.5) | next step documenté, pas implémenté à l'aveugle sans cluster |
| Pas d'Alertmanager (routage email/Slack) | Prometheus (4.1) | pas de canal de notification réel à câbler pour ce projet |
| Déploiement cloud non exécuté | (4.4) | pas de crédits, stratégie documentée |
| BentoML construit puis retiré | (3.4) | ne sert à rien avec une seule image — décision du tuteur |

## 8. Références

- `README.md` — une section par phase, avec les commandes.
- `docs/cadrage/README.md` — objectifs, périmètre, modèle, RGPD, KGE détaillé.
- `.claude/skills/hydro-mlops/SKILL.md` — roadmap, tableau de suivi, grille jury.
