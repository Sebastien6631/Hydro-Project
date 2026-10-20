# Démo live — 5 minutes

> Pilote : **XH** au clavier · **SG** commente. Tout est **pré-lancé avant la soutenance** : on ne lance rien en direct.
> Plan B : si quoi que ce soit plante, on bascule sur les captures (slides cachées en fin de deck), sans s'excuser plus d'une phrase.

## ⚠️ Règles apprises lors de la relance du 20/10

- **Ne JAMAIS lancer `train.py` à la main avant la démo.** Il rafraîchit `centrales/<dossier>/data_preparation.csv` (météo jusqu'à aujourd'hui, débit arrêté au 10/09) → `/predict` plante en 500 « out-of-bounds » (le mode figé part de la dernière ligne). Réparation : `docker compose run --rm app dvc checkout -f centrales/<dossier>.dvc`.
- **Ne pas déclencher un DAG en direct** : un run `hydro_predict` dure ~6 min, un entraînement 45-60 min sur le PC de Xavier. On montre les runs déjà faits.
- **`.env` à jour** (cf. `.env.example`) : `MLFLOW_TRACKING_URI` non vide (sinon MLflow n'enregistre rien), `MLFLOW_TRACKING_INSECURE_TLS=true`, `API_KEY` renseignée (sinon pas de 401 à montrer).
- **MLflow nécessite la PR #27** (`--allowed-hosts`), sinon chaque run est rejeté en 403.
- **Model Registry vide** tant qu'aucune promotion n'a abouti (les candidats du 20/10 ont été refusés : KGE 0,66 < 0,80). On montre le run et on **explique** l'alias `@production` (cf. déroulé 2:00) — le refus du candidat est d'ailleurs un bon exemple de promotion conditionnelle.

## Avant de passer (T-30 min)

```bash
docker compose up -d nginx airflow grafana     # démarre api, mlflow, minio, prometheus, node-exporter
curl -k https://localhost:8443/health          # doit répondre ok
```

- Onglets de navigateur ouverts **dans cet ordre**, certificats auto-signés **déjà acceptés** (« Paramètres avancés → Continuer vers localhost », une fois par port) :
  1. `https://localhost:5443/#/experiments/2` — MLflow, expérience `projet_hydro`, mode *Model training*
  2. `https://localhost:8843/dags/hydro_predict` — Airflow, vue graphe
  3. `http://localhost:3001/d/hydro-overview?kiosk` — Grafana (connecté)
  4. `http://localhost:9090/alerts` — Prometheus, `DataDrift` déplié
- Un terminal avec une police grande (≥ 18 pt), les commandes ci-dessous **prêtes dans l'historique**
- Lancer une fois `check-drift.py` pour avoir des rapports de dérive frais
- Envoyer quelques `/predict` à l'avance pour que les courbes Grafana ne soient pas vides

## Déroulé

| Temps | Écran | Action | À dire |
|---|---|---|---|
| 0:00 | Terminal | `curl -k https://localhost:8443/models` → **401** | « L'API est en HTTPS et protégée : sans clé, refusé. » |
| 0:20 | Terminal | même commande avec `-H "X-API-Key: …"` → 200, les 2 centrales | « Avec la clé, on voit les modèles disponibles. » |
| 0:40 | Terminal | `POST /predict {"dossier": "touzac_g2_G2"}` (+ clé) | « Voilà le débit prévu heure par heure sur 8 heures. Rejoué depuis un état figé : 100 % reproductible. » |
| 1:20 | MLflow | Runs → un run : paramètres, KGE, artefacts | « Chaque entraînement est tracé : réglages, scores, fichiers du modèle stockés dans MinIO. » |
| 2:00 | MLflow | sur le run : `kge_stacking` 0,66 | « Ce candidat a fait 0,66, la production 0,80 : il a été refusé, la production n'a pas bougé. C'est ça, la promotion conditionnelle — et quand un candidat gagne, il reçoit l'alias `@production` dans le registre. » |
| 2:30 | Airflow | les 2 DAGs, puis le graphe de `hydro_predict` | **SG** : « Toutes les heures : on vérifie la météo et le débit, et seulement après on prévoit. Le lundi, réentraînement. » |
| 3:20 | Grafana | dashboard : requêtes, latence, KGE par centrale, CPU | « Ici, ce qu'on vient de faire en direct : nos requêtes apparaissent. Et le KGE de chaque modèle en production. » |
| 4:10 | Prometheus `:9090` | onglet *Alerts*, règle `DataDrift` (le dashboard Grafana n'a pas de panneau dérive) | « Et la dérive : 45 % de colonnes ont dérivé après le changement de source météo. L'alerte l'a vu. » |
| 4:45 | — | retour aux slides | « Tout ça tourne avec une seule commande Docker. » |

## Commandes (à mettre dans l'historique)

PowerShell (Windows) — `curl.exe`, pas `curl` (alias PowerShell d'une autre commande) :

```powershell
cd ~\Hydro-Project
$env:API_KEY = ((Select-String -Path .env -Pattern '^API_KEY=').Line | Select-Object -Last 1).Split('=',2)[1]

curl.exe -k https://localhost:8443/models
curl.exe -k -H "X-API-Key: $env:API_KEY" https://localhost:8443/models | python -m json.tool
curl.exe -k -X POST https://localhost:8443/predict -H "X-API-Key: $env:API_KEY" -H "content-type: application/json" -d '{\"dossier\": \"touzac_g2_G2\"}' | python -m json.tool
```

Bash (Git Bash, Linux, macOS) :

```bash
export API_KEY=$(grep '^API_KEY=' .env | tail -1 | cut -d= -f2)
curl -k https://localhost:8443/models
curl -k -H "X-API-Key: $API_KEY" https://localhost:8443/models | python -m json.tool
curl -k -X POST https://localhost:8443/predict -H "X-API-Key: $API_KEY" -H 'content-type: application/json' -d '{"dossier": "touzac_g2_G2"}' | python -m json.tool
```

> La clé est lue dans `.env`, jamais tapée ni affichée à l'écran. Vérifier à la répétition que la sortie JSON tient à l'écran.

## Plan B — captures faites le 20/10 (`docs/soutenance/captures/`)

Faites : `mlflow_run_touzac.jpg`, `airflow_liste_dags.jpg`, `airflow_hydro_predict_run_vert.jpg`, `grafana_dashboard.jpg`, `prometheus_alerte_datadrift.jpg`. Terminal (PowerShell) : `terminal_api_401_models.png` (refus sans clé puis `/models`), `terminal_api_predict_1.png` + `_2.png` (prévision Touzac sur 8 h). Pas de capture du registre (vide, cf. règles ci-dessus).

Liste initiale :

1. Terminal : 401 puis 200 sur `/models`
2. Terminal : réponse de `/predict`
3. MLflow : liste des runs
4. MLflow : détail d'un run (paramètres + métriques)
5. MLflow : Model Registry avec l'alias `@production`
6. Airflow : liste des DAGs
7. Airflow : graphe de `hydro_predict` avec un run vert
8. Grafana : dashboard complet
9. Prometheus : onglet *Alerts* (règle `DataDrift`)

Ces captures servent **aussi** de visuels pour les slides 8, 9 et 10.

## À vérifier à la répétition (jeudi 22/10)

- [ ] La démo tient en 5:00 chrono
- [ ] Le `/predict` répond en moins de quelques secondes (sinon : le lancer une fois avant pour chauffer le cache)
- [ ] L'alerte `DataDrift` est bien visible dans Prometheus *Alerts* (4ᵉ onglet à ouvrir : `http://localhost:9090/alerts`)
- [ ] Le run Airflow est vert sur `hydro_predict` (si l'API Hub'Eau est en panne le jour J, c'est le check bloquant qui s'arrête : **le montrer comme une fonctionnalité**, pas comme une panne)
- [ ] La clé API est dans une variable : on ne la tape pas à l'écran
