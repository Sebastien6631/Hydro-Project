# Fiche démo — 5 minutes

> **Xavier** au clavier · **Sébastien** commente Airflow (2:30).
> **Rien n'est lancé en direct** : ni entraînement, ni DAG. On montre ce qui a déjà tourné.
> **Plan B** : si une étape plante, on montre la capture correspondante (`docs/soutenance/captures/`) et on enchaîne, avec une seule phrase.

---

## ⚠️ Règles apprises lors de la relance du 20/10

- **Ne JAMAIS lancer `train.py` à la main avant la démo.** Il rafraîchit `centrales/<dossier>/data_preparation.csv` (météo jusqu'à aujourd'hui, débit arrêté au 10/09) → `/predict` plante en 500 « out-of-bounds » (le mode figé part de la dernière ligne). Réparation : `docker compose run --rm app dvc checkout -f centrales/<dossier>.dvc`.
- **Ne pas déclencher un DAG en direct** : un run `hydro_predict` dure ~6 min, un entraînement 45-60 min sur le PC de Xavier.
- **`.env` à jour** (cf. `.env.example`) : `MLFLOW_TRACKING_URI` non vide (sinon MLflow n'enregistre rien), `MLFLOW_TRACKING_INSECURE_TLS=true`, `API_KEY` renseignée (sinon pas de 401 à montrer).
- **MLflow nécessite la PR #27** (`--allowed-hosts`), sinon chaque run est rejeté en 403.
- **Model Registry vide** tant qu'aucune promotion n'a abouti (candidats du 20/10 refusés : KGE 0,66 < 0,80). On montre le run et le refus — c'est l'exemple de promotion conditionnelle.
- Le PC de Xavier est sous **PowerShell 5.1** : `curl.exe` (pas `curl`, alias d'une autre commande), `\"` dans le JSON.

---

## 1. La veille ou le matin

```powershell
cd ~\Hydro-Project
git switch fix/mlflow-allowed-hosts      # ou dev une fois la PR #27 mergée
docker compose up -d nginx airflow grafana
```

Attendre 1-2 min, puis :

```powershell
curl.exe -k https://localhost:8443/health      # doit répondre ok
```

## 2. 30 minutes avant de passer

**Chrome, 4 onglets dans cet ordre** (certificats auto-signés déjà acceptés : « Paramètres avancés → Continuer vers localhost », une fois par port) :

| # | URL | À préparer |
|---|---|---|
| 1 | `https://localhost:5443/#/experiments/2/runs/5fe7c6164c194faf990a176e1183a31d` | run Touzac affiché, assistant MLflow fermé |
| 2 | `https://localhost:8843/dags/hydro_predict` | vue graphe, groupe « sources » déplié |
| 3 | `http://localhost:3001/d/hydro-overview?kiosk` | Grafana connecté (admin / admin, « Skip » au changement de mot de passe) |
| 4 | `http://localhost:9090/alerts` | règle `DataDrift` dépliée |

> Si l'ID du run MLflow a changé (volume recréé), ouvrir `https://localhost:5443/#/experiments/2` et cliquer sur le run `touzac_g2_G2-h8`.

**Terminal** — PowerShell, police ≥ 18 pt :

```powershell
cd ~\Hydro-Project
$env:API_KEY = ((Select-String -Path .env -Pattern '^API_KEY=').Line | Select-Object -Last 1).Split('=',2)[1]
curl.exe -s -k -X POST https://localhost:8443/predict -H "X-API-Key: $env:API_KEY" -H "content-type: application/json" -d '{\"dossier\": \"touzac_g2_G2\"}' > $null
cls
```

(La requête `/predict` « chauffe » l'API : la première réponse est plus lente.)

**Explorateur** : dossier `docs\soutenance\captures\` ouvert, pour le plan B.

## 3. Pendant la démo

| Temps | On fait | On dit |
|---|---|---|
| **0:00** | `curl.exe -k https://localhost:8443/models` | « L'API est en HTTPS et protégée : sans clé, refusé. » |
| **0:20** | `curl.exe -s -k -H "X-API-Key: $env:API_KEY" https://localhost:8443/models \| python -m json.tool` | « Avec la clé : les 2 centrales et le KGE de leur modèle en production. » |
| **0:40** | `curl.exe -s -k -X POST https://localhost:8443/predict -H "X-API-Key: $env:API_KEY" -H "content-type: application/json" -d '{\"dossier\": \"touzac_g2_G2\"}' \| python -m json.tool` | « Le débit prévu heure par heure sur 8 heures. Rejoué depuis un état figé : 100 % reproductible. » |
| **1:20** | Onglet 1 · MLflow : métriques à gauche, traçabilité à droite (commit, branche, graine) | « Chaque entraînement est tracé : réglages, scores par régime et par saison, et le code exact qui l'a produit. » |
| **2:00** | MLflow : pointer `kge_stacking` = 0,66 | « Ce candidat a fait 0,66, la production 0,80 : refusé, la production n'a pas bougé. C'est la promotion conditionnelle. » |
| **2:30** | Onglet 2 · Airflow (**Sébastien**) : la coche verte « Dernière exécution », puis le graphe | « Toutes les heures : vérification météo et débit, et seulement ensuite la prévision. Le lundi, réentraînement. » |
| **3:20** | Onglet 3 · Grafana | « Le KGE de chaque modèle en production, la latence de l'API, la charge machine. » |
| **4:10** | Onglet 4 · Prometheus | « La dérive : 45 % des colonnes ont dérivé après le changement de source météo, l'alerte s'est déclenchée sur les 2 centrales. » |
| **4:45** | Retour aux slides | « Tout ça tourne avec une seule commande Docker. » |

## 4. Si ça plante

| Problème | Réaction pendant la démo | Réparation (après) |
|---|---|---|
| Une étape ne répond pas | montrer la capture, « même chose en capture, faite ce matin », enchaîner | — |
| `/predict` renvoie 500 | capture `terminal_api_predict_1.png` | `docker compose run --rm app dvc checkout -f centrales/touzac_g2_G2.dvc` |
| Onglet « connexion non privée » | « Paramètres avancés → Continuer vers localhost » | — |
| Grafana redemande le login | admin / admin, « Skip » | — |

## 5. Captures du plan B (faites le 20/10)

| Étape | Fichier |
|---|---|
| 0:00 – 0:20 | `terminal_api_401_models.png` |
| 0:40 | `terminal_api_predict_1.png`, `terminal_api_predict_2.png` |
| 1:20 – 2:00 | `mlflow_run_touzac.jpg` |
| 2:30 | `airflow_liste_dags.jpg`, `airflow_hydro_predict_run_vert.jpg` |
| 3:20 | `grafana_dashboard.jpg` |
| 4:10 | `prometheus_alerte_datadrift.jpg` |

## 6. À vérifier à la répétition (jeudi 22/10)

- [ ] La démo tient en 5:00 chrono
- [ ] Sébastien prend la parole à 2:30
- [ ] La sortie JSON de `/predict` tient à l'écran (sinon réduire la police d'un cran)
- [ ] Les 4 onglets s'ouvrent sans avertissement de certificat
- [ ] Le DAG `hydro_predict` a des runs verts récents (il tourne toutes les heures tant que Docker est lancé)
