# Démo live — 5 minutes

> Pilote : **XH** au clavier · **SG** commente. Tout est **pré-lancé avant la soutenance** : on ne lance rien en direct.
> Plan B : si quoi que ce soit plante, on bascule sur les captures (slides cachées en fin de deck), sans s'excuser plus d'une phrase.

## Avant de passer (T-30 min)

```bash
docker compose up -d nginx airflow grafana     # démarre api, mlflow, minio, prometheus, node-exporter
curl -k https://localhost:8443/health          # doit répondre ok
```

- Onglets de navigateur ouverts **dans cet ordre**, certificats déjà acceptés :
  1. `https://localhost:5443` — MLflow (Model Registry déjà affiché)
  2. `https://localhost:8843` — Airflow (liste des DAGs)
  3. `http://localhost:3001` — Grafana (dashboard *hydro-overview*, déjà connecté)
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
| 2:00 | MLflow | Model Registry → alias `@production` | « Le modèle servi, c'est celui qui porte l'alias production. On ne le déplace que si le nouveau fait mieux. » |
| 2:30 | Airflow | les 2 DAGs, puis le graphe de `hydro_predict` | **SG** : « Toutes les heures : on vérifie la météo et le débit, et seulement après on prévoit. Le lundi, réentraînement. » |
| 3:20 | Grafana | dashboard : requêtes, latence, KGE par centrale, CPU | « Ici, ce qu'on vient de faire en direct : nos requêtes apparaissent. Et le KGE de chaque modèle en production. » |
| 4:10 | Prometheus `:9090` | onglet *Alerts*, règle `DataDrift` (le dashboard Grafana n'a pas de panneau dérive) | « Et la dérive : 45 % de colonnes ont dérivé après le changement de source météo. L'alerte l'a vu. » |
| 4:45 | — | retour aux slides | « Tout ça tourne avec une seule commande Docker. » |

## Commandes (à mettre dans l'historique)

```bash
curl -k https://localhost:8443/models
curl -k -H "X-API-Key: $API_KEY" https://localhost:8443/models
curl -k -X POST https://localhost:8443/predict -H "X-API-Key: $API_KEY" -H 'content-type: application/json' -d '{"dossier": "touzac_g2_G2"}'
```

> Pour une sortie lisible, ajouter `| python -m json.tool` (vérifier à la répétition que le format tient à l'écran).

## Plan B — captures à faire mardi 20/10

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
