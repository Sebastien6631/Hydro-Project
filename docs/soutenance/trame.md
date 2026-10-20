# Trame de soutenance — previ-R2-D2 (projet_hydro)

> Oral blanc : **ven. 23/10/2026** · Soutenance : **ven. 06/11/2026, 16h**
> Format : **20 min** (démo comprise) + **10 min** de questions.
> Répartition **XH** / **SG** = proposition basée sur qui a construit quoi — à ajuster ensemble.

**Minutage total : 20:00** — 16 slides, dont la démo (5 min).

| # | Slide | Qui | Durée | Cumul |
|---|---|---|---|---|
| 1 | Titre | XH | 0:20 | 0:20 |
| 2 | Le problème métier | XH | 1:15 | 1:35 |
| 3 | Ce qu'on a construit | XH | 0:45 | 2:20 |
| 4 | Les données | SG | 1:15 | 3:35 |
| 5 | Le modèle et sa métrique | SG | 1:15 | 4:50 |
| 6 | L'architecture | XH | 1:15 | 6:05 |
| 7 | Notre règle : au plus simple | XH | 1:00 | 7:05 |
| 8 | Cycle de vie d'un modèle | SG | 1:00 | 8:05 |
| 9 | Orchestration & CI | SG | 0:50 | 8:55 |
| 10 | Monitoring & dérive | XH | 1:00 | 9:55 |
| 11 | Sécurité & RGPD | XH | 0:45 | 10:40 |
| 12 | Scalabilité & cloud | XH | 0:40 | 11:20 |
| 13 | **DÉMO** | XH pilote, SG commente | 5:00 | 16:20 |
| 14 | Ce que l'outillage nous a appris | SG | 1:15 | 17:35 |
| 15 | Et demain ? | XH | 1:30 | 19:05 |
| 16 | Équipe & merci | les deux | 0:40 | 19:45 |

Marge : 15 s. Si on déborde à l'oral blanc, on coupe d'abord les slides 9 et 12 (fusion possible avec 6).

---

## 1 — Titre

**À l'écran**
- **previ-R2-D2** — prévoir le débit d'une rivière, 8 heures à l'avance, et le faire tourner en production
- Projet MLOps · DataScientest · novembre 2026
- Xavier Henry · Sébastien G.

**À dire** — « On va vous montrer comment on a pris un modèle de prévision de débit qui existait et comment on l'a rendu exploitable de bout en bout : reproductible, versionné, surveillé. »

---

## 2 — Le problème métier

**À l'écran**
- Une centrale **au fil de l'eau** turbine l'eau qui arrive, sans réservoir
- **Marché J+1** : la veille, on s'engage sur un profil de production horaire → chaque écart est pénalisé
- **Crues** : repérer la montée du débit plusieurs heures avant qu'elle n'atteigne la prise d'eau
- → Il faut savoir **combien d'eau va arriver dans les prochaines heures**

**Visuel** — photo d'une centrale au fil de l'eau ou d'un barrage (banque d'images Canva).

**À dire** — Le cas vient d'un vrai pipeline de production (Barthe EnR). Ici c'est une version réduite et publique, faite hors du réseau de l'entreprise.

---

## 3 — Ce qu'on a construit

**À l'écran**
- **Une API** qui donne le débit prévu **heure par heure sur 8 heures** pour 2 centrales (Apas, Touzac)
- À partir de **données publiques uniquement** (débit des rivières + météo)
- Avec toute la chaîne autour : versioning, suivi des expériences, orchestration, monitoring, détection de dérive

**Hors périmètre, par choix** : les 10 autres centrales, les horizons longs (48-72 h), tout LLM / agent.

**À dire** — On n'a pas réinventé le modèle : le sujet, c'est la mise en condition de production.

---

## 4 — Les données

**À l'écran**
- **Hub'Eau** (eaufrance) : débit des stations, au point de référence et en amont — API publique
- **Open-Meteo / Météo-France** : pluie, température sur le bassin versant
- Historique depuis **janvier 2021**, pas horaire
- **Contrat de données** avant chaque entraînement :
  - erreur bloquante → le pipeline s'arrête (débit négatif, doublons, cible vide…)
  - avertissement → on continue mais on le signale (trou > 24 h, > 10 % manquant…)
- Données et modèles **versionnés avec DVC** (stockage sur DagsHub)

**Visuel** — schéma : rivière amont → station → centrale, avec un nuage météo.

**À dire** — Le contrat est un vrai stage du pipeline : si les données sont cassées, on n'entraîne pas.

---

## 5 — Le modèle et sa métrique

**À l'écran**
- Modèle hybride :
  - **LightGBM** — relations entre météo, amont et débit
  - **BiLSTM** — la dynamique dans le temps (montées, décrues)
  - **Stacking** — combine les deux
- Métrique : **KGE** (Kling-Gupta) — le standard en hydrologie
  - 1 = parfait · > 0,75 = exploitable en production
  - se décompose : corrélation · variabilité · biais → on sait **pourquoi** une prévision est mauvaise

| Centrale | KGE en production |
|---|---|
| Apas | **0,86** |
| Touzac | **0,82** |

**Visuel** — `models/apas_G1_G4/h8/plots/kge_by_step.png` (qualité selon l'heure prévue).

**À dire** — Ce qui compte : on ne promeut un nouveau modèle que s'il bat celui en production sur les mêmes données de test.

---

## 6 — L'architecture

**À l'écran** — un schéma (à redessiner proprement dans Canva) :

```
            Utilisateur / jury
                   │  HTTPS
                ┌──▼──┐
                │nginx│  porte d'entrée unique
                └┬─┬─┬┘
        ┌────────┘ │ └────────┐
     ┌──▼──┐   ┌───▼──┐   ┌───▼───┐
     │ API │   │MLflow│   │Airflow│
     └──┬──┘   └──┬───┘   └───┬───┘
        │      ┌──▼──┐        │ entraîne / prévoit
        │      │MinIO│        ▼
        ▼      └─────┘    modèles versionnés (DVC)
   Prometheus → Grafana   + Evidently (dérive)
```

- Tout tourne en **Docker Compose** — une commande pour tout lancer

**À dire** — Un service = un rôle. nginx est la seule porte vers l'extérieur.

---

## 7 — Notre règle : au plus simple

**À l'écran** — retenu vs écarté, et pourquoi :

| Besoin | Retenu | Écarté | Pourquoi |
|---|---|---|---|
| Orchestration | Airflow | Prefect | essayé puis abandonné ; Airflow est connu de l'équipe |
| Servir le modèle | FastAPI | BentoML | **construit puis retiré** : utile seulement pour plusieurs modèles dans des images séparées |
| CI | GitHub Actions | Jenkins | trop lourd à héberger à deux |
| Cloud | documenté | déployé | pas de crédits, aucun utilisateur réel à servir |

**À dire** — « Chaque outil doit justifier sa place. BentoML, on l'a codé, et on l'a retiré parce qu'il ne servait à rien ici. On préfère expliquer pourquoi plutôt que de garder du code pour la démo. »

---

## 8 — Cycle de vie d'un modèle

**À l'écran** — flèche horizontale :

`Entraînement` → `MLflow (paramètres, KGE, artefacts)` → `Comparaison avec la production` → `Promotion (alias @production + tag git)` → `API`

- Chaque entraînement est tracé dans MLflow
- **Promotion seulement si le candidat bat la production**
- Revenir en arrière : `git checkout <tag> && dvc pull` → état exact restauré
- Graines fixées : même configuration = même résultat, au bit près

**Visuel** — capture MLflow : liste des runs + Model Registry avec l'alias `@production`.

---

## 9 — Orchestration & CI

**À l'écran**
- **Airflow**, 2 DAGs :
  - `hydro_predict` — toutes les heures : vérifie la météo et le débit, puis prévoit
  - `hydro_train` — chaque lundi 2h : réentraîne puis promeut si meilleur
- **GitHub Actions** : lint + 345 tests à chaque PR
- Workflow Git : `phaseN/<sujet>` → PR → `dev` → `main`

**Visuel** — capture Airflow : les 2 DAGs + un run vert.

---

## 10 — Monitoring & dérive

**À l'écran**
- **Prometheus + Grafana**
  - système : requêtes, latence, CPU / RAM
  - ML : KGE par centrale
- **Evidently** : test de Kolmogorov-Smirnov colonne par colonne, alerte au-delà de 40 % de colonnes en dérive
- 4 règles d'alerte : API down · KGE bas · latence élevée · dérive

**Fait marquant** — Dérive mesurée en vrai : **45 %** sur les 2 centrales, après le changement de source météo. **L'alerte a vu ce qui s'est réellement passé.**

**Visuel** — capture du dashboard Grafana.

---

## 11 — Sécurité & RGPD

**À l'écran**
- **HTTPS** obligatoire (les anciens ports redirigent vers HTTPS)
- **Clé API** sur `/predict`
- Limite de débit + timeouts sur nginx
- Logs structurés, aucun détail technique renvoyé en cas d'erreur
- Secrets hors du dépôt
- **RGPD** : uniquement des données environnementales publiques → **aucune donnée personnelle**
- **Éthique** : outil d'aide à la décision — il ne pilote **aucun** organe de sécurité ; les vannes restent sous responsabilité humaine

---

## 12 — Scalabilité & cloud

**À l'écran**
- **Kubernetes** : chart Helm pour l'API (réplicas + autoscaling)
- **Cloud** : chaque service a son équivalent managé, **sans changer le code**
- Non déployé : pas de crédits, pas d'utilisateur réel → même logique que pour BentoML

---

## 13 — DÉMO (5 min)

Voir `demo.md` (scénario minute par minute + plan B en captures).

À l'écran pendant la bascule : « Démo — tout tourne en local, en Docker ».

---

## 14 — Ce que l'outillage nous a appris

**À l'écran**

| | LightGBM seul | BiLSTM seul | Stacking (en prod) |
|---|---|---|---|
| Apas | **0,88** | 0,82 | 0,86 |
| Touzac | **0,88** | 0,69 | 0,82 |

- Selon la centrale, c'est le stacking ou LightGBM seul qui gagne. Sur nos 2 centrales, **LightGBM seul l'emporte** dans tous les régimes (étiage, normal, crue)
- On l'a vu **grâce** au KGE décomposé et au suivi MLflow — sans cette chaîne, on ne l'aurait pas remarqué
- Prochaine étape, simple : comparer les 3 candidats à chaque réentraînement et **promouvoir le meilleur**, pas forcément le stacking

**À dire** — « Le but du MLOps, c'est que ce genre de chose se voie. Ici, ça s'est vu. »

**Justification orale du choix de garder le stacking (≈ 30 s)**

> « Nous n'avons pas changé de modèle, et c'est un choix. Notre sujet, c'est la chaîne MLOps autour d'un modèle hérité, pas le modèle lui-même. Le stacking est l'architecture de la production, et sur d'autres centrales c'est lui qui gagne : nos deux centrales ne suffisent pas à le remettre en cause. Les deux restent au-dessus de 0,75, le seuil d'exploitabilité. Surtout, on ne change pas un modèle en production à la va-vite avant une échéance. La correction existe, elle tourne déjà côté entreprise, et notre chaîne sait l'accueillir : il suffit de laisser la promotion choisir entre les candidats. »

Si le jury insiste :
- **Touzac** : l'écart est réel (0,82 contre 0,88). On le reconnaît, on ne le minimise pas.
- **Ce qu'il faudrait changer** : la promotion compare déjà des scores ; il faudrait lui faire comparer LightGBM et le stacking, retenir le meilleur par centrale et le servir via l'API. C'est un petit changement, déjà validé en entreprise.
- **Pourquoi ne pas l'avoir fait** : à deux semaines de la soutenance, on préfère une démo stable à un changement de modèle peu validé (une seule période de test, automne mauvais pour tous les modèles).

À ne pas dire : « le stacking est plus robuste ». Rien dans nos résultats ne le montre.

**Autres limites assumées** (à dire si le temps le permet, sinon garder pour les questions) : certificat auto-signé, `git push` / `dvc push` manuels après le réentraînement, pas d'Alertmanager.

---

## 15 — Et demain ?

**À l'écran** (court, pas technique)
- **Dans 5 ans** — un outil quotidien des exploitants : un écran qui dit, centrale par centrale, l'eau attendue et la production possible, sur toutes les centrales et à plusieurs jours
- **En vrai** — décider quand ouvrir ou fermer une turbine, et fixer l'engagement de production de la veille. Il manque une **couche de décision** (seuils par turbine, débit réservé) qui demande l'expertise des exploitants. Ce n'est pas à nous de l'inventer en codant.
- **Pourquoi ces choix** — on a choisi des outils standard, légers, et remplaçables un par un. Le jour où il faut passer à l'échelle, on change une brique sans tout reconstruire.

**À dire** — L'IA générative ou des agents ? Volontairement hors périmètre : en ajouter un demanderait sa propre évaluation et son propre suivi.

---

## 16 — Équipe & merci

**À l'écran**
- **Sébastien** — données (Hub'Eau, Météo-France), MLflow + registre, Airflow, réentraînement automatique
- **Xavier** — cadrage, Docker, API, nginx / MinIO, CI, sécurité, Kubernetes, monitoring, dérive, documentation
- Dépôt : `github.com/Sebastien6631/Hydro-Project` (QR code)
- **Merci — questions ?**

---

## Questions probables du jury — réponses courtes

| Question | Réponse |
|---|---|
| Pourquoi garder le stacking s'il est moins bon ? | Selon la centrale, c'est le stacking ou LightGBM seul qui gagne (constat de Sébastien sur la version de production). On a généralisé le stacking à tort, alors que sur nos 2 centrales c'est LightGBM qui l'emporte. Les scores décomposés l'ont montré. **Déjà corrigé dans la production de l'entreprise** : le meilleur des 2 est choisi centrale par centrale et c'est lui qui est servi. À porter dans notre projet (pas encore fait au 19/10). |
| Les outils d'admin (MLflow, Airflow, MinIO) sont-ils protégés ? | **Non, et c'est assumé** : ce sont des outils internes qui tournent sur nos machines, pas exposés à Internet. Seule l'API, pensée pour être consommée, a sa clé. Avant toute exposition réelle : (1) ports d'admin limités à la machine (`127.0.0.1`), (2) login centralisé sur nginx devant MLflow et Airflow, (3) mots de passe par défaut de MinIO et Grafana changés. **Solution identifiée et chiffrée** (~30 min), mais inutile en local. |
| Pourquoi pas de vrai déploiement cloud ? | Pas de crédits, pas d'utilisateur réel. Équivalents managés documentés service par service, sans changement de code. |
| Comment rejouer exactement un résultat ? | `git checkout <tag> && dvc pull` + graines fixées + `source="frozen"` dans l'API. |
| Et si Hub'Eau tombe ? | Le check bloquant du DAG arrête la prévision ; l'API peut rejouer le dernier état connu (`frozen`) ; l'alerte part. |
| Pourquoi la prévision date du 9 septembre ? | C'est le mode **figé** de l'API : elle rejoue le dernier état connu des données, pour que la démo donne toujours le même résultat (mêmes chiffres le 20/10 matin et soir). Le mode **temps réel** existe : c'est celui qu'Airflow fait tourner toutes les heures (`hydro_predict`, run vert du 20/10). |
| Pourquoi deux débits, `q_stacking` et `q_entrant` ? | `q_stacking` est le débit prévu par le modèle **à la station Hub'Eau**, en amont. `q_entrant` est ce débit ramené **à la turbine** : multiplié par le facteur propre à la centrale (0,898 pour Touzac, d'où 26,19 → 23,52 m³/s) et décalé du temps de transit station → turbine (5 h en automne pour Touzac, il varie selon la saison). |
| Pourquoi KGE et pas RMSE ? | Standard hydrologique, et il dit *pourquoi* c'est mauvais (corrélation / variabilité / biais). On a aussi le RMSE par pas d'horizon. |
| Pourquoi le seuil de dérive à 40 % ? | On alerte sur une dérive d'ensemble, pas sur une colonne isolée. Le seuil est repris de notre prototype et reste réglable. C'est un signal de surveillance qui n'arrête jamais le pipeline. ⚠️ Pas de justification chiffrée dans le dépôt : à préparer à deux. |
| Le réentraînement est-il vraiment automatique ? | Oui jusqu'à la promotion. Le `git push` / `dvc push` reste manuel : décision volontaire (pas de token en écriture dans le conteneur). |
