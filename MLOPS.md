# MLOps — correspondance grille jury

> Documentation technique finale (phase 4.5). Fait le lien entre chaque
> critère de la grille d'évaluation (cf. `.claude/skills/hydro-mlops/SKILL.md`
> §7) et son implémentation concrète dans ce dépôt — pour vérifier la
> couverture rapidement, en soutenance ou en relecture.

| Critère jury | Implémenté par | Détail |
|---|---|---|
| Proposition claire + cas d'usage | `docs/cadrage/README.md` §1-2 | Prévision de débit horaire h8 pour 2 centrales hydroélectriques, cas d'usage marché J+1 et anticipation hydraulique |
| Doc technique complète (README + slides) | `README.md`, `ARCHITECTURE.md`, `MLOPS.md` (ce fichier), `docs/cadrage/README.md` | Une section par phase dans le README (commandes), architecture consolidée dans `ARCHITECTURE.md`. Slides de soutenance : à faire (T7, cf. skill §8) |
| Données propres / traitées / versionnées | `docs/cadrage/README.md` §4, `preprocessing/data_preparation/validation.py` | Contrat de données (erreurs bloquantes / avertissements), stage DVC `validate` intercalé avant l'entraînement, historique 3,8 ans homogène |
| Sélection & validation du modèle justifiées | `docs/cadrage/README.md` §5-6 | Hybride LightGBM+BiLSTM+stacking justifié (régimes hydrologiques mixtes), KGE défini et décomposé, promotion conditionnelle (candidat vs production) |
| Résultats reproductibles | `source="frozen"` (API), DVC + tags git par version de modèle | `git checkout <tag> && dvc pull` restaure un état exact ; l'API rejoue une prévision 100% reproductible en démo |
| Code versionné et organisé sur GitHub | `github.com/Sebastien6631/Hydro-Project`, structure `src/projet_hydro/` | Historique de branches `phaseN/<sujet>` → PR → `dev` → `main`, un commit = un changement logique |
| Infra conteneurisée + orchestrée | `docker-compose.yml` (phase 1), `airflow/dags/` (phase 3.1) | Tous les services en Docker Compose ; Airflow orchestre `hydro_predict` (horaire) et `hydro_train` (hebdo) via les mêmes scripts que DVC |
| Serving opérationnel | `src/projet_hydro/serving/api.py` (phase 1), sécurisé phase 3.3 | FastAPI : `/health`, `/models`, `/predict` — testé en vrai (curl réel), pas que mocké |
| Monitoring système et ML | Prometheus + Grafana (phase 4.1) | `/metrics` : requêtes/latence (système) + KGE par centrale (ML), dashboard provisionné, 3 règles d'alerte |
| Détection de dérive | Evidently (phase 4.2) | Test K-S par colonne, seuil 40%, `cron/scripts/check-drift.py`, alerte `DataDrift` — vérifié en vrai (45% de dérive détectée sur les 2 centrales, cohérent avec le changement de schéma météo) |
| Infra scalable (Kubernetes) | `infrastructure/helm/projet-hydro/` (phase 3.5) | Chart Helm (Deployment+Service+Ingress+HPA) pour l'API, validé par `helm lint`/`template` |
| Sécurité des APIs | Phase 3.3 + phase 3.6 | Clé API optionnelle, logs structurés, rate-limit+timeouts nginx, HTTPS (certificat auto-signé, limite documentée) |
| RGPD & éthique | `docs/cadrage/README.md` §4.5 | Données exclusivement publiques et environnementales, aucune donnée personnelle ; seuls secrets = jetons d'accès API, hors dépôt |
| Stratégie de déploiement documentée | `README.md` §Déploiement cloud (phase 4.4) | Mapping service compose → équivalent managé, sans changement de code ; non exécuté (pas de crédits), même raisonnement que BentoML écarté |
| Collaboration d'équipe (historique Git, répartition) | Workflow `phaseN/<sujet>` → PR → `dev` | Deux contributeurs (Xavier, Sébastien), tâches réparties et tracées dans le tableau de suivi du skill `hydro-mlops` |
| Suivi d'expériences documenté (MLflow) | MLflow + Model Registry (phase 2.1-2.2) | Run par entraînement (params, métriques KGE, artefacts sur MinIO), alias `@production` pour le modèle promu |
| Plan de maintenance/évolution esquissé | `hydro_train.py` (phase 4.3), §7 de ce fichier | Réentraînement hebdomadaire automatisé (`train --promote`), décision de push manuel documentée comme limite assumée, pas un oubli |

## Choix technologiques écartés (justifiés)

| Besoin | Retenu | Écarté — pourquoi |
|---|---|---|
| Orchestration | Airflow | Prefect — essayé puis abandonné (Airflow est au programme, connu de l'équipe) |
| Serving | FastAPI seul | BentoML — construit (phase 3.4) puis retiré sur décision du tuteur : valeur uniquement si plusieurs modèles dans des images séparées, non pertinent ici (une seule image, 2 centrales, 1 horizon) |
| CI | GitHub Actions | Jenkins — plus lourd à héberger pour un projet à deux |
| Cloud | Documenté, non déployé | Pas de crédits, pas de justification à payer pour zéro utilisateur réel |

Détail complet dans `.claude/skills/hydro-mlops/SKILL.md` §5 (stack cible)
et le tableau de suivi §8 (une ligne par tâche, avec PR et vérifications).
