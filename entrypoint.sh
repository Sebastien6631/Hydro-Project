#!/bin/sh
# Configure DVC (token DagsHub) et l'identité git, puis exécute la commande.
set -e

if [ -n "$DAGSHUB_TOKEN" ] && [ -f .dvc/config ]; then
  # `auth basic` explicite : sans ça DagsHub répond 401 et DVC affiche
  # « missing cache files » au lieu d'une erreur d'authentification.
  dvc remote modify dagshub --local auth basic                   2>/dev/null || true
  dvc remote modify dagshub --local user "${DAGSHUB_USER:-token}" 2>/dev/null || true
  dvc remote modify dagshub --local password "$DAGSHUB_TOKEN"     2>/dev/null || true
fi

git config --global --get user.email >/dev/null 2>&1 || \
  git config --global user.email "${GIT_AUTHOR_EMAIL:-previ-r2d2@localhost}"
git config --global --get user.name  >/dev/null 2>&1 || \
  git config --global user.name  "${GIT_AUTHOR_NAME:-previ-r2d2}"
git config --global --add safe.directory /app 2>/dev/null || true

exec "$@"
