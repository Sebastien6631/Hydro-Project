#!/bin/sh
# Génère un certificat auto-signé (localhost) dans le volume partagé avec
# nginx, une seule fois (idempotent -- ne régénère pas si déjà présent, pour
# ne pas invalider un certificat qu'un navigateur aurait déjà accepté).
# Lancé avec l'image projet_hydro:latest (openssl déjà présent, cf. compose).
set -e

if [ -f /certs/nginx.crt ] && [ -f /certs/nginx.key ]; then
  echo "certificat déjà présent, rien à faire"
  exit 0
fi

openssl req -x509 -nodes -newkey rsa:2048 -days 825 \
  -keyout /certs/nginx.key -out /certs/nginx.crt \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

echo "certificat auto-signé généré (825 jours, CN=localhost)"
