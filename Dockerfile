# Image projet_hydro — cœur ML + DVC + tests. Le code est bind-monté au
# runtime (docker-compose) ; ici on installe juste l'environnement.
FROM python:3.11-slim

# PIP_DEFAULT_TIMEOUT : torch fait 192 Mo, le delai pip par defaut (15 s de
# silence) lache sur une connexion lente -- build echoue vu le 2026-09-15.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=300

# git + dvc : promotion.py les appelle en sous-process. libgomp1 : runtime
# OpenMP de LightGBM. curl : healthchecks des phases suivantes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch CPU-only (gros paquet) sur un layer séparé pour le cache.
RUN pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# Dépendances + package. `.[dev]` ajoute pytest.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[dev]"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
CMD ["python", "-m", "pytest", "-q"]
