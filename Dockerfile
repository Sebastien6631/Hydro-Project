# Image previ-R2-D2 — cœur ML + DVC + tests. Le code est bind-monté au
# runtime (docker-compose) ; ici on installe juste l'environnement.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

# git + dvc : promotion.py les appelle en sous-process. libgomp1 : runtime
# OpenMP de LightGBM. curl : healthchecks des phases suivantes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch CPU-only (gros paquet) sur un layer séparé pour le cache.
RUN pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# Dépendances + package. Les extras GIS (pysheds/rasterio/geopandas) ne sont
# PAS installés : lazy-import only, jamais exercés pour les 2 centrales
# (shapefile connu). `.[dev]` ajoute pytest.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[dev]"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
CMD ["python", "-m", "pytest", "-q"]
