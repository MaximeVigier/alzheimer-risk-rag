# Image légère : dépendances runtime API seulement (voir requirements-api.txt).
# L'index vectoriel (data/) est monté en volume, pas embarqué dans l'image -- il est
# construit une fois en local (ingestion + chunking + embed) et réutilisé tel quel,
# ce qui évite de reconstruire l'image à chaque mise à jour du corpus.
FROM python:3.12-slim

WORKDIR /app

# Dépendances système minimales requises par sentence-transformers/torch (CPU only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/

EXPOSE 8000

# Le modèle d'embedding (all-MiniLM-L6-v2) et le cross-encoder de reranking sont
# téléchargés au premier appel (cache HuggingFace dans le conteneur) -- premier /ask
# plus lent, les suivants rapides.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
