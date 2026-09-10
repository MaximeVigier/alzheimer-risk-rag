"""
API HTTP pour le RAG Alzheimer — expose le pipeline (retrieval + reranking + génération)
via FastAPI, pour une intégration facile (front web, autre service, démo publique).

Usage:
    uvicorn src.api:app --reload --port 8000
    (ou depuis src/: uvicorn api:app --reload --port 8000)

Endpoints:
    GET  /health        -> statut du service + stratégies chargées
    POST /ask            -> {"query": "...", "strategy": "fixed"} -> réponse sourcée
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generate import OLLAMA_MODEL, answer_question
from retrieval import HybridRetriever

logger = logging.getLogger("rag_api")

app = FastAPI(
    title="Alzheimer Risk Factors RAG API",
    description="Question-réponse sourcée sur la littérature scientifique des facteurs de "
                "risque de la maladie d'Alzheimer (corpus PubMed, RAG 100% local).",
    version="1.0.0",
)

# Retrievers chargés paresseusement et mis en cache par stratégie : le chargement (embeddings,
# ChromaDB, BM25) prend quelques secondes, on ne veut pas le refaire à chaque requête.
_retrievers: dict[str, HybridRetriever] = {}


def _get_retriever(strategy: str) -> HybridRetriever:
    if strategy not in _retrievers:
        logger.info("Chargement du retriever (strategy=%s)...", strategy)
        _retrievers[strategy] = HybridRetriever(strategy=strategy)
    return _retrievers[strategy]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Question sur les facteurs de risque Alzheimer")
    strategy: str = Field("fixed", pattern="^(fixed|semantic)$", description="Stratégie de chunking")
    model: str = Field(OLLAMA_MODEL, description="Modèle Ollama utilisé pour la génération")


class Source(BaseModel):
    pmid: str
    title: str
    year: str | None = None
    url: str


class AskResponse(BaseModel):
    answer: str
    sources_cited: list[str]
    sources: list[Source]
    refused: bool


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "strategies_loaded": list(_retrievers.keys()),
        "generation_model": OLLAMA_MODEL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        retriever = _get_retriever(request.strategy)
    except Exception as exc:  # index absent, chemin invalide, etc.
        logger.exception("Echec de chargement du retriever (strategy=%s)", request.strategy)
        raise HTTPException(status_code=503, detail=f"Index indisponible pour la stratégie "
                                                      f"'{request.strategy}': {exc}") from exc

    try:
        result = answer_question(request.query, strategy=request.strategy, retriever=retriever,
                                  model=request.model)
    except Exception as exc:  # Ollama down, modèle absent, etc.
        logger.exception("Echec de génération")
        raise HTTPException(status_code=502, detail=f"Echec de la génération (Ollama disponible ? "
                                                      f"modèle '{request.model}' installé ?) : {exc}") from exc

    return AskResponse(
        answer=result["answer"],
        sources_cited=result["sources_cited"],
        sources=[
            Source(pmid=s["pmid"], title=s["title"], year=s.get("year"), url=s["url"])
            for s in result["sources_retrieved"]
        ],
        refused=result["refused"],
    )
