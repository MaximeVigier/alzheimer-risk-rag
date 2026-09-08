"""
Reranking — affine le classement des chunks retrouvés par le retrieval hybride
en utilisant un cross-encoder (bien plus précis qu'un simple score de similarité
bi-encoder, car il traite (query, document) ensemble plutôt que séparément).

Pipeline : retrieval hybride récupère un top-N large (ex. 20) -> reranking -> top-k final (ex. 5).

Usage (import) :
    from rerank import rerank
    top5 = rerank(query, candidates, k=5)

Usage (CLI) :
    python src/rerank.py --strategy fixed --query "..." --candidate-pool 20 --k 5
"""
from __future__ import annotations

import argparse

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Modèle généraliste léger (CPU-friendly). Alternative bio-spécifique possible
# (ex. cross-encoder entraîné sur BEIR-bio) si le temps le permet — à noter en limite.

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank(query: str, candidates: list[dict], k: int = 5) -> list[dict]:
    """candidates: liste de chunks (dicts avec au moins 'text'). Retourne les k meilleurs,
    triés par score de reranking décroissant, avec un champ 'rerank_score' ajouté."""
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [{**c, "rerank_score": float(s)} for c, s in scored[:k]]


def main():
    from retrieval import HybridRetriever

    parser = argparse.ArgumentParser(description="Test manuel du reranking")
    parser.add_argument("--strategy", choices=["fixed", "semantic"], default="fixed")
    parser.add_argument("--query", required=True)
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    retriever = HybridRetriever(strategy=args.strategy)
    candidates = retriever.search(args.query, k=args.candidate_pool, mode="hybrid")
    reranked = rerank(args.query, candidates, k=args.k)

    print(f"\nQuery: {args.query}\n")
    for i, r in enumerate(reranked, 1):
        print(f"[{i}] rerank_score={r['rerank_score']:.4f} pmid={r['pmid']} ({r['year']}) — {r['title'][:90]}")


if __name__ == "__main__":
    main()
