"""
Retrieval hybride : combine recherche dense (embeddings ChromaDB) et sparse (BM25),
fusionnés par Reciprocal Rank Fusion (RRF).

Usage (import) :
    from retrieval import HybridRetriever
    retriever = HybridRetriever(strategy="fixed")
    results = retriever.search("does APOE genotype increase Alzheimer's risk?", k=5)

Usage (CLI, pour test manuel) :
    python src/retrieval.py --strategy fixed --query "does sleep quality affect Alzheimer's risk?"
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

CHROMA_DIR = "data/chroma"
CHUNKS_DIR = "data/processed"
BM25_CACHE_DIR = "data/bm25_cache"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    """Retrieval dense (ChromaDB) + sparse (BM25) fusionnés par RRF, avec fallback dense-only."""

    def __init__(self, strategy: str = "fixed", chroma_dir: str = CHROMA_DIR,
                 chunks_dir: str = CHUNKS_DIR, bm25_cache_dir: str = BM25_CACHE_DIR,
                 rrf_k: int = 60):
        self.strategy = strategy
        self.rrf_k = rrf_k

        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_collection(f"alzheimer_risk_{strategy}")

        with open(Path(chunks_dir) / f"chunks_{strategy}.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}

        self.bm25 = self._load_or_build_bm25(bm25_cache_dir)

    def _load_or_build_bm25(self, cache_dir: str) -> BM25Okapi:
        cache_path = Path(cache_dir) / f"bm25_{self.strategy}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        bm25 = BM25Okapi(tokenized)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(bm25, f)
        return bm25

    def _dense_search(self, query: str, top_n: int) -> list[str]:
        """Retourne une liste de chunk_ids classée par similarité dense (meilleur d'abord)."""
        model = _get_embed_model()
        query_emb = model.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(query_embeddings=query_emb, n_results=top_n)
        return result["ids"][0]

    def _sparse_search(self, query: str, top_n: int) -> list[str]:
        """Retourne une liste de chunk_ids classée par score BM25 (meilleur d'abord)."""
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [self.chunks[i]["chunk_id"] for i in ranked_indices]

    def search(self, query: str, k: int = 5, mode: str = "hybrid", candidate_pool: int = 30) -> list[dict]:
        """
        mode: "dense" | "sparse" | "hybrid" (RRF sur dense+sparse)
        Retourne les k meilleurs chunks (dicts complets, avec un champ 'score' ajouté).
        """
        if mode == "dense":
            ids = self._dense_search(query, k)
            return [{**self.chunk_by_id[cid], "score": None} for cid in ids]

        if mode == "sparse":
            ids = self._sparse_search(query, k)
            return [{**self.chunk_by_id[cid], "score": None} for cid in ids]

        # hybrid : Reciprocal Rank Fusion
        dense_ids = self._dense_search(query, candidate_pool)
        sparse_ids = self._sparse_search(query, candidate_pool)

        rrf_scores: dict[str, float] = {}
        for rank, cid in enumerate(dense_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, cid in enumerate(sparse_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{**self.chunk_by_id[cid], "score": score} for cid, score in ranked]


def main():
    parser = argparse.ArgumentParser(description="Test manuel du retrieval hybride")
    parser.add_argument("--strategy", choices=["fixed", "semantic"], default="fixed")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    args = parser.parse_args()

    retriever = HybridRetriever(strategy=args.strategy)
    results = retriever.search(args.query, k=args.k, mode=args.mode)

    print(f"\nQuery: {args.query}")
    print(f"Mode: {args.mode} | Strategy: {args.strategy}\n")
    for i, r in enumerate(results, 1):
        score_str = f"{r['score']:.4f}" if r["score"] is not None else "n/a"
        print(f"[{i}] score={score_str} pmid={r['pmid']} ({r['year']}) — {r['title'][:90]}")
        print(f"    categories: {r['risk_categories']}")
        print(f"    {r['text'][:200]}...\n")


if __name__ == "__main__":
    main()
